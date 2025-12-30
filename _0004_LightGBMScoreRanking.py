# _0004_LightGBMScoreRanking.py — same LGBM as ratios version, but using
# avg / slope / volatility of composite score from AnchorRanking

from __future__ import annotations
from pathlib import Path
from datetime import date, timedelta, datetime
import os, math, re
import numpy as np
import polars as pl

# ---------------- paths & config ----------------
BASE    = Path(".")
AR_DIR  = BASE / "AnchorRanking"                    # from _0003_AnchorRanking.py
RET_DS  = BASE / "FilteredRawData" / "Returns_ds"   # same as ratios version
OUT_DIR = BASE / "LightGBMScoreRanking"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# LightGBM params (same defaults as ratios version)
LGB_NUM_LEAVES    = int(os.environ.get("LGB_NUM_LEAVES", "31"))
LGB_MAX_DEPTH     = int(os.environ.get("LGB_MAX_DEPTH",  "6"))
LGB_LR            = float(os.environ.get("LGB_LR",       "0.05"))
LGB_N_EST         = int(os.environ.get("LGB_N_EST",      "300"))
LGB_MIN_LEAF      = int(os.environ.get("LGB_MIN_LEAF",   "200"))
LGB_FEAT_FRAC     = float(os.environ.get("LGB_FEAT_FRAC","0.7"))
LGB_BAG_FRAC      = float(os.environ.get("LGB_BAG_FRAC", "0.7"))
LGB_L2            = float(os.environ.get("LGB_L2",       "1.0"))
LGB_THREADS       = int(os.environ.get("LGB_THREADS",    str(os.cpu_count() or 4)))
EARLY_STOP_ROUNDS = int(os.environ.get("EARLY_STOP",     "100"))

# Rolling window & weighting (identical logic)
TRAIN_YEARS       = float(os.environ.get("TRAIN_YEARS", "3"))
HALF_LIFE_YEARS   = float(os.environ.get("HALF_LIFE_YEARS", "1.5"))
VAL_MONTHS        = int(os.environ.get("VAL_MONTHS", "6"))

# Blend weights for (3m, 6m, 12m)
W3, W6, W12       = (float(os.environ.get("BLEND_W3M","0.2")),
                     float(os.environ.get("BLEND_6M","0.3")),
                     float(os.environ.get("BLEND_12M","0.5")))

# Score feature window: number of anchors in rolling composite
SCORE_WIN_ANCHORS = int(os.environ.get("SCORE_WIN_ANCHORS", "12"))
MIN_FRAC          = float(os.environ.get("SCORE_MIN_FRAC", "0.60"))

def MIN_OBS(n: int) -> int:
    return max(3, int(math.ceil(n * MIN_FRAC)))

FEAT_COLS = ["avg_score", "slope_score", "vol_dscore"]

# --------------- LightGBM import (fail-friendly) ---------------
try:
    import lightgbm as lgb
except Exception as e:
    raise RuntimeError("LightGBM is required. Please `pip install lightgbm`.") from e

# ---------------- score feature construction from AnchorRanking ----------------
def list_anchor_files() -> list[Path]:
    files: list[Path] = []
    for ydir in sorted(AR_DIR.glob("year=*")):
        for f in sorted(ydir.glob("*.parquet")):
            # expect YYYYMMDD in filename
            if re.search(r"\d{8}", f.name):
                files.append(f)
    return files

def parse_date_tag(p: Path) -> str:
    m = re.search(r"(\d{8})", p.name)
    if not m:
        raise ValueError(f"Cannot find 8-digit date in filename: {p.name}")
    return m.group(1)

def compute_window_scores(files_win: list[Path]) -> tuple[pl.DataFrame, str | None]:
    """
    For a rolling window of anchor files, compute avg_score, slope_score, vol_dscore
    of cross-sectionally standardized composite score_z for each permno.
    """
    n = len(files_win)
    if n == 0:
        return pl.DataFrame(), None
    min_obs = MIN_OBS(n)

    panels: list[pl.DataFrame] = []
    for t, f in enumerate(files_win):
        df = pl.read_parquet(str(f), columns=["permno", "score"])
        if df.is_empty():
            continue
        df = df.drop_nulls(subset=["permno", "score"])

        # robust standardization of "score" within this anchor => score_z
        med = float(df.select(pl.col("score").median()).item())
        mad = float(df.select((pl.col("score") - med).abs().median()).item())
        scale = 1.4826 * mad if (mad is not None and math.isfinite(mad)) else None
        if not scale or scale <= 0:
            std = float(df.select(pl.col("score").std()).item())
            scale = std if (std and math.isfinite(std) and std > 0) else 1.0

        df = df.with_columns(
            ((pl.col("score") - med) / scale).alias("score_z"),
            pl.lit(float(t)).alias("t"),
            pl.col("permno").cast(pl.Int64),
        )
        panels.append(df.select("permno", "score_z", "t"))

    if not panels:
        return pl.DataFrame(), None

    panel = (
        pl.concat(panels, how="vertical_relaxed")
          .with_columns(
              pl.col("score_z").cast(pl.Float64),
              pl.col("t").cast(pl.Float64),
          )
    )

    # OLS slope of score_z vs t, plus average level
    sums = (
        panel.group_by("permno").agg(
            pl.count().alias("k"),
            pl.col("t").sum().alias("sum_t"),
            (pl.col("t") ** 2).sum().alias("sum_t2"),
            pl.col("score_z").sum().alias("sum_s"),
            (pl.col("score_z") * pl.col("t")).sum().alias("sum_st"),
            pl.col("score_z").mean().alias("avg_score"),
        )
        .with_columns(
            denom = pl.col("k") * pl.col("sum_t2") - (pl.col("sum_t") ** 2)
        )
        .with_columns(
            slope_score = pl.when(
                (pl.col("denom") > 0) & (pl.col("k") >= min_obs)
            )
            .then(
                (pl.col("k") * pl.col("sum_st") - pl.col("sum_t") * pl.col("sum_s"))
                / pl.col("denom")
            )
            .otherwise(None),
            avg_score = pl.when(pl.col("k") >= max(1, min_obs))
            .then(pl.col("avg_score"))
            .otherwise(None),
        )
        .select(["permno", "slope_score", "avg_score", "k"])
    )

    # Volatility of Δscore_z (lower is better, but we do not invert here)
    panel = (
        panel.sort(["permno", "t"])
              .with_columns(
                  pl.col("score_z").shift(1).over("permno").alias("score_z_lag")
              )
              .with_columns(
                  (pl.col("score_z") - pl.col("score_z_lag")).alias("dscore_z")
              )
    )

    diffs = (
        panel.group_by("permno").agg(
            pl.col("dscore_z").drop_nulls().std().alias("vol_dscore"),
            pl.col("dscore_z").drop_nulls().count().alias("nd"),
        )
        .with_columns(
            pl.when(pl.col("nd") >= max(1, min_obs - 1))
             .then(pl.col("vol_dscore"))
             .otherwise(None)
             .alias("vol_dscore")
        )
        .select(["permno", "vol_dscore"])
    )

    xsec = sums.join(diffs, on="permno", how="left")
    if xsec.is_empty():
        return pl.DataFrame(), None

    last = files_win[-1]
    date_tag = parse_date_tag(last)
    xsec = xsec.select(["permno", "avg_score", "slope_score", "vol_dscore"])
    return xsec, date_tag

def clean_float_cols(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    # Coerce non-finite to null (LightGBM can handle NaN as missing)
    return df.with_columns(
        [
            pl.when(pl.col(c).is_finite())
              .then(pl.col(c))
              .otherwise(None)
              .alias(c)
            for c in cols
        ]
    )

def build_score_features() -> pl.DataFrame:
    files = list_anchor_files()
    if len(files) < SCORE_WIN_ANCHORS:
        print("Not enough anchor files in AnchorRanking/ to build score features.")
        return pl.DataFrame(
            schema={
                "date": pl.Date,
                "permno": pl.Int64,
                "avg_score": pl.Float64,
                "slope_score": pl.Float64,
                "vol_dscore": pl.Float64,
            }
        )

    files = sorted(files, key=parse_date_tag)
    frames: list[pl.DataFrame] = []

    for i in range(SCORE_WIN_ANCHORS - 1, len(files)):
        win = files[i - SCORE_WIN_ANCHORS + 1 : i + 1]
        xsec, date_tag = compute_window_scores(win)
        if date_tag is None or xsec.is_empty():
            continue
        dt = datetime.strptime(date_tag, "%Y%m%d").date()
        xsec = xsec.with_columns(
            pl.lit(dt).alias("date").cast(pl.Date),
            pl.col("permno").cast(pl.Int64),
        )
        frames.append(xsec)

    if not frames:
        print("No score features were created.")
        return pl.DataFrame(
            schema={
                "date": pl.Date,
                "permno": pl.Int64,
                "avg_score": pl.Float64,
                "slope_score": pl.Float64,
                "vol_dscore": pl.Float64,
            }
        )

    df = pl.concat(frames, how="vertical_relaxed")
    df = clean_float_cols(df, FEAT_COLS)
    return df

# ---------------- returns / labels (unchanged) ----------------
def scan_year_paths(ds_dir: Path, years) -> list[str]:
    paths = []
    for y in years:
        p = ds_dir / f"year={y}"
        if p.exists():
            paths += [str(pp) for pp in p.glob("*.parquet")]
    return paths

def load_returns_block(dmin: date, dt: date, permnos: list[int]) -> pl.DataFrame:
    years = range(dmin.year, dt.year + 1)
    files = scan_year_paths(RET_DS, years)
    if not files:
        return pl.DataFrame(schema={"permno": pl.Int64, "DATE": pl.Date, "RET": pl.Float64})
    ret = (
        pl.scan_parquet(files)
          .select(
              permno = pl.col("PERMNO").cast(pl.Int64),
              DATE   = pl.coalesce(
                           [
                               pl.col("DATE").cast(pl.Date, strict=False),
                               pl.col("DATE")
                                 .cast(pl.Utf8)
                                 .str.strptime(pl.Date, "%Y-%m-%d", strict=False),
                               pl.col("DATE")
                                 .cast(pl.Utf8)
                                 .str.strptime(pl.Date, "%Y%m%d", strict=False),
                           ]
                       ),
              RET    = pl.col("RET").cast(pl.Float64),
          )
          .filter(
              pl.col("permno").is_in(pl.lit(permnos))
              & (pl.col("DATE") > pl.lit(dmin))
              & (pl.col("DATE") <= pl.lit(dt))
          )
          .with_columns(pl.col("RET").fill_null(0.0))
          .collect(engine="streaming")
    )
    if ret.is_empty():
        return ret
    ret = ret.sort(["permno", "DATE"])
    ret = ret.with_columns(
        pl.col("RET").log1p().over("permno").cum_sum().alias("cumlog")
    )
    return ret

def build_labels_horizon(
    feats_train: pl.DataFrame, ret_df: pl.DataFrame, dt: date, horizon: str
) -> pl.DataFrame:
    # horizon: "3mo", "6mo", "1y"
    if feats_train.is_empty() or ret_df.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "permno": pl.Int64, "y": pl.Float64})

    cal = ret_df.select(pl.col("DATE")).unique().sort("DATE")

    anchors = (
        feats_train.select("date")
                   .unique()
                   .sort("date")
                   .with_columns(
                       pl.col("date")
                         .cast(pl.Datetime)
                         .dt.offset_by(horizon)
                         .cast(pl.Date)
                         .alias("end_target")
                   )
    )
    anchors = (
        anchors.join_asof(
                    cal,
                    left_on="end_target",
                    right_on="DATE",
                    strategy="backward",
                )
               .rename({"DATE": "end_date"})
               .drop("end_target")
               .drop_nulls(subset=["end_date"])
               .filter(pl.col("end_date") <= pl.lit(dt))  # no look-ahead
    )
    if anchors.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "permno": pl.Int64, "y": pl.Float64})

    A = (
        feats_train.select(["date", "permno"])
                   .unique()
                   .join(anchors, on="date", how="inner")
    )

    start = (
        A.sort(["permno", "date"])
         .join_asof(
             ret_df,
             left_on="date",
             right_on="DATE",
             by="permno",
             strategy="backward",
         )
         .rename({"cumlog": "cum_start"})
         .select("date", "permno", "end_date", "cum_start")
    )
    end = (
        A.sort(["permno", "end_date"])
         .join_asof(
             ret_df,
             left_on="end_date",
             right_on="DATE",
             by="permno",
             strategy="backward",
         )
         .rename({"cumlog": "cum_end"})
         .select("date", "permno", "end_date", "cum_end")
    )

    lbl = (
        start.join(end, on=["date", "permno", "end_date"], how="inner")
             .with_columns(
                 ((pl.col("cum_end") - pl.col("cum_start")).exp() - 1.0).alias("y")
             )
             .select(["date", "permno", "y"])
             .drop_nulls(subset=["y"])
    )
    return lbl

def time_decay_weights(dates: np.ndarray, dt: date, half_life_years: float) -> np.ndarray:
    days = (np.datetime64(dt) - dates.astype("datetime64[D]")).astype("timedelta64[D]").astype(np.float64)
    age_years = days / 365.2425
    w = np.exp(-math.log(2.0) * (age_years / max(half_life_years, 1e-9)))
    w = np.clip(w, 1e-8, None)
    return w

def lgb_train_predict(
    train_df: pl.DataFrame, pred_df: pl.DataFrame, dt: date, booster_prev=None
) -> tuple[np.ndarray, object]:
    fcols = [c for c in train_df.columns if c not in ("date", "permno", "y")]
    if not fcols or train_df.is_empty() or pred_df.is_empty():
        return np.array([]), booster_prev

    X = train_df.select(fcols).to_numpy().astype(np.float32)
    y = train_df["y"].to_numpy().astype(np.float32)
    Xp = pred_df.select(fcols).to_numpy().astype(np.float32)

    w = time_decay_weights(train_df["date"].to_numpy(), dt, HALF_LIFE_YEARS).astype(
        np.float32
    )

    cutoff = (
        np.datetime64(dt)
        - np.timedelta64(30 * VAL_MONTHS, "D")
    ).astype("datetime64[D]")
    mask_val = train_df["date"].to_numpy().astype("datetime64[D]") >= cutoff

    if mask_val.any() and (~mask_val).any():
        dtrain = lgb.Dataset(
            X[~mask_val],
            y[~mask_val],
            weight=w[~mask_val],
            feature_name=fcols,
            free_raw_data=False,
        )
        dvalid = lgb.Dataset(
            X[mask_val],
            y[mask_val],
            weight=w[mask_val],
            reference=dtrain,
            free_raw_data=False,
        )
        valid_sets = [dtrain, dvalid]
        valid_names = ["train", "valid"]
        do_early_stop = True
    else:
        dtrain = lgb.Dataset(
            X,
            y,
            weight=w,
            feature_name=fcols,
            free_raw_data=False,
        )
        valid_sets = [dtrain]
        valid_names = ["train"]
        do_early_stop = False

    params = {
        "objective": "regression",
        "metric": "l2",
        "num_leaves": LGB_NUM_LEAVES,
        "max_depth": LGB_MAX_DEPTH,
        "learning_rate": LGB_LR,
        "feature_fraction": LGB_FEAT_FRAC,
        "bagging_fraction": LGB_BAG_FRAC,
        "bagging_freq": 1,
        "lambda_l2": LGB_L2,
        "min_data_in_leaf": LGB_MIN_LEAF,
        "verbosity": -1,
        "num_threads": LGB_THREADS,
    }

    callbacks = []
    if do_early_stop and EARLY_STOP_ROUNDS > 0:
        callbacks.append(lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False))
    callbacks.append(lgb.log_evaluation(0))

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=LGB_N_EST,
        valid_sets=valid_sets,
        valid_names=valid_names,
        init_model=booster_prev,
        keep_training_booster=True,
        callbacks=callbacks,
    )
    yhat = booster.predict(
        Xp, num_iteration=booster.best_iteration or booster.current_iteration()
    )
    return yhat.astype(np.float64), booster

# ---------------- main loop ----------------
def main():
    # Build score-based features once (avg / slope / vol over last SCORE_WIN_ANCHORS anchors)
    score_feats = build_score_features()
    if score_feats.is_empty():
        print("No score features available; aborting.")
        return

    # Extract available anchor dates from score_feats
    anchors = sorted(score_feats["date"].unique().to_list())
    if len(anchors) < 12:
        print("Not enough anchor dates in score features to run.")
        return

    win_days = int(round(365.2425 * TRAIN_YEARS))
    one_year = timedelta(days=365)
    first_dt = anchors[0]

    # require TRAIN_YEARS history + 12m matured labels
    start_idx = next(
        (
            i
            for i, d in enumerate(anchors)
            if (d - first_dt) >= (timedelta(days=win_days) + one_year)
        ),
        None,
    )
    if start_idx is None:
        print("No legal starting anchor found.")
        return

    # normalized blend
    wsum = max(W3 + W6 + W12, 1e-12)
    blend = np.array([W3 / wsum, W6 / wsum, W12 / wsum], dtype=np.float64)

    boosters = {"3mo": None, "6mo": None, "1y": None}

    for i in range(start_idx, len(anchors)):
        dt = anchors[i]
        dmin = dt - timedelta(days=win_days)
        dmax = dt - timedelta(days=1)

        feats_train = score_feats.filter(
            (pl.col("date") >= pl.lit(dmin)) & (pl.col("date") <= pl.lit(dmax))
        )
        if feats_train.is_empty():
            continue

        permnos = feats_train.select("permno").unique().to_series().to_list()
        ret_df = load_returns_block(dmin, dt, permnos)
        if ret_df.is_empty():
            continue

        lbl3 = build_labels_horizon(feats_train, ret_df, dt, "3mo")
        lbl6 = build_labels_horizon(feats_train, ret_df, dt, "6mo")
        lbl12 = build_labels_horizon(feats_train, ret_df, dt, "1y")

        tr3 = feats_train.join(lbl3, on=["date", "permno"], how="inner")
        tr6 = feats_train.join(lbl6, on=["date", "permno"], how="inner")
        tr12 = feats_train.join(lbl12, on=["date", "permno"], how="inner")

        feats_pred = score_feats.filter(pl.col("date") == pl.lit(dt))
        if feats_pred.is_empty():
            continue

        preds = []
        avail = []

        if tr3.height:
            y3, boosters["3mo"] = lgb_train_predict(tr3, feats_pred, dt, boosters["3mo"])
            preds.append(y3)
            avail.append(0)
        if tr6.height:
            y6, boosters["6mo"] = lgb_train_predict(tr6, feats_pred, dt, boosters["6mo"])
            preds.append(y6)
            avail.append(1)
        if tr12.height:
            y12, boosters["1y"] = lgb_train_predict(tr12, feats_pred, dt, boosters["1y"])
            preds.append(y12)
            avail.append(2)

        if not preds:
            continue

        P = np.vstack(preds)        # (k, N)
        w = blend[avail]
        w = w / w.sum()
        pred_12m = (w @ P).ravel()

        out_df = (
            feats_pred
            .select(["date", "permno"])
            .with_columns(pl.Series("pred_12m", pred_12m))
            .sort("pred_12m", descending=True)
        )

        outy = OUT_DIR / f"year={dt.year}"
        outy.mkdir(parents=True, exist_ok=True)
        out_path = outy / f"LightGBMscore_rank_{dt.strftime('%Y%m%d')}.parquet"
        out_df.write_parquet(str(out_path))
        print(f"[OK] {dt} -> {out_path} (N={out_df.height:,})")

if __name__ == "__main__":
    try:
        pl.Config.set_engine_affinity("streaming")
    except Exception:
        pass
    try:
        pl.enable_string_cache()
    except Exception:
        pass
    main()
