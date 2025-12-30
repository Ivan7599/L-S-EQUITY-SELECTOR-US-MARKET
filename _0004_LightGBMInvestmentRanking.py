# _0004_LightGBMRanking.py — 3y rolling, multi-horizon LightGBM; no look-ahead; blazing-fast Polars
from __future__ import annotations
from pathlib import Path
from datetime import date, timedelta, datetime
import os, math
import numpy as np
import polars as pl

# ---------------- paths & config ----------------
BASE     = Path(".")
RAT_DIR  = BASE / "Ratios"                          # from _0002_RatiosCalculation.py
RET_DS   = BASE / "FilteredRawData" / "Returns_ds"  # from _0001_RawDataProcessing.py
OUT_DIR  = BASE / "LightGBMRanking"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# LightGBM params (fast & robust defaults; tune via env)
LGB_NUM_LEAVES   = int(os.environ.get("LGB_NUM_LEAVES", "31"))
LGB_MAX_DEPTH    = int(os.environ.get("LGB_MAX_DEPTH",  "6"))
LGB_LR           = float(os.environ.get("LGB_LR",       "0.05"))
LGB_N_EST        = int(os.environ.get("LGB_N_EST",      "300"))
LGB_MIN_LEAF     = int(os.environ.get("LGB_MIN_LEAF",   "200"))
LGB_FEAT_FRAC    = float(os.environ.get("LGB_FEAT_FRAC","0.7"))
LGB_BAG_FRAC     = float(os.environ.get("LGB_BAG_FRAC", "0.7"))
LGB_L2           = float(os.environ.get("LGB_L2",       "1.0"))
LGB_THREADS      = int(os.environ.get("LGB_THREADS",    str(os.cpu_count() or 4)))
EARLY_STOP_ROUNDS= int(os.environ.get("EARLY_STOP",     "100"))

# Rolling window & weighting
TRAIN_YEARS      = float(os.environ.get("TRAIN_YEARS", "3"))       # years of history
HALF_LIFE_YEARS  = float(os.environ.get("HALF_LIFE_YEARS", "1.5")) # exp half-life
VAL_MONTHS       = int(os.environ.get("VAL_MONTHS", "6"))          # OOT validation tail

# Blend weights for (3m,6m,12m)
W3, W6, W12      = (float(os.environ.get("BLEND_W3M","0.2")),
                    float(os.environ.get("BLEND_6M","0.3")),
                    float(os.environ.get("BLEND_12M","0.5")))

# Ratios consumed (present in Ratios/ outputs)
RATIO_COLS = [
    "GP_AT","OP_BE","ROA_TTM","CFO_AT","SloanAcc_AT","AssetGrowth","NOA_AT","NetShareIss",
    "MB","PE","PC","EV_EBITDA",
]

# --------------- LightGBM import (fail-friendly) ---------------
try:
    import lightgbm as lgb
except Exception as e:
    raise RuntimeError("LightGBM is required. Please `pip install lightgbm`.") from e

# --------------- utils ---------------
def scan_year_paths(ds_dir: Path, years) -> list[str]:
    paths = []
    for y in years:
        p = ds_dir / f"year={y}"
        if p.exists():
            paths += [str(pp) for pp in p.glob("*.parquet")]
    return paths

def unique_anchor_dates() -> list[date]:
    lf = (
        pl.scan_parquet(str(RAT_DIR / "year=*" / "ratios_*.parquet"))
          .select(pl.col("asof_date").cast(pl.Date).alias("date"))
          .unique()
    )
    return sorted(lf.collect(engine="streaming").get_column("date").to_list())

def clean_float_cols(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    # Coerce non-finite to null (LightGBM can handle NaN as missing)
    return df.with_columns([pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c) for c in cols])

def join_feats_ratios(dmin: date, dmax: date) -> pl.DataFrame:
    lf = (
        pl.scan_parquet(str(RAT_DIR / "year=*" / "ratios_*.parquet"))
          .select(
              date   = pl.col("asof_date").cast(pl.Date),
              permno = pl.col("permno").cast(pl.Int64),
              *[pl.col(c).cast(pl.Float64) for c in RATIO_COLS],
          )
          .filter((pl.col("date") >= pl.lit(dmin)) & (pl.col("date") <= pl.lit(dmax)))
    )
    df = lf.collect(engine="streaming")
    df = clean_float_cols(df, RATIO_COLS).drop_nulls(subset=["permno","date"])
    return df

def feats_for_date(dt: date) -> pl.DataFrame:
    lf = (
        pl.scan_parquet(str(RAT_DIR / "year=*" / "ratios_*.parquet"))
          .select(
              date   = pl.col("asof_date").cast(pl.Date),
              permno = pl.col("permno").cast(pl.Int64),
              *[pl.col(c).cast(pl.Float64) for c in RATIO_COLS],
          )
          .filter(pl.col("date") == pl.lit(dt))
    )
    df = lf.collect(engine="streaming")
    return clean_float_cols(df, RATIO_COLS)

def load_returns_block(dmin: date, dt: date, permnos: list[int]) -> pl.DataFrame:
    years = range(dmin.year, dt.year + 1)
    files = scan_year_paths(RET_DS, years)
    if not files:
        return pl.DataFrame(schema={"permno": pl.Int64, "DATE": pl.Date, "RET": pl.Float64})
    ret = (
        pl.scan_parquet(files)
          .select(
              permno = pl.col("PERMNO").cast(pl.Int64),
              DATE   = pl.coalesce([
                          pl.col("DATE").cast(pl.Date, strict=False),
                          pl.col("DATE").cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False),
                          pl.col("DATE").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d",  strict=False),
                      ]),
              RET    = pl.col("RET").cast(pl.Float64),
          )
          .filter(pl.col("permno").is_in(pl.lit(permnos)) &
                  (pl.col("DATE") > pl.lit(dmin)) & (pl.col("DATE") <= pl.lit(dt)))
          .with_columns(pl.col("RET").fill_null(0.0))
          .collect(engine="streaming")
    )
    if ret.is_empty():
        return ret
    ret = ret.sort(["permno","DATE"])
    ret = ret.with_columns(pl.col("RET").log1p().over("permno").cum_sum().alias("cumlog"))
    return ret

def build_labels_horizon(feats_train: pl.DataFrame, ret_df: pl.DataFrame, dt: date, horizon: str) -> pl.DataFrame:
    # horizon: "3mo", "6mo", "1y"
    if feats_train.is_empty() or ret_df.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "permno": pl.Int64, "y": pl.Float64})

    cal = ret_df.select(pl.col("DATE")).unique().sort("DATE")

    anchors = (
        feats_train.select("date").unique().sort("date")
                   .with_columns(pl.col("date").cast(pl.Datetime).dt.offset_by(horizon).cast(pl.Date).alias("end_target"))
    )
    anchors = (
        anchors.join_asof(cal, left_on="end_target", right_on="DATE", strategy="backward")
               .rename({"DATE":"end_date"})
               .drop("end_target")
               .drop_nulls(subset=["end_date"])
               .filter(pl.col("end_date") <= pl.lit(dt))  # no look-ahead
    )
    if anchors.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "permno": pl.Int64, "y": pl.Float64})

    A = feats_train.select(["date","permno"]).unique().join(anchors, on="date", how="inner")

    start = (A.sort(["permno","date"])
               .join_asof(ret_df, left_on="date", right_on="DATE", by="permno", strategy="backward")
               .rename({"cumlog":"cum_start"})
               .select("date","permno","end_date","cum_start"))
    end   = (A.sort(["permno","end_date"])
               .join_asof(ret_df, left_on="end_date", right_on="DATE", by="permno", strategy="backward")
               .rename({"cumlog":"cum_end"})
               .select("date","permno","end_date","cum_end"))

    lbl = (
        start.join(end, on=["date","permno","end_date"], how="inner")
             .with_columns(((pl.col("cum_end") - pl.col("cum_start")).exp() - 1.0).alias("y"))
             .select(["date","permno","y"])
             .drop_nulls(subset=["y"])
    )
    return lbl

def time_decay_weights(dates: np.ndarray, dt: date, half_life_years: float) -> np.ndarray:
    days = (np.datetime64(dt) - dates.astype("datetime64[D]")).astype("timedelta64[D]").astype(np.float64)
    age_years = days / 365.2425
    w = np.exp(-math.log(2.0) * (age_years / max(half_life_years, 1e-9)))
    w = np.clip(w, 1e-8, None)
    return w

def lgb_train_predict(train_df: pl.DataFrame, pred_df: pl.DataFrame, dt: date,
                      booster_prev=None) -> tuple[np.ndarray, object]:
    fcols = [c for c in train_df.columns if c not in ("date","permno","y")]
    if not fcols or train_df.is_empty() or pred_df.is_empty():
        return np.array([]), booster_prev

    # ndarray (float32)
    X = train_df.select(fcols).to_numpy().astype(np.float32)
    y = train_df["y"].to_numpy().astype(np.float32)
    Xp = pred_df.select(fcols).to_numpy().astype(np.float32)

    # time-decay weights
    w = time_decay_weights(train_df["date"].to_numpy(), dt, HALF_LIFE_YEARS).astype(np.float32)

    # OOT validation = last VAL_MONTHS of window
    cutoff = (np.datetime64(dt) - np.timedelta64(30*VAL_MONTHS, "D")).astype("datetime64[D]")
    mask_val = (train_df["date"].to_numpy().astype("datetime64[D]") >= cutoff)
    if mask_val.any() and (~mask_val).any():
        dtrain = lgb.Dataset(X[~mask_val], y[~mask_val], weight=w[~mask_val], feature_name=fcols, free_raw_data=False)
        dvalid = lgb.Dataset(X[ mask_val], y[ mask_val], weight=w[ mask_val], reference=dtrain, free_raw_data=False)
        valid_sets = [dtrain, dvalid]; valid_names = ["train","valid"]
        do_early_stop = True
    else:
        dtrain = lgb.Dataset(X, y, weight=w, feature_name=fcols, free_raw_data=False)
        valid_sets = [dtrain]; valid_names = ["train"]
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
    # Use callbacks instead of early_stopping_rounds for broad version compatibility
    if do_early_stop and EARLY_STOP_ROUNDS > 0:
        callbacks.append(lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False))
    # Optional: silence logs
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
    yhat = booster.predict(Xp, num_iteration=booster.best_iteration or booster.current_iteration())
    return yhat.astype(np.float64), booster

# --------------- main loop ---------------
def main():
    anchors = unique_anchor_dates()
    if len(anchors) < 12:
        print("Not enough anchor dates in Ratios/ to run."); return

    win_days = int(round(365.2425 * TRAIN_YEARS))
    one_year = timedelta(days=365)
    first_dt = anchors[0]
    # need TRAIN_YEARS + 12m matured labels to start
    start_idx = next((i for i, d in enumerate(anchors)
                      if (d - first_dt) >= (timedelta(days=win_days) + one_year)), None)
    if start_idx is None:
        print("No legal starting anchor found."); return

    # normalized blend
    wsum = max(W3 + W6 + W12, 1e-12)
    blend = np.array([W3/wsum, W6/wsum, W12/wsum], dtype=np.float64)

    # warm-start boosters
    boosters = {"3mo": None, "6mo": None, "1y": None}

    for i in range(start_idx, len(anchors)):
        dt   = anchors[i]
        dmin = dt - timedelta(days=win_days)
        dmax = dt - timedelta(days=1)

        feats_train = join_feats_ratios(dmin, dmax)
        if feats_train.is_empty():
            continue

        permnos = feats_train.select("permno").unique().to_series().to_list()
        ret_df  = load_returns_block(dmin, dt, permnos)
        if ret_df.is_empty():
            continue

        # labels per horizon (no look-ahead)
        lbl3  = build_labels_horizon(feats_train, ret_df, dt, "3mo")
        lbl6  = build_labels_horizon(feats_train, ret_df, dt, "6mo")
        lbl12 = build_labels_horizon(feats_train, ret_df, dt, "1y")

        # join labels to features
        tr3  = feats_train.join(lbl3,  on=["date","permno"], how="inner")
        tr6  = feats_train.join(lbl6,  on=["date","permno"], how="inner")
        tr12 = feats_train.join(lbl12, on=["date","permno"], how="inner")

        # current cross-section features to score
        feats_pred = feats_for_date(dt)
        if feats_pred.is_empty():
            continue

        preds = []
        avail = []
        if tr3.height:
            y3, boosters["3mo"] = lgb_train_predict(tr3, feats_pred, dt, boosters["3mo"])
            preds.append(y3); avail.append(0)
        if tr6.height:
            y6, boosters["6mo"] = lgb_train_predict(tr6, feats_pred, dt, boosters["6mo"])
            preds.append(y6); avail.append(1)
        if tr12.height:
            y12, boosters["1y"] = lgb_train_predict(tr12, feats_pred, dt, boosters["1y"])
            preds.append(y12); avail.append(2)

        if not preds:
            continue

        P = np.vstack(preds)                    # (k, N)
        w = blend[avail]                        # select available weights
        w = w / w.sum()                         # renormalize if some horizons missing
        pred_12m = (w @ P).ravel()

        out_df = (
            feats_pred
              .select(["date","permno"])
              .with_columns(pl.Series("pred_12m", pred_12m))
              .sort("pred_12m", descending=True)
        )

        outy = OUT_DIR / f"year={dt.year}"
        outy.mkdir(parents=True, exist_ok=True)
        out_path = outy / f"LightGBMrank_{dt.strftime('%Y%m%d')}.parquet"
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
