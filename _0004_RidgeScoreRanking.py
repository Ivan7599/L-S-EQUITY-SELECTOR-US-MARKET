# _0005_RidgeScoreRanking.py — Ridge ranking on avg / slope / vol of composite score
from __future__ import annotations
from pathlib import Path
from datetime import date, timedelta, datetime
import os, math, re
import numpy as np
import polars as pl

# ---------------- paths & config ----------------
BASE    = Path(".")
AR_DIR  = BASE / "AnchorRanking"                    # from _0004_AnchorRanking.py
RET_DS  = BASE / "FilteredRawData" / "Returns_ds"   # from _0001_RawDataProcessing.py
OUT_DIR = BASE / "RidgeScoreRanking"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Ridge + window knobs (same as _0005_RidgeRanking.py)
ALPHA           = float(os.environ.get("RIDGE_ALPHA", "1.0"))
TRAIN_YEARS     = float(os.environ.get("TRAIN_YEARS", "3"))
HALF_LIFE_YEARS = float(os.environ.get("HALF_LIFE_YEARS", "1.5"))

# Blend weights (normalized at runtime)
W3  = float(os.environ.get("BLEND_W3M",  "0.2"))
W6  = float(os.environ.get("BLEND_6M",   "0.3"))
W12 = float(os.environ.get("BLEND_12M",  "0.5"))

# Missingness flags toggle
ADD_MISS_FLAGS = int(os.environ.get("ADD_MISS_FLAGS", "1"))

# Score feature window (in anchors)
SCORE_WIN_ANCHORS = int(os.environ.get("SCORE_WIN_ANCHORS", "12"))
MIN_FRAC          = float(os.environ.get("SCORE_MIN_FRAC", "0.60"))

def MIN_OBS(n: int) -> int:
    return max(3, int(math.ceil(n * MIN_FRAC)))

FEAT_COLS = ["avg_score", "slope_score", "vol_dscore"]

# ---------------- AnchorRanking file utilities ----------------
def list_anchor_files() -> list[Path]:
    files: list[Path] = []
    for ydir in sorted(AR_DIR.glob("year=*")):
        for f in sorted(ydir.glob("*.parquet")):
            if re.search(r"\d{8}", f.name):
                files.append(f)
    return files

def parse_date_tag(p: Path) -> str:
    m = re.search(r"(\d{8})", p.name)
    if not m:
        raise ValueError(f"Cannot find 8-digit date in filename: {p.name}")
    return m.group(1)

# ---------------- score features from composite score ----------------
def compute_window_scores(files_win: list[Path]) -> tuple[pl.DataFrame, str | None]:
    """
    For one rolling window of anchor files, compute avg_score, slope_score, vol_dscore
    based on cross-sectionally standardized composite score.
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

        # cross-sectional robust standardization of score within this anchor
        med = float(df.select(pl.col("score").median()).item())
        df_dev = df.with_columns((pl.col("score") - med).alias("abs_dev_seed"))
        mad = float(df_dev.select(pl.col("abs_dev_seed").abs().median()).item())
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

    # OLS slope and average of standardized score_z in the window
    sums = (
        panel.group_by("permno").agg(
            pl.len().alias("k"),
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

    # volatility of Δscore_z across the window
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

# ---------------- returns / labels (same as Ridge on ratios) ----------------
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
                      pl.col("DATE").cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False),
                      pl.col("DATE").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False),
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
    """
    horizon: "3mo", "6mo", "1y"
    Returns: DataFrame[date, permno, y] where y is forward return over horizon.
    """
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
               .filter(pl.col("end_date") <= pl.lit(dt))
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

# ---------- median imputation (+ optional missingness flags) ----------
def compute_impute_map(train_feats: pl.DataFrame, cols: list[str]) -> dict[str, float]:
    med = train_feats.select([pl.col(c).median().alias(c) for c in cols])
    vals = med.row(0) if med.height else [None] * len(cols)
    out: dict[str, float] = {}
    for c, v in zip(cols, vals):
        out[c] = float(v) if (v is not None and np.isfinite(v)) else 0.0
    return out

def apply_impute_with_flags(
    df: pl.DataFrame, imap: dict[str, float], cols: list[str], add_flags: bool
) -> pl.DataFrame:
    exprs = []
    for c in cols:
        clean = pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None)
        if add_flags:
            exprs.append(clean.is_null().cast(pl.Int8).alias(f"miss_{c}"))
        exprs.append(clean.fill_null(imap[c]).alias(c))
    return df.with_columns(exprs)

# ---------- weighted ridge (same core math as _0005_RidgeRanking.py) ----------
def ridge_fit_predict_weighted(
    train_df: pl.DataFrame,
    pred_df: pl.DataFrame,
    alpha: float,
    dt: date,
    half_life_years: float,
) -> np.ndarray:
    fcols = [c for c in train_df.columns if c not in ("date", "permno", "y")]
    if not fcols:
        return np.array([])

    X = train_df.select(fcols).to_numpy()
    y = train_df["y"].to_numpy()
    if X.size == 0 or y.size == 0:
        return np.array([])

    days = (
        np.datetime64(dt) - train_df["date"].to_numpy()
    ).astype("timedelta64[D]").astype(np.float64)
    age_years = days / 365.2425
    w = np.exp(-math.log(2.0) * (age_years / max(half_life_years, 1e-9)))
    w = np.clip(w, 1e-8, None)
    wn = w / w.sum()

    mu = (wn[:, None] * X).sum(axis=0)
    Xc = X - mu
    ybar = float((wn * y).sum())
    yc = y - ybar
    var = (wn[:, None] * (Xc * Xc)).sum(axis=0)
    std = np.sqrt(np.maximum(var, 1e-12))
    Z = Xc / std

    sqrtw = np.sqrt(w)[:, None]
    Zw = Z * sqrtw
    yw = yc * np.sqrt(w)
    S = Zw.T @ Zw
    b = Zw.T @ yw
    S.flat[:: S.shape[0] + 1] += alpha
    beta_std = np.linalg.solve(S, b)
    beta = beta_std / std

    Xp = pred_df.select(fcols).to_numpy()
    yhat = ybar + (Xp - mu) @ beta
    return yhat

# --------------- main loop ---------------
def main():
    # build score features once (avg / slope / vol of composite score)
    score_feats = build_score_features()
    if score_feats.is_empty():
        print("No score features available; aborting.")
        return

    anchors = sorted(score_feats["date"].unique().to_list())
    if len(anchors) < 12:
        print("Not enough anchor dates in score features to run.")
        return

    win_days = int(round(365.2425 * TRAIN_YEARS))
    one_year = timedelta(days=365)
    first_dt = anchors[0]

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

    ws = np.array([W3, W6, W12], dtype=float)
    ws = ws / max(ws.sum(), 1e-12)

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

        imap = compute_impute_map(feats_train, FEAT_COLS)

        tr3 = apply_impute_with_flags(tr3, imap, FEAT_COLS, ADD_MISS_FLAGS)
        tr6 = apply_impute_with_flags(tr6, imap, FEAT_COLS, ADD_MISS_FLAGS)
        tr12 = apply_impute_with_flags(tr12, imap, FEAT_COLS, ADD_MISS_FLAGS)
        feats_pred = apply_impute_with_flags(feats_pred, imap, FEAT_COLS, ADD_MISS_FLAGS)

        yhat3 = (
            ridge_fit_predict_weighted(tr3, feats_pred, ALPHA, dt, HALF_LIFE_YEARS)
            if tr3.height
            else np.array([])
        )
        yhat6 = (
            ridge_fit_predict_weighted(tr6, feats_pred, ALPHA, dt, HALF_LIFE_YEARS)
            if tr6.height
            else np.array([])
        )
        yhat12 = (
            ridge_fit_predict_weighted(tr12, feats_pred, ALPHA, dt, HALF_LIFE_YEARS)
            if tr12.height
            else np.array([])
        )

        preds, weights = [], []
        if yhat3.size:
            preds.append(yhat3)
            weights.append(ws[0])
        if yhat6.size:
            preds.append(yhat6)
            weights.append(ws[1])
        if yhat12.size:
            preds.append(yhat12)
            weights.append(ws[2])
        if not preds:
            continue

        wsum = sum(weights)
        weights = [w / wsum for w in weights] if wsum > 0 else [1.0 / len(preds)] * len(preds)

        P = np.vstack(preds)
        pred_12m = (weights @ P).ravel()

        out_df = (
            feats_pred.select(["date", "permno"])
            .with_columns(pl.Series("pred_12m", pred_12m))
            .sort("pred_12m", descending=True)
        )

        outy = OUT_DIR / f"year={dt.year}"
        outy.mkdir(parents=True, exist_ok=True)
        out_path = outy / f"ridgescore_rank_{dt.strftime('%Y%m%d')}.parquet"
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
