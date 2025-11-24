# _0005_RidgeRanking.py — 3y rolling, multi-horizon ridge with fast median imputation (no all-12 requirement)
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
OUT_DIR  = BASE / "RidgeRanking"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Ridge + window knobs
ALPHA            = float(os.environ.get("RIDGE_ALPHA", "1.0"))
TRAIN_YEARS      = float(os.environ.get("TRAIN_YEARS", "3"))
HALF_LIFE_YEARS  = float(os.environ.get("HALF_LIFE_YEARS", "1.5"))

# Blend weights (normalized at runtime)
W3   = float(os.environ.get("BLEND_W3M",  "0.2"))
W6   = float(os.environ.get("BLEND_6M",   "0.3"))
W12  = float(os.environ.get("BLEND_12M",  "0.5"))

# Missingness flags toggle (1 = add miss_* columns, 0 = no flags)
ADD_MISS_FLAGS   = int(os.environ.get("ADD_MISS_FLAGS", "1"))

# 12 raw ratios to use (present in _0002_RatiosCalculation.py outputs)
RATIO_COLS = [
    "GP_AT","OP_BE","ROA_TTM","CFO_AT",
    "SloanAcc_AT","AssetGrowth","NOA_AT","NetShareIss",
    "MB","PE","PC","EV_EBITDA",
]

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

def join_feats_ratios(dmin: date, dmax: date) -> pl.DataFrame:
    # NOTE: do NOT drop rows for null ratios; keep coverage and impute later.
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
    # convert non-finite to null for clean imputation
    df = df.with_columns([
        pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c) for c in RATIO_COLS
    ])
    # keep permno/date only
    return df.drop_nulls(subset=["permno","date"])

def feats_for_date(dt: date) -> pl.DataFrame:
    return join_feats_ratios(dt, dt)

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
                  (pl.col("DATE") > pl.lit(dmin)) &
                  (pl.col("DATE") <= pl.lit(dt)))
          .with_columns(pl.col("RET").fill_null(0.0))
          .collect(engine="streaming")
    )
    if ret.is_empty():
        return ret
    ret = ret.sort(["permno","DATE"])
    ret = ret.with_columns(pl.col("RET").log1p().over("permno").cum_sum().alias("cumlog"))
    return ret

def build_labels_horizon(feats_train: pl.DataFrame, ret_df: pl.DataFrame, dt: date, horizon: str) -> pl.DataFrame:
    """
    horizon: "3mo", "6mo", "1y" (calendar offsets)
    Returns: DataFrame[date, permno, y] where y is forward return over horizon
    """
    if feats_train.is_empty() or ret_df.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "permno": pl.Int64, "y": pl.Float64})

    # trading calendar up to dt
    cal = ret_df.select(pl.col("DATE")).unique().sort("DATE")

    # τ with τ + horizon <= dt (calendar end mapped as-of to trading day)
    anchors = (
        feats_train.select("date").unique().sort("date")
                   .with_columns(pl.col("date").cast(pl.Datetime).dt.offset_by(horizon).cast(pl.Date).alias("end_target"))
    )
    anchors = (
        anchors.join_asof(cal, left_on="end_target", right_on="DATE", strategy="backward")
               .rename({"DATE":"end_date"})
               .drop("end_target")
               .drop_nulls(subset=["end_date"])
               .filter(pl.col("end_date") <= pl.lit(dt))
    )
    if anchors.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "permno": pl.Int64, "y": pl.Float64})

    # pairs (date, permno) with valid end_date
    A = (feats_train.select(["date","permno"]).unique()
                      .join(anchors, on="date", how="inner"))

    # as-of at start and end
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

# ---------- fast median imputation (+ optional missingness flags) ----------
def compute_impute_map(train_feats: pl.DataFrame, cols: list[str]) -> dict[str, float]:
    # column-wise medians; fallback to 0.0 if entirely null
    med = train_feats.select([pl.col(c).median().alias(c) for c in cols])
    vals = med.row(0) if med.height else [None]*len(cols)
    out = {}
    for c, v in zip(cols, vals):
        out[c] = float(v) if (v is not None and np.isfinite(v)) else 0.0
    return out

def apply_impute_with_flags(df: pl.DataFrame, imap: dict[str,float], cols: list[str], add_flags: bool) -> pl.DataFrame:
    exprs = []
    for c in cols:
        clean = pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None)
        if add_flags:
            exprs.append(clean.is_null().cast(pl.Int8).alias(f"miss_{c}"))
        exprs.append(clean.fill_null(imap[c]).alias(c))
    return df.with_columns(exprs)

# ---------- weighted ridge ----------
def ridge_fit_predict_weighted(train_df: pl.DataFrame, pred_df: pl.DataFrame, alpha: float, dt: date,
                               half_life_years: float) -> np.ndarray:
    # features = all except date/permno/y
    fcols = [c for c in train_df.columns if c not in ("date","permno","y")]
    if not fcols: return np.array([])

    X = train_df.select(fcols).to_numpy()
    y = train_df["y"].to_numpy()
    if X.size == 0 or y.size == 0:
        return np.array([])

    # time-decay weights by age (years)
    days = (np.datetime64(dt) - train_df["date"].to_numpy()).astype("timedelta64[D]").astype(np.float64)
    age_years = days / 365.2425
    w = np.exp(-math.log(2.0) * (age_years / max(half_life_years, 1e-9)))
    w = np.clip(w, 1e-8, None)
    wn = w / w.sum()

    # weighted standardization
    mu  = (wn[:, None] * X).sum(axis=0)
    Xc  = X - mu
    ybar= float((wn * y).sum())
    yc  = y - ybar
    var = (wn[:, None] * (Xc * Xc)).sum(axis=0)
    std = np.sqrt(np.maximum(var, 1e-12))
    Z   = Xc / std

    # solve (Z'WZ + αI)β = Z'Wy
    sqrtw = np.sqrt(w)[:, None]
    Zw = Z * sqrtw
    yw = yc * np.sqrt(w)
    S = Zw.T @ Zw
    b = Zw.T @ yw
    S.flat[::S.shape[0]+1] += alpha
    beta_std = np.linalg.solve(S, b)
    beta = beta_std / std

    # predict
    Xp = pred_df.select(fcols).to_numpy()
    yhat = ybar + (Xp - mu) @ beta
    return yhat

# --------------- main loop ---------------
def main():
    anchors = unique_anchor_dates()
    if len(anchors) < 12:
        print("Not enough anchor dates in Ratios/ to run."); return

    # start when we have TRAIN_YEARS + 12m mature labels
    win_days = int(round(365.2425 * TRAIN_YEARS))
    one_year = timedelta(days=365)
    first_dt = anchors[0]
    start_idx = next((i for i, d in enumerate(anchors)
                      if (d - first_dt) >= (timedelta(days=win_days) + one_year)), None)
    if start_idx is None:
        print("No legal starting anchor found."); return

    # blend normalization
    ws = np.array([W3, W6, W12], dtype=float)
    ws = ws / max(ws.sum(), 1e-12)

    for i in range(start_idx, len(anchors)):
        dt   = anchors[i]
        dmin = dt - timedelta(days=win_days)
        dmax = dt - timedelta(days=1)

        feats_train = join_feats_ratios(dmin, dmax)
        if feats_train.is_empty():
            continue

        # Load returns once per dt
        permnos = feats_train.select("permno").unique().to_series().to_list()
        ret_df  = load_returns_block(dmin, dt, permnos)
        if ret_df.is_empty():
            continue

        # Labels per horizon
        lbl3  = build_labels_horizon(feats_train, ret_df, dt, "3mo")
        lbl6  = build_labels_horizon(feats_train, ret_df, dt, "6mo")
        lbl12 = build_labels_horizon(feats_train, ret_df, dt, "1y")

        # Join labels to features (window-level features; impute AFTER joins)
        tr3  = feats_train.join(lbl3,  on=["date","permno"], how="inner")
        tr6  = feats_train.join(lbl6,  on=["date","permno"], how="inner")
        tr12 = feats_train.join(lbl12, on=["date","permno"], how="inner")

        # Current cross-section features (at dt)
        feats_pred = feats_for_date(dt)
        if feats_pred.is_empty():
            continue

        # Compute imputation map ON THE WHOLE WINDOW FEATURES (feats_train) once; reuse for all horizons + pred
        imap = compute_impute_map(feats_train, RATIO_COLS)

        # Apply imputation (+ optional missingness flags) consistently
        tr3  = apply_impute_with_flags(tr3,  imap, RATIO_COLS, ADD_MISS_FLAGS)
        tr6  = apply_impute_with_flags(tr6,  imap, RATIO_COLS, ADD_MISS_FLAGS)
        tr12 = apply_impute_with_flags(tr12, imap, RATIO_COLS, ADD_MISS_FLAGS)
        feats_pred = apply_impute_with_flags(feats_pred, imap, RATIO_COLS, ADD_MISS_FLAGS)

        # Fit/predict per horizon
        yhat3  = ridge_fit_predict_weighted(tr3,  feats_pred, ALPHA, dt, HALF_LIFE_YEARS)  if tr3.height  else np.array([])
        yhat6  = ridge_fit_predict_weighted(tr6,  feats_pred, ALPHA, dt, HALF_LIFE_YEARS)  if tr6.height  else np.array([])
        yhat12 = ridge_fit_predict_weighted(tr12, feats_pred, ALPHA, dt, HALF_LIFE_YEARS)  if tr12.height else np.array([])

        preds, weights = [], []
        if yhat3.size:   preds.append(yhat3);  weights.append(ws[0])
        if yhat6.size:   preds.append(yhat6);  weights.append(ws[1])
        if yhat12.size:  preds.append(yhat12); weights.append(ws[2])
        if not preds:    continue

        wsum = sum(weights)
        weights = [w/wsum for w in weights] if wsum > 0 else [1.0/len(preds)]*len(preds)

        P = np.vstack(preds)              # (k, N)
        pred_12m = (weights @ P).ravel()  # (N,)

        out_df = (
            feats_pred
              .select(["date","permno"])
              .with_columns(pl.Series("pred_12m", pred_12m))
              .sort("pred_12m", descending=True)
        )

        outy = OUT_DIR / f"year={dt.year}"
        outy.mkdir(parents=True, exist_ok=True)
        out_path = outy / f"ridgerank_{dt.strftime('%Y%m%d')}.parquet"
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
