# _0008_FFLegsRegression_AllModels.py
# OLS vs FF5+Mom for Long & Short legs — all models at once
from __future__ import annotations
from pathlib import Path
import io, re
import numpy as np
import polars as pl

RAW_DIR = Path("./RawDatabase")
FF5_CSV = RAW_DIR / "Factors5.csv"
MOM_CSV = RAW_DIR / "Momentum.csv"

# ----------------- Model specs -----------------
MODEL_SPECS = [
    {
        "name": "Base",
        "long_parq":  Path("./FilteredRawData/BaseLong_Returns.parquet"),
        "short_parq": Path("./FilteredRawData/BaseShort_Returns.parquet"),
        "out_long":   Path("./BaseOLSFamaLong.csv"),
        "out_short":  Path("./BaseOLSFamaShort.csv"),
    },
    {
        "name": "LightGBM",
        "long_parq":  Path("./FilteredRawData/LightGBMLong_Returns.parquet"),
        "short_parq": Path("./FilteredRawData/LightGBMShort_Returns.parquet"),
        "out_long":   Path("./LightGBMOLSFamaLong.csv"),
        "out_short":  Path("./LightGBMOLSFamaShort.csv"),
    },
    {
        "name": "Ridge",
        "long_parq":  Path("./FilteredRawData/RidgeLong_Returns.parquet"),
        "short_parq": Path("./FilteredRawData/RidgeShort_Returns.parquet"),
        "out_long":   Path("./RidgeOLSFamaLong.csv"),
        "out_short":  Path("./RidgeOLSFamaShort.csv"),
    },
]

# ----------------- FF data readers -----------------
def _read_ff_like(path: Path, header: str) -> pl.DataFrame:
    patt = re.compile(r"^\s*\d{8},")
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if patt.match(line):
                rows.append(line.rstrip("\n"))
    if not rows:
        raise RuntimeError(f"No data lines detected in {path}")
    dtypes = {c: pl.Utf8 for c in header.split(",")}
    return pl.read_csv(io.StringIO(header + "\n" + "\n".join(rows)), dtypes=dtypes)

def read_ff5(path: Path) -> pl.DataFrame:
    hdr = "DATE,Mkt_RF,SMB,HML,RMW,CMA,RF"
    df = _read_ff_like(path, hdr)
    nums = ["Mkt_RF","SMB","HML","RMW","CMA","RF"]
    return (
        df.with_columns([
            pl.col("DATE")
              .str.strip_chars()
              .str.strptime(pl.Date, "%Y%m%d", strict=False),
            *[
                (pl.col(c).str.strip_chars().cast(pl.Float64) / 100.0).alias(c)
                for c in nums
            ],
        ])
        .drop_nulls(subset=["DATE"])
    )

def read_mom(path: Path) -> pl.DataFrame:
    hdr = "DATE,Mom"
    df = _read_ff_like(path, hdr)
    return (
        df.with_columns([
            pl.col("DATE")
              .str.strip_chars()
              .str.strptime(pl.Date, "%Y%m%d", strict=False),
            (pl.col("Mom").str.strip_chars().cast(pl.Float64) / 100.0).alias("Mom"),
        ])
        .drop_nulls(subset=["DATE"])
    )

# ----------------- Strategy leg reader -----------------
def read_leg(parq: Path, colname: str) -> pl.DataFrame:
    df = pl.read_parquet(str(parq))
    if "DATE" not in df.columns or colname not in df.columns:
        raise RuntimeError(f"{parq.name} must contain DATE and {colname}.")
    if df.schema["DATE"] != pl.Date:
        df = df.with_columns(pl.col("DATE").cast(pl.Date, strict=False))
    return df.select(["DATE", colname]).drop_nulls()

# ----------------- OLS helper -----------------
def ols_with_tstats(y: np.ndarray, X: np.ndarray):
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    n, k = X.shape
    resid = y - X @ beta
    s2 = (resid @ resid) / (n - k)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(s2 * XtX_inv))
    tstats = beta / se
    ss_tot = ((y - y.mean())**2).sum()
    r2 = 1.0 - (resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if n > k else float("nan")
    return beta, tstats, r2, adj_r2

# ----------------- Core runner per leg -----------------
def run_one(leg_df: pl.DataFrame, ff: pl.DataFrame, ret_col: str, out_csv: Path, label: str):
    data = leg_df.join(ff, on="DATE", how="inner").sort("DATE")
    if data.height < 30:
        raise RuntimeError(f"Too few overlapping days for {label} ({out_csv.name}).")

    y = data[ret_col].to_numpy()
    X = np.column_stack([
        np.ones(len(y)),
        data["Mkt_RF"].to_numpy(),
        data["SMB"].to_numpy(),
        data["HML"].to_numpy(),
        data["RMW"].to_numpy(),
        data["CMA"].to_numpy(),
        data["Mom"].to_numpy(),
    ])

    beta, tstats, r2, adj_r2 = ols_with_tstats(y, X)

    pl.DataFrame({
        "leg":          [label],
        "sample_start": [data["DATE"].min()],
        "sample_end":   [data["DATE"].max()],
        "N":            [len(y)],
        "alpha":        [beta[0]],
        "t_alpha":      [tstats[0]],
        "beta_mkt":     [beta[1]], "t_mkt":  [tstats[1]],
        "beta_smb":     [beta[2]], "t_smb":  [tstats[2]],
        "beta_hml":     [beta[3]], "t_hml":  [tstats[3]],
        "beta_rmw":     [beta[4]], "t_rmw":  [tstats[4]],
        "beta_cma":     [beta[5]], "t_cma":  [tstats[5]],
        "beta_mom":     [beta[6]], "t_mom":  [tstats[6]],
        "R2":           [r2],
        "Adj_R2":       [adj_r2],
    }).write_csv(str(out_csv))

    print(f"Saved {out_csv}")

# ----------------- Main: loop over models -----------------
def main():
    ff5 = read_ff5(FF5_CSV)
    mom = read_mom(MOM_CSV)
    ff  = ff5.join(mom, on="DATE", how="inner").select(
        ["DATE","Mkt_RF","SMB","HML","RMW","CMA","RF","Mom"]
    )

    for spec in MODEL_SPECS:
        name      = spec["name"]
        long_parq = spec["long_parq"]
        short_parq= spec["short_parq"]
        out_long  = spec["out_long"]
        out_short = spec["out_short"]

        print(f"\n=== {name}: running FF5+Mom OLS for long & short legs ===")

        long_leg  = read_leg(long_parq,  "long_ret")
        short_leg = read_leg(short_parq, "short_ret")  # already P&L (positive when shorts win)

        run_one(long_leg,  ff, "long_ret",  out_long,  label=f"{name}_Long")
        run_one(short_leg, ff, "short_ret", out_short, label=f"{name}_Short")

if __name__ == "__main__":
    main()
