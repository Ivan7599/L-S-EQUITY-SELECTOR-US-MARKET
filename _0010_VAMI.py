# _plot_VAMI_Base_LGBM_Ridge_Mkt.py
from __future__ import annotations
from pathlib import Path
import io, re
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ------------- Paths -------------
RAW_DIR   = Path("./RawDatabase")
FF5_CSV   = RAW_DIR / "Factors5.csv"

BASE_RET  = Path("./FilteredRawData/BaseLS_Returns.parquet")
LGBM_RET  = Path("./FilteredRawData/LightGBMLS_Returns.parquet")
RIDGE_RET = Path("./FilteredRawData/RidgeLS_Returns.parquet")

OUT_IMG   = Path("./VAMI_Base_LGBM_Ridge_Mkt.jpg")


# ------------- Fama–French market factor -------------
def _read_ff_like(path: Path, header: str) -> pl.DataFrame:
    patt = re.compile(r"^\s*\d{8},")  # keep only YYYYMMDD lines
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if patt.match(line):
                rows.append(line.rstrip("\n"))
    if not rows:
        raise RuntimeError(f"No data lines detected in {path}")
    dtypes = {c: pl.Utf8 for c in header.split(",")}
    return pl.read_csv(io.StringIO(header + "\n" + "\n".join(rows)), dtypes=dtypes)


def read_ff5_market(path: Path) -> pl.DataFrame:
    """
    Return DATE (pl.Date) and mkt_ret (decimal daily return) from Fama–French CSV.
    Uses Mkt_RF as the market factor.
    """
    hdr = "DATE,Mkt_RF,SMB,HML,RMW,CMA,RF"
    df = _read_ff_like(path, hdr)
    return (
        df.with_columns(
            [
                pl.col("DATE")
                .str.strip_chars()
                .str.strptime(pl.Date, "%Y%m%d", strict=False)
                .alias("DATE"),
                (
                    pl.col("Mkt_RF")
                    .str.strip_chars()
                    .cast(pl.Float64)
                    / 100.0
                ).alias("mkt_ret"),
            ]
        )
        .select(["DATE", "mkt_ret"])
        .drop_nulls(subset=["DATE", "mkt_ret"])
    )


# ------------- Strategy return readers -------------
def read_ls_series(path: Path, colname: str) -> pl.DataFrame:
    """
    Read long–short daily returns Parquet with columns DATE, ls_ret,
    and rename ls_ret -> colname.
    """
    df = pl.read_parquet(str(path))
    if "DATE" not in df.columns or "ls_ret" not in df.columns:
        raise RuntimeError(f"{path.name} must contain DATE and ls_ret columns.")
    df = df.select(["DATE", "ls_ret"])
    if df.schema["DATE"] != pl.Date:
        df = df.with_columns(pl.col("DATE").cast(pl.Date, strict=False))
    return df.rename({"ls_ret": colname}).drop_nulls(subset=["DATE", colname])


# ------------- Plotting -------------
def plot_vami_four(df: pl.DataFrame, out_path: Path):
    dates      = df["DATE"].to_list()
    base_arr   = df["base_ret"].to_numpy()
    lgbm_arr   = df["lgbm_ret"].to_numpy()
    ridge_arr  = df["ridge_ret"].to_numpy()
    mkt_arr    = df["mkt_ret"].to_numpy()

    vami_base  = 100.0 * np.cumprod(1.0 + base_arr)
    vami_lgbm  = 100.0 * np.cumprod(1.0 + lgbm_arr)
    vami_ridge = 100.0 * np.cumprod(1.0 + ridge_arr)
    vami_mkt   = 100.0 * np.cumprod(1.0 + mkt_arr)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dates, vami_base,  linewidth=1.6, label="Base L/S")
    ax.plot(dates, vami_lgbm,  linewidth=1.6, linestyle="--", label="LightGBM L/S")
    ax.plot(dates, vami_ridge, linewidth=1.6, linestyle=":",  label="Ridge L/S")
    ax.plot(dates, vami_mkt,   linewidth=1.2, color="black", label="US Mkt")

    ax.set_title("VAMI (Base 100) — Base vs LightGBM vs Ridge vs US Market")
    ax.set_ylabel("VAMI (Base = 100)")
    ax.grid(True, alpha=0.3)

    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.legend()
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


# ------------- Main -------------
def main():
    base  = read_ls_series(BASE_RET,  "base_ret")
    lgbm  = read_ls_series(LGBM_RET,  "lgbm_ret")
    ridge = read_ls_series(RIDGE_RET, "ridge_ret")
    mkt   = read_ff5_market(FF5_CSV)

    # Align all four on common DATE range
    df = (
        base.join(lgbm,  on="DATE", how="inner")
            .join(ridge, on="DATE", how="inner")
            .join(mkt,   on="DATE", how="inner")
            .sort("DATE")
    )

    if df.is_empty():
        raise RuntimeError("No overlapping dates across Base, LightGBM, Ridge, and market.")

    plot_vami_four(df, OUT_IMG)
    print(f"Saved VAMI plot to {OUT_IMG}")


if __name__ == "__main__":
    main()
