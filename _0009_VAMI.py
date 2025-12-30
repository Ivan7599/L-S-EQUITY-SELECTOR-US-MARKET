# _plot_VAMI_AllModels_Mkt.py
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

BASE_RET       = Path("./FilteredRawData/BaseLS_Returns.parquet")
LGBM_RET       = Path("./FilteredRawData/LightGBMLS_Returns.parquet")
LGBM_SCORE_RET = Path("./FilteredRawData/LightGBMScoreLS_Returns.parquet")
RIDGE_RET      = Path("./FilteredRawData/RidgeLS_Returns.parquet")
RIDGE_SCORE_RET= Path("./FilteredRawData/RidgeScoreLS_Returns.parquet")

OUT_IMG   = Path("./VAMI_AllModels_Mkt.jpg")

MODEL_SPECS = [
    {"key": "base",        "label": "Base L/S",                 "path": BASE_RET},
    {"key": "lgbm",        "label": "LightGBM Raw L/S",         "path": LGBM_RET},
    {"key": "lgbm_score",  "label": "LightGBM 3 features L/S",  "path": LGBM_SCORE_RET},
    {"key": "ridge",       "label": "Ridge Raw L/S",            "path": RIDGE_RET},
    {"key": "ridge_score", "label": "Ridge 3 features L/S",     "path": RIDGE_SCORE_RET},
]

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


# ------------- Strategy return reader -------------
def read_ls_series(path: Path, colname: str) -> pl.DataFrame | None:
    """
    Read long–short daily returns Parquet with columns DATE, ls_ret,
    and rename ls_ret -> colname. Returns None if file missing or malformed.
    """
    if not path.exists():
        print(f"[warn] {path} not found, skipping.")
        return None
    df = pl.read_parquet(str(path))
    if "DATE" not in df.columns or "ls_ret" not in df.columns:
        print(f"[warn] {path.name} must contain DATE and ls_ret columns. Skipping.")
        return None
    if df.schema["DATE"] != pl.Date:
        df = df.with_columns(pl.col("DATE").cast(pl.Date, strict=False))
    return df.select(["DATE", "ls_ret"]).rename({"ls_ret": colname}).drop_nulls(subset=["DATE", colname])


# ------------- Plotting -------------
def plot_vami_all(df: pl.DataFrame, cols_info: list[dict], out_path: Path):
    dates   = df["DATE"].to_list()
    mkt_arr = df["mkt_ret"].to_numpy()
    vami_mkt = 100.0 * np.cumprod(1.0 + mkt_arr)

    fig, ax = plt.subplots(figsize=(11, 5))

    # Market first (black)
    ax.plot(dates, vami_mkt, linewidth=1.6, color="black", label="US Mkt (Mkt_RF)")

    # Then all models
    for spec in cols_info:
        col   = spec["col"]
        label = spec["label"]
        arr   = df[col].to_numpy()
        vami  = 100.0 * np.cumprod(1.0 + arr)
        ax.plot(dates, vami, linewidth=1.2, label=label)

    ax.set_title("VAMI (Base 100) — All L/S Models vs US Market")
    ax.set_ylabel("VAMI (Base = 100)")
    ax.grid(True, alpha=0.3)

    locator   = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.legend()
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


# ------------- Main -------------
def main():
    mkt = read_ff5_market(FF5_CSV)

    # Start with market as base DF
    df = mkt.clone()
    cols_info: list[dict] = []

    for spec in MODEL_SPECS:
        colname = f"{spec['key']}_ret"
        s_df = read_ls_series(spec["path"], colname)
        if s_df is None or s_df.is_empty():
            continue
        df = df.join(s_df, on="DATE", how="inner")
        cols_info.append({"col": colname, "label": spec["label"]})

    if not cols_info:
        raise RuntimeError("No strategy return series found; nothing to plot.")

    if df.is_empty():
        raise RuntimeError("No overlapping dates across market and strategy return series.")

    plot_vami_all(df, cols_info, OUT_IMG)
    print(f"Saved VAMI plot to {OUT_IMG}")


if __name__ == "__main__":
    main()
