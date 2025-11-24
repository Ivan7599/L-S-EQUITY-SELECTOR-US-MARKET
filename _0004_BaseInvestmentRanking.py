# _0004_InvestmentRanking.py — score-only composite (slope 45%, avg 35%, vol 20%)
# Requires: polars>=1.25, pyarrow
from __future__ import annotations
from pathlib import Path
import re, math
import polars as pl

BASE   = Path(".")
AR_DIR = BASE / "AnchorRanking"
OUTDIR = BASE / "InvestmentRanking"
OUTDIR.mkdir(parents=True, exist_ok=True)

WINDOW_N = 12            # number of anchors per rolling window (e.g., 3 years if quarterly)
MIN_FRAC = 0.60          # min fraction of window observations required per firm
def MIN_OBS(n: int) -> int: return max(3, int(math.ceil(n * MIN_FRAC)))

# -------- file discovery --------
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

# -------- robust standardization (winsor 1–99%, MAD; fallback std) --------
def robust_z(df: pl.DataFrame, col: str, invert: bool=False) -> pl.Series:
    s = df.get_column(col)
    if s.is_null().all():
        return pl.Series(name=f"z_{col}", values=[None]*s.len(), dtype=pl.Float64)
    lo = float(df.select(pl.col(col).quantile(0.01)).item())
    hi = float(df.select(pl.col(col).quantile(0.99)).item())
    med = float(df.select(pl.col(col).median()).item())
    mad = float(df.select((pl.col(col) - med).abs().median()).item())
    scale = 1.4826 * mad
    if not math.isfinite(scale) or scale <= 0:
        std = float(df.select(pl.col(col).std()).item())
        scale = std if (std and math.isfinite(std) and std > 0) else 1.0
    z = (s.clip(lo, hi) - med) / scale
    return (-z if invert else z).cast(pl.Float64).rename(f"z_{col}")

def add_std(df: pl.DataFrame, specs: list[tuple[str,bool]]) -> pl.DataFrame:
    return df.with_columns([robust_z(df, c, inv) for (c, inv) in specs])

# -------- per-window computation --------
def compute_window(files_win: list[Path]):
    n = len(files_win)
    min_obs = MIN_OBS(n)

    panels = []
    for t, f in enumerate(files_win):
        df = pl.read_parquet(str(f), columns=["gvkey","permno","score"])
        if df.is_empty():
            continue
        # Cross-sectional restandardization of score within the anchor
        med = float(df.select(pl.col("score").median()).item())
        df_dev = df.with_columns((pl.col("score") - med).alias("abs_dev_seed"))
        mad = float(df_dev.select(pl.col("abs_dev_seed").abs().median()).item())
        scale = 1.4826 * mad if (mad is not None and math.isfinite(mad)) else None
        if not scale or scale <= 0:
            std = float(df.select(pl.col("score").std()).item())
            scale = std if (std and math.isfinite(std) and std > 0) else 1.0
        df = df.with_columns(((pl.col("score") - med) / scale).alias("score_z"),
                             pl.lit(t).alias("t"))
        panels.append(df.select("gvkey","permno","score_z","t"))
    if not panels:
        return None, None, None

    panel = pl.concat(panels, how="vertical_relaxed").with_columns(
        pl.col("score_z").cast(pl.Float64),
        pl.col("t").cast(pl.Float64),
    )

    # OLS slope of score_z vs time (denom = k*Σt^2 − (Σt)^2)
    sums = (panel.group_by(["permno","gvkey"]).agg(
                pl.count().alias("k"),
                pl.col("t").sum().alias("sum_t"),
                (pl.col("t")**2).sum().alias("sum_t2"),
                pl.col("score_z").sum().alias("sum_s"),
                (pl.col("score_z")*pl.col("t")).sum().alias("sum_st"),
                pl.col("score_z").mean().alias("avg_score"),
            )
            .with_columns(denom = pl.col("k")*pl.col("sum_t2") - (pl.col("sum_t")**2))
            .with_columns(
                slope_score = pl.when((pl.col("denom") > 0) & (pl.col("k") >= min_obs)).then(
                    (pl.col("k")*pl.col("sum_st") - pl.col("sum_t")*pl.col("sum_s")) / pl.col("denom")
                ).otherwise(None),
                avg_score = pl.when(pl.col("k") >= max(1, min_obs)).then(pl.col("avg_score")).otherwise(None),
            )
            .select(["permno","gvkey","slope_score","avg_score","k"])
           )

    # Volatility of Δscore_z (lower is better)
    panel = (panel.sort(["permno","gvkey","t"])
                  .with_columns(pl.col("score_z").shift(1).over(["permno","gvkey"]).alias("score_z_lag"))
                  .with_columns((pl.col("score_z") - pl.col("score_z_lag")).alias("dscore_z")))

    diffs = (panel.group_by(["permno","gvkey"]).agg(
                pl.col("dscore_z").drop_nulls().std().alias("vol_dscore"),
                pl.col("dscore_z").drop_nulls().count().alias("nd")
            )
            .with_columns(pl.when(pl.col("nd") >= max(1, min_obs-1))
                           .then(pl.col("vol_dscore"))
                           .otherwise(None)
                           .alias("vol_dscore"))
            .select(["permno","gvkey","vol_dscore"])
            )

    xsec = sums.join(diffs, on=["permno","gvkey"], how="left")
    if xsec.is_empty():
        return None, None, None

    # Standardize features; invert volatility so "higher z" = better
    xsec = add_std(xsec, [
        ("slope_score", False),   # improving score => good
        ("avg_score",   False),   # higher level => good
        ("vol_dscore",  True),    # lower volatility => better
    ])

    # Composite (row-wise reweighting): 45% slope, 35% level, 20% stability
    zcols = ["z_slope_score","z_avg_score","z_vol_dscore"]
    wts   = [1/3, 1/3, 1/3] # old was .45, .35, .20
    masks = [pl.col(c).is_not_null().cast(pl.Float64) for c in zcols]
    num = pl.sum_horizontal([pl.col(c)*w*m for c,w,m in zip(zcols,wts,masks)])
    den = pl.sum_horizontal([pl.lit(w)*m for w,m in zip(wts,masks)])
    xsec = (xsec.with_columns(pl.when(den > 0).then(num/den).otherwise(None).alias("score"))
                 .select(["gvkey","permno","score"])
                 .filter(pl.col("score").is_not_null()))

    if xsec.is_empty():
        return None, None, None

    last = files_win[-1]
    year = int(last.parent.name.split("=")[1])
    date_tag = parse_date_tag(last)
    return xsec, year, date_tag

def main():
    files = list_anchor_files()
    if len(files) < WINDOW_N:
        print("Not enough anchors to compute investment ranking.")
        return
    files = sorted(files, key=parse_date_tag)

    for i in range(WINDOW_N-1, len(files)):
        win = files[i-WINDOW_N+1 : i+1]
        xsec, year, date_tag = compute_window(win)
        if xsec is None:
            continue
        outy = OUTDIR / f"year={year}"
        outy.mkdir(parents=True, exist_ok=True)
        outp = outy / f"invrank_{date_tag}.parquet"
        xsec = xsec.sort("score", descending=True).with_row_count("rank", offset=1)
        xsec.select(["rank","gvkey","permno","score"]).write_parquet(str(outp))
        print(f"[OK] {date_tag} -> {outp} ({xsec.height:,} rows)")

if __name__ == "__main__":
    main()
