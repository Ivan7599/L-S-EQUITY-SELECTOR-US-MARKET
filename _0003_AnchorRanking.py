# _0003_AnchorRanking.py
# Fast composite scoring & ranking per anchor date (updated for multiples: higher=worse)
from __future__ import annotations
from pathlib import Path
import re, math
import polars as pl

BASE = Path(".")
RATIOS_DIR = BASE / "Ratios"
OUT_DIR = BASE / "AnchorRanking"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def list_ratio_files() -> list[Path]:
    files = []
    for ydir in sorted(RATIOS_DIR.glob("year=*")):
        files.extend(sorted(ydir.glob("ratios_*.parquet")))
    return files

def parse_anchor_date_from_name(path: Path) -> str:
    m = re.search(r"ratios_(\d{8})\.parquet$", path.name)
    if not m:
        raise ValueError(f"Unexpected file name: {path.name}")
    return m.group(1)

def robust_z_from_series(df: pl.DataFrame, col: str, invert: bool=False) -> pl.Series:
    s = df.get_column(col)
    if s.null_count() == s.len():
        return pl.Series(name=f"z_{col}", values=[None]*s.len(), dtype=pl.Float64)
    lo = float(df.select(pl.col(col).quantile(0.01)).item())
    hi = float(df.select(pl.col(col).quantile(0.99)).item())
    med = float(df.select(pl.col(col).median()).item())
    mad = float(df.select((pl.col(col) - med).abs().median()).item())
    scale = 1.4826 * mad
    if not math.isfinite(scale) or scale <= 0:
        std = float(df.select(pl.col(col).std()).item())
        scale = std if (std is not None and math.isfinite(std) and std > 0) else 1.0
    x = s.clip(lo, hi)
    z = (x - med) / scale
    if invert:
        z = -z
    return z.cast(pl.Float64).rename(f"z_{col}")

def add_std_cols(df: pl.DataFrame, specs: list[tuple[str, bool]]) -> pl.DataFrame:
    cols = [robust_z_from_series(df, c, invert=inv) for (c, inv) in specs]
    return df.with_columns(cols)

def weighted_row_mean(df: pl.DataFrame, zcols: list[str], wts: list[float], outname: str) -> pl.DataFrame:
    masks = [pl.col(c).is_not_null().cast(pl.Float64) for c in zcols]
    num = pl.sum_horizontal([pl.col(c) * w * m for c, w, m in zip(zcols, wts, masks)])
    den = pl.sum_horizontal([pl.lit(w) * m for w, m in zip(wts, masks)])
    return df.with_columns(pl.when(den > 0).then(num/den).otherwise(None).alias(outname))

def build_score(df: pl.DataFrame) -> pl.DataFrame:
    # Choose strict when available, else FLEX (or alias)
    df = df.with_columns([
        pl.coalesce([pl.col("CFO_AT"), pl.col("CFO_AT_FLEX")]).alias("CFO_AT_use"),
        pl.coalesce([pl.col("PC"), pl.col("PC_FLEX")]).alias("PC_use"),
        pl.coalesce([pl.col("SloanAcc_AT"), pl.col("SloanAcc_AT_FLEX")]).alias("SloanAcc_use"),
        pl.coalesce([pl.col("NOA_AT"), pl.col("NOA_AT_FLEX")]).alias("NOA_AT_use"),
        pl.coalesce([pl.col("EV_EBITDA_STRICT"), pl.col("EV_EBITDA"), pl.col("EV_EBITDA_FLEX")]).alias("EV_EBITDA_use"),
    ])

    # Standardize: higher=better. For valuation multiples (MB, PE, PC, EV/EBITDA), higher=worse → invert=True
    std_specs = [
        ("GP_AT", False), ("OP_BE", False), ("ROA_TTM", False), ("CFO_AT_use", False),
        ("MB", True), ("PE", True), ("PC_use", True), ("EV_EBITDA_use", True),
        ("SloanAcc_use", True), ("AssetGrowth", True), ("NOA_AT_use", True), ("NetShareIss", True),
    ]
    df = add_std_cols(df, std_specs)

    # Profitability (1/3): GP 35, OP_BE 30, ROA 20, CFO 15
    P_cols = ["z_GP_AT","z_OP_BE","z_ROA_TTM","z_CFO_AT_use"]; P_w = [1/12,1/12,1/12,1/12]
    df = weighted_row_mean(df, P_cols, P_w, "score_P")

    # Valuation (1/3): MB 35, PE 35, PC 20, EV/EBITDA 10 (already inverted)
    V_cols = ["z_MB","z_PE","z_PC_use","z_EV_EBITDA_use"]; V_w = [1/12,1/12,1/12,1/12]
    df = weighted_row_mean(df, V_cols, V_w, "score_V")

    # Quality (1/3): Sloan 25, AssetGrowth 35, NOA 25, NetShareIss 15 (all inverted)
    Q_cols = ["z_SloanAcc_use","z_AssetGrowth","z_NOA_AT_use","z_NetShareIss"]; Q_w = [1/12,1/12,1/12,1/12]
    df = weighted_row_mean(df, Q_cols, Q_w, "score_Q")

    # Combine categories; require at least 2 present
    cat_cols, cat_wts = ["score_P","score_V","score_Q"], [1/3, 1/3, 1/3]
    masks = [pl.col(c).is_not_null().cast(pl.Float64) for c in cat_cols]
    num = pl.sum_horizontal([pl.col(c)*w*m for c,w,m in zip(cat_cols,cat_wts,masks)])
    den = pl.sum_horizontal([pl.lit(w)*m for w,m in zip(cat_wts,masks)])
    return df.with_columns(
        pl.when((masks[0]+masks[1]+masks[2]) >= 2)
          .then(pl.when(den > 0).then(num/den).otherwise(None))
          .otherwise(None)
          .alias("score")
    )

def rank_and_save(path_in: Path):
    ydir = path_in.parent
    year = int(ydir.name.split("=")[1])
    date_tag = parse_anchor_date_from_name(path_in)

    df = pl.read_parquet(str(path_in))
    if df.is_empty():
        return

    keep = [
        "gvkey","permno","GP_AT","OP_BE","ROA_TTM","CFO_AT","CFO_AT_FLEX",
        "MB","PE","PC","PC_FLEX","EV_EBITDA","EV_EBITDA_STRICT","EV_EBITDA_FLEX",
        "SloanAcc_AT","SloanAcc_AT_FLEX","AssetGrowth","NOA_AT","NOA_AT_FLEX","NetShareIss"
    ]
    present = [c for c in keep if c in df.columns]
    df = df.select(present + [c for c in ("gvkey","permno") if c not in present])

    df = build_score(df).select("gvkey","permno","score").filter(pl.col("score").is_not_null())
    if df.is_empty():
        return

    df = df.sort("score", descending=True).with_row_count("rank", offset=1)
    outy = OUT_DIR / f"year={year}"
    outy.mkdir(parents=True, exist_ok=True)
    out_path = outy / f"ancrank_{date_tag}.parquet"
    df.select(["rank","gvkey","permno","score"]).write_parquet(str(out_path))
    print(f"[OK] {date_tag} -> {out_path} ({df.height:,} rows)")

def main():
    for f in list_ratio_files():
        rank_and_save(f)

if __name__ == "__main__":
    main()
