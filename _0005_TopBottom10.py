# -*- coding: utf-8 -*-
# All-model TopBottom snapshot builder (no liquidity screens, no performance)
# Models:
#   - Base:      BaseInvestmentRanking/year=*/invrank_YYYYMMDD.parquet
#   - LightGBM:  LightGBMInvestmentRanking/year=*/LightGBMrank_YYYYMMDD.parquet
#   - Ridge:     RidgeRanking/year=*/ridgerank_YYYYMMDD.parquet
#
# Outputs:
#   - BaseTopBottom10/year=YYYY/topbottom10_YYYYMMDD.parquet
#   - LightGBMTopBottom10/year=YYYY/topbottom10_YYYYMMDD.parquet
#   - RidgeTopBottom10/year=YYYY/topbottom10_YYYYMMDD.parquet

from __future__ import annotations
from pathlib import Path
import re, math
import polars as pl

# ------------------------------ paths ------------------------------
BASE = Path(".")

MODEL_SPECS = [
    # Base composite model (score + rank + gvkey)
    {
        "name":          "Base",
        "in_dir":        BASE / "BaseInvestmentRanking",
        "pattern":       "invrank_*.parquet",
        "score_col":     "score",
        "required_cols": {"permno", "score"},
        "out_dir":       BASE / "BaseTopBottom10",
        "out_cols":      ("gvkey", "permno", "score", "rank", "signal"),
    },
    # LightGBM model (pred_12m as score)
    {
        "name":          "LightGBM",
        "in_dir":        BASE / "LightGBMInvestmentRanking",
        "pattern":       "LightGBMrank_*.parquet",
        "score_col":     "pred_12m",
        "required_cols": {"date", "permno", "pred_12m"},
        "out_dir":       BASE / "LightGBMTopBottom10",
        "out_cols":      ("date", "permno", "signal"),
    },
    # Ridge model (pred_12m as score)
    {
        "name":          "Ridge",
        "in_dir":        BASE / "RidgeRanking",
        "pattern":       "ridgerank_*.parquet",
        "score_col":     "pred_12m",
        "required_cols": {"date", "permno", "pred_12m"},
        "out_dir":       BASE / "RidgeTopBottom10",
        "out_cols":      ("date", "permno", "signal"),
    },
]

# Create output dirs
for spec in MODEL_SPECS:
    spec["out_dir"].mkdir(parents=True, exist_ok=True)

# (optional) restrict years; set to None to process all
START_YEAR: int | None = None
END_YEAR:   int | None = None

# Keep the historical filename prefix to avoid breaking downstream readers
FILE_PREFIX = "topbottom10"  # contains 10% longs + 10% shorts = 20% total

# ----------------------------- utils -------------------------------
def parse_date_tag(path: Path) -> str:
    m = re.search(r"(\d{8})", path.name)
    if not m:
        raise ValueError(f"Cannot parse yyyymmdd from {path.name}")
    return m.group(1)


def list_rank_files(spec: dict) -> list[Path]:
    in_dir  = spec["in_dir"]
    pattern = spec["pattern"]
    xs: list[Path] = []

    for ydir in sorted(in_dir.glob("year=*")):
        y_str = ydir.name.split("=")[-1]
        try:
            y = int(y_str)
        except Exception:
            continue
        if START_YEAR is not None and y < START_YEAR:
            continue
        if END_YEAR   is not None and y > END_YEAR:
            continue
        xs += sorted(ydir.glob(pattern))

    # sort by date tag so we process in chronological order
    return sorted(xs, key=parse_date_tag)

# -------------------------- core builder ---------------------------
def build_topbottom_for_model(spec: dict) -> None:
    name          = spec["name"]
    score_col     = spec["score_col"]
    required_cols = spec["required_cols"]
    out_dir_root  = spec["out_dir"]
    out_cols      = spec["out_cols"]

    files = list_rank_files(spec)
    if not files:
        print(f"[warn] No ranking files found for {name}.")
        return

    for f in files:
        tag = parse_date_tag(f)
        year = int(tag[:4])

        out_dir = out_dir_root / f"year={year}"
        out_dir.mkdir(parents=True, exist_ok=True)
        outp = out_dir / f"{FILE_PREFIX}_{tag}.parquet"
        if outp.exists():
            print(f"[skip] {name}: {outp.name} exists")
            continue

        df = pl.read_parquet(str(f))
        cols = set(df.columns)
        if not required_cols.issubset(cols):
            print(f"[warn] {name}: {f.name} missing required columns {required_cols}; skipping.")
            continue

        # type hygiene & de-dup per permno (keep highest score)
        exprs = [pl.col("permno").cast(pl.Int64),
                 pl.col(score_col).cast(pl.Float64)]
        if "date" in cols:
            exprs.append(pl.col("date"))  # keep as-is; will cast in select if needed

        df = (
            df.with_columns(exprs)
              .drop_nulls(subset=["permno", score_col])
              .sort(score_col, descending=True)
              .unique(subset=["permno"], keep="first")
        )

        n = df.height
        if n == 0:
            print(f"[warn] {name}: {f.name} is empty after cleaning; skipping.")
            continue

        top_n = max(1, int(math.floor(0.10 * n)))
        bot_n = max(1, int(math.floor(0.10 * n)))

        # top by descending score
        top = df.head(top_n).with_columns(pl.lit(1).alias("signal"))

        # bottom by ascending score (ensure disjoint from top if sample is tiny)
        bottom_sorted = df.sort(score_col, descending=False)
        bot = bottom_sorted.head(bot_n)

        if bot.join(top.select("permno"), on="permno", how="inner").height > 0:
            # remove overlaps, then fill from next lowest
            bot = bot.filter(~pl.col("permno").is_in(top.get_column("permno")))
            if bot.height < bot_n:
                fill = (
                    bottom_sorted
                    .filter(~pl.col("permno").is_in(top.get_column("permno")))
                    .slice(bot.height, bot_n - bot.height)
                )
                bot = pl.concat([bot, fill], how="vertical_relaxed")

        bot = bot.with_columns(pl.lit(-1).alias("signal"))

        # align output schema (tolerate missing optional cols like gvkey/rank/date)
        keep = [c for c in out_cols if c in set(top.columns) | set(bot.columns) | {"signal"}]
        out = pl.concat([top, bot], how="vertical_relaxed").select(keep)

        out.write_parquet(str(outp))
        print(f"[OK] {name}: {tag} -> {outp} ({out.height:,} rows: {top_n} long, {bot_n} short)")

# ------------------------------ main -------------------------------
if __name__ == "__main__":
    for spec in MODEL_SPECS:
        build_topbottom_for_model(spec)
