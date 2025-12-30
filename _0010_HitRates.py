# _0010_ModelHitRates.py
# Compute per-model hit rates (position-level winners/losers over ~12m horizon)

from __future__ import annotations

from pathlib import Path
from datetime import date
import math
import re

import polars as pl
import numpy as np  # not strictly needed but kept for symmetry


BASE   = Path(".")
RET_DS = BASE / "FilteredRawData" / "Returns_ds"
"""
MODEL_SPECS = [
    {"name": "Base",          "tb_dir": BASE / "BaseTopBottom10"},
    {"name": "LightGBM",      "tb_dir": BASE / "LightGBMTopBottom10"},
    {"name": "Ridge",         "tb_dir": BASE / "RidgeTopBottom10"},
    {"name": "LightGBMScore", "tb_dir": BASE / "LightGBMScoreTopBottom10"},
    {"name": "RidgeScore",    "tb_dir": BASE / "RidgeScoreTopBottom10"},
]
"""
MODEL_SPECS = [

    {"name": "Ridge",         "tb_dir": BASE / "RidgeTopBottom10"},

]

START_YEAR: int | None = None   # e.g. 1990
END_YEAR:   int | None = None   # e.g. 2024


def parse_tag(p: Path) -> date:
    m = re.search(r"(\d{8})", p.name)
    if not m:
        raise ValueError(f"Cannot parse yyyymmdd from {p.name}")
    y, mth, d = int(m.group(1)[:4]), int(m.group(1)[4:6]), int(m.group(1)[6:8])
    return date(y, mth, d)


def list_tb_files(tb_dir: Path) -> list[Path]:
    files: list[Path] = []
    if not tb_dir.exists():
        return files
    for ydir in sorted(tb_dir.glob("year=*")):
        y_str = ydir.name.split("=")[-1]
        try:
            y = int(y_str)
        except ValueError:
            continue
        if START_YEAR is not None and y < START_YEAR:
            continue
        if END_YEAR   is not None and y > END_YEAR:
            continue
        files += sorted(ydir.glob("topbottom10_*.parquet"))
    return sorted(files, key=parse_tag)


def year_paths(ds_dir: Path, years) -> list[str]:
    out: list[str] = []
    for y in years:
        d = ds_dir / f"year={y}"
        if d.exists():
            out += [str(p) for p in d.glob("*.parquet")]
    return out


def parse_date_expr(col: str) -> pl.Expr:
    c = pl.col(col)
    return pl.coalesce(
        [
            c.cast(pl.Date, strict=False),
            c.cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False),
            c.cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d",   strict=False),
        ]
    )


def build_month_index(tb_files: list[Path]) -> dict[tuple[int, int], date]:
    idx: dict[tuple[int, int], date] = {}
    for f in tb_files:
        dt = parse_tag(f)
        idx[(dt.year, dt.month)] = dt
    return idx


def compute_hits_for_model(name: str, tb_dir: Path) -> dict[str, float]:
    tb_files = list_tb_files(tb_dir)
    if not tb_files:
        print(f"[{name}] No TopBottom10 files in {tb_dir}, skipping.")
        return {
            "Model": name,
            "GlobalHitRate": math.nan,
            "LongHitRate": math.nan,
            "ShortHitRate": math.nan,
        }

    month_idx = build_month_index(tb_files)

    long_hits = long_n = 0
    short_hits = short_n = 0

    for f in tb_files:
        asof = parse_tag(f)
        end = month_idx.get((asof.year + 1, asof.month))
        if end is None:
            end = date(asof.year + 1, asof.month, min(asof.day, 28))

        snap = (
            pl.read_parquet(str(f))
            .select(
                PERMNO=pl.col("permno").cast(pl.Int64),
                signal=pl.col("signal").cast(pl.Int8),
            )
            .drop_nulls(subset=["PERMNO", "signal"])
        )
        if snap.is_empty():
            continue

        files = year_paths(RET_DS, range(asof.year, end.year + 1))
        if not files:
            continue

        panel = (
            pl.scan_parquet(files, low_memory=True)
            .select(
                PERMNO = pl.col("PERMNO").cast(pl.Int64),
                DATE   = parse_date_expr("DATE"),
                RET    = pl.col("RET"),
                DLRET  = pl.col("DLRET").cast(pl.Float64, strict=False),
            )
            .with_columns(
                TOTRET = (
                    (1.0 + pl.col("RET").fill_null(0.0))
                    * (1.0 + pl.col("DLRET").fill_null(0.0))
                    - 1.0
                )
            )
            .select("PERMNO", "DATE", "TOTRET")
            .filter(
                (pl.col("DATE") > pl.lit(asof))
                & (pl.col("DATE") < pl.lit(end))
            )
            .join(snap.lazy(), on="PERMNO", how="inner")
            .collect(engine="streaming")
        )
        if panel.is_empty():
            continue

        panel = (
            panel.sort(["PERMNO", "DATE"])
            .with_columns(
                GROSS=(1.0 + pl.col("TOTRET")).cum_prod().over("PERMNO")
            )
        )

        final = (
            panel.group_by(["PERMNO", "signal"])
            .agg(pl.col("GROSS").last().alias("GROSS_FINAL"))
            .drop_nulls(subset=["GROSS_FINAL"])
            .with_columns(totret_final=pl.col("GROSS_FINAL") - 1.0)
        )
        if final.is_empty():
            continue

        df = final.select(["signal", "totret_final"]).to_pandas()
        if df.empty:
            continue

        long_mask = df["signal"] == 1
        short_mask = df["signal"] == -1

        long_hits += int((df.loc[long_mask, "totret_final"] > 0.0).sum())
        long_n    += int(long_mask.sum())

        # Short wins when underlying goes down over horizon
        short_hits += int((df.loc[short_mask, "totret_final"] < 0.0).sum())
        short_n    += int(short_mask.sum())

    def _rate(h: int, n: int) -> float:
        return float(h) / float(n) if n > 0 else math.nan

    global_hits = long_hits + short_hits
    global_n    = long_n + short_n

    return {
        "Model": name,
        "GlobalHitRate": _rate(global_hits, global_n),
        "LongHitRate": _rate(long_hits, long_n),
        "ShortHitRate": _rate(short_hits, short_n),
    }


def main():
    import pandas as pd

    rows = [compute_hits_for_model(spec["name"], spec["tb_dir"]) for spec in MODEL_SPECS]
    df = pd.DataFrame(rows, columns=["Model", "GlobalHitRate", "LongHitRate", "ShortHitRate"])
    out_path = BASE / "ModelHitRates.xlsx"
    df.to_excel(out_path, index=False)
    print(f"Saved hit-rate summary to {out_path}")

if __name__ == "__main__":
    main()
