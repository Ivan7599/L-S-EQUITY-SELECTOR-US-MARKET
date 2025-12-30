# -*- coding: utf-8 -*-
# Unified multi-model Performance script — Buy&hold sleeves, snapshot-day rebal
#
# This is the merge of:
#   - _0006_Performance.py      (Base, LightGBM, Ridge)
#   - _0006_Performance-add.py  (LightGBMScore, RidgeScore)
#
# Inputs (unchanged):
#   - ./BaseTopBottom10/year=*/topbottom10_YYYYMMDD.parquet
#   - ./LightGBMTopBottom10/year=*/topbottom10_YYYYMMDD.parquet
#   - ./RidgeTopBottom10/year=*/topbottom10_YYYYMMDD.parquet
#   - ./LightGBMScoreTopBottom10/year=*/topbottom10_YYYYMMDD.parquet
#   - ./RidgeScoreTopBottom10/year=*/topbottom10_YYYYMMDD.parquet
#   - ./FilteredRawData/Returns_ds/year=*/...parquet
#   - ./RawDatabase/Factors5.csv
#
# Outputs (unchanged, same names/paths):
#   - ./FilteredRawData/<Model>LS_Returns.parquet
#   - ./<Model>PerfMeasure.csv
#   - ./<Model>VAMI.jpg
#   - ./FilteredRawData/<Model>Long_Returns.parquet
#   - ./FilteredRawData/<Model>Short_Returns.parquet
#   - ./<Model>VAMI_Long.jpg
#   - ./<Model>VAMI_Short.jpg

from __future__ import annotations

from pathlib import Path
from datetime import date
import io, re, math
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------- Paths ----------
RET_DS  = Path("./FilteredRawData/Returns_ds")   # CRSP daily (partitioned by year=YYYY/)
RAW_DIR = Path("./RawDatabase")
FF5_CSV = RAW_DIR / "Factors5.csv"               # DATE,Mkt_RF,SMB,HML,RMW,CMA,RF (in percent)

# ---------- Model specs (merged) ----------
MODEL_SPECS = [
    # Base / "economic logic" model
    {
        "name": "Base",
        "tb_dir": Path("./BaseTopBottom10"),
        "out_ret": Path("./FilteredRawData/BaseLS_Returns.parquet"),
        "out_csv": Path("./BasePerfMeasure.csv"),
        "out_img": Path("./BaseVAMI.jpg"),
        "out_long_ret": Path("./FilteredRawData/BaseLong_Returns.parquet"),
        "out_short_ret": Path("./FilteredRawData/BaseShort_Returns.parquet"),
        "out_long_img": Path("./BaseVAMI_Long.jpg"),
        "out_short_img": Path("./BaseVAMI_Short.jpg"),
    },
    # LightGBM (pred_12m ranking)
    {
        "name": "LightGBM",
        "tb_dir": Path("./LightGBMTopBottom10"),
        "out_ret": Path("./FilteredRawData/LightGBMLS_Returns.parquet"),
        "out_csv": Path("./LightGBMPerfMeasure.csv"),
        "out_img": Path("./LightGBMVAMI.jpg"),
        "out_long_ret": Path("./FilteredRawData/LightGBMLong_Returns.parquet"),
        "out_short_ret": Path("./FilteredRawData/LightGBMShort_Returns.parquet"),
        "out_long_img": Path("./LightGBMVAMI_Long.jpg"),
        "out_short_img": Path("./LightGBMVAMI_Short.jpg"),
    },
    # Ridge (pred_12m ranking)
    {
        "name": "Ridge",
        "tb_dir": Path("./RidgeTopBottom10"),
        "out_ret": Path("./FilteredRawData/RidgeLS_Returns.parquet"),
        "out_csv": Path("./RidgePerfMeasure.csv"),
        "out_img": Path("./RidgeVAMI.jpg"),
        "out_long_ret": Path("./FilteredRawData/RidgeLong_Returns.parquet"),
        "out_short_ret": Path("./FilteredRawData/RidgeShort_Returns.parquet"),
        "out_long_img": Path("./RidgeVAMI_Long.jpg"),
        "out_short_img": Path("./RidgeVAMI_Short.jpg"),
    },
    # LightGBMScore (score-based variant)
    {
        "name": "LightGBMScore",
        "tb_dir": Path("./LightGBMScoreTopBottom10"),
        "out_ret": Path("./FilteredRawData/LightGBMScoreLS_Returns.parquet"),
        "out_csv": Path("./LightGBMScorePerfMeasure.csv"),
        "out_img": Path("./LightGBMScoreVAMI.jpg"),
        "out_long_ret": Path("./FilteredRawData/LightGBMScoreLong_Returns.parquet"),
        "out_short_ret": Path("./FilteredRawData/LightGBMScoreShort_Returns.parquet"),
        "out_long_img": Path("./LightGBMScoreVAMI_Long.jpg"),
        "out_short_img": Path("./LightGBMScoreVAMI_Short.jpg"),
    },
    # RidgeScore (score-based variant)
    {
        "name": "RidgeScore",
        "tb_dir": Path("./RidgeScoreTopBottom10"),
        "out_ret": Path("./FilteredRawData/RidgeScoreLS_Returns.parquet"),
        "out_csv": Path("./RidgeScorePerfMeasure.csv"),
        "out_img": Path("./RidgeScoreVAMI.jpg"),
        "out_long_ret": Path("./FilteredRawData/RidgeScoreLong_Returns.parquet"),
        "out_short_ret": Path("./FilteredRawData/RidgeScoreShort_Returns.parquet"),
        "out_long_img": Path("./RidgeScoreVAMI_Long.jpg"),
        "out_short_img": Path("./RidgeScoreVAMI_Short.jpg"),
    },
]

# Ensure output directories exist (unchanged behavior)
for spec in MODEL_SPECS:
    spec["tb_dir"].mkdir(parents=True, exist_ok=True)
    spec["out_ret"].parent.mkdir(parents=True, exist_ok=True)
    spec["out_long_ret"].parent.mkdir(parents=True, exist_ok=True)
    spec["out_short_ret"].parent.mkdir(parents=True, exist_ok=True)

# ---------- Helpers shared across models ----------

def parse_tag(p: Path) -> date:
    m = re.search(r"(\d{8})", p.name)
    if not m:
        raise ValueError(f"Bad filename: {p.name}")
    y, mth, d = int(m.group(1)[:4]), int(m.group(1)[4:6]), int(m.group(1)[6:8])
    return date(y, mth, d)

def list_tb_files(tb_dir: Path) -> list[Path]:
    xs = []
    for ydir in sorted(tb_dir.glob("year=*")):
        xs += sorted(ydir.glob("topbottom10_*.parquet"))
        xs += sorted(ydir.glob("topbottom20_*.parquet"))
    return sorted(xs, key=parse_tag)

def year_paths(ds_dir: Path, years):
    out = []
    for y in years:
        d = ds_dir / f"year={y}"
        if d.exists():
            out += [str(p) for p in d.glob("*.parquet")]
    return out

def parse_date_expr(col: str) -> pl.Expr:
    c = pl.col(col)
    return pl.coalesce([
        c.cast(pl.Date, strict=False),
        c.cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False),
        c.cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d",   strict=False),
    ])

def build_month_index(files: list[Path]) -> dict[tuple[int, int], date]:
    idx = {}
    for f in files:
        dt = parse_tag(f)
        idx[(dt.year, dt.month)] = dt   # ≤1 snapshot per month
    return idx

def build_cohort_sleeves(tb_files, month_idx):
    sleeves, counts = [], []
    for f in tb_files:
        asof = parse_tag(f)
        end  = month_idx.get((asof.year + 1, asof.month))
        if end is None:
            end = date(asof.year + 1, asof.month, min(asof.day, 28))

        snap = (
            pl.read_parquet(str(f))
            .select(
                pl.col("permno").cast(pl.Int64).alias("PERMNO"),
                pl.col("signal").cast(pl.Int8).alias("signal"),
            )
            .drop_nulls(subset=["PERMNO", "signal"])
        )
        if snap.is_empty():
            continue

        n_long  = int(snap.filter(pl.col("signal") == 1).height)
        n_short = int(snap.filter(pl.col("signal") == -1).height)
        counts.append({"ASOF": asof, "END": end, "n_long": n_long, "n_short": n_short})

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
            panel.sort(["signal", "PERMNO", "DATE"])
            .with_columns(
                GROSS=(1.0 + pl.col("TOTRET")).cum_prod().over(["signal", "PERMNO"])
            )
        )
        nav = (
            panel.group_by(["DATE", "signal"])
            .agg(pl.col("GROSS").mean().alias("NAV"))
            .sort("DATE")
        )
        coh = nav.pivot(index="DATE", on="signal", values="NAV")

        ren = {}
        if "1" in coh.columns:
            ren["1"] = "NAV_LONG"
        if "-1" in coh.columns:
            ren["-1"] = "NAV_SHORT"
        if ren:
            coh = coh.rename(ren)
        if "NAV_LONG" not in coh.columns:
            coh = coh.with_columns(pl.lit(None).alias("NAV_LONG"))
        if "NAV_SHORT" not in coh.columns:
            coh = coh.with_columns(pl.lit(None).alias("NAV_SHORT"))

        coh = (
            coh.sort("DATE")
            .with_columns(
                long_ret=(pl.col("NAV_LONG") / pl.col("NAV_LONG").shift(1) - 1.0),
                short_ret=(pl.col("NAV_SHORT") / pl.col("NAV_SHORT").shift(1) - 1.0),
            )
            .drop_nulls(subset=["long_ret", "short_ret"])
        )

        if coh.is_empty():
            continue

        coh = (
            coh.with_columns(pl.lit(asof).alias("ASOF"), pl.lit(end).alias("END"))
            .select(["DATE", "ASOF", "END", "long_ret", "short_ret"])
        )
        sleeves.append(coh)

    sleeves_df = (
        pl.concat(sleeves, how="vertical_relaxed")
        if sleeves
        else pl.DataFrame(
            {"DATE": [], "ASOF": [], "END": [], "long_ret": [], "short_ret": []}
        )
    )
    counts_df = (
        pl.DataFrame(counts)
        if counts
        else pl.DataFrame({"ASOF": [], "END": [], "n_long": [], "n_short": []})
    )
    return sleeves_df.sort(["DATE", "ASOF"]), counts_df

def build_daily_ls(tb_files, month_idx) -> pl.DataFrame:
    sleeves, counts = build_cohort_sleeves(tb_files, month_idx)
    if sleeves.is_empty() or counts.is_empty():
        return pl.DataFrame(
            {"DATE": [], "long_ret": [], "short_ret": [], "ls_ret": []}
        )

    rebals = sorted({parse_tag(f) for f in tb_files})
    reb_df = pl.DataFrame({"REBAL": rebals}).sort("REBAL")

    live = (
        reb_df.join(counts, how="cross")
        .filter(
            (pl.col("ASOF") < pl.col("REBAL"))
            & (pl.col("REBAL") < pl.col("END"))
        )
    )

    long_norm = live.group_by("REBAL").agg(
        pl.col("n_long").sum().alias("N_L")
    )
    short_norm = live.group_by("REBAL").agg(
        pl.col("n_short").sum().alias("N_S")
    )

    live = (
        live.join(long_norm, on="REBAL", how="left")
        .join(short_norm, on="REBAL", how="left")
        .with_columns(
            W_LONG=pl.when(pl.col("N_L") > 0)
            .then(pl.col("n_long") / pl.col("N_L"))
            .otherwise(0.0),
            W_SHORT=pl.when(pl.col("N_S") > 0)
            .then(pl.col("n_short") / pl.col("N_S"))
            .otherwise(0.0),
        )
        .select(["REBAL", "ASOF", "W_LONG", "W_SHORT"])
    )

    sw = (
        sleeves.sort("DATE")
        .join_asof(
            reb_df,
            left_on="DATE",
            right_on="REBAL",
            strategy="backward",
        )
        .drop_nulls(subset=["REBAL"])
    )
    sw = (
        sw.join(live, on=["REBAL", "ASOF"], how="left")
        .with_columns(
            pl.col("W_LONG").fill_null(0.0),
            pl.col("W_SHORT").fill_null(0.0),
        )
    )

    daily = (
        sw.group_by("DATE")
        .agg(
            (pl.col("long_ret") * pl.col("W_LONG")).sum().alias("long_ret"),
            (pl.col("short_ret") * pl.col("W_SHORT")).sum().alias("short_ret"),
        )
        .with_columns(
            (pl.col("long_ret") - pl.col("short_ret")).alias("ls_ret")
        )
        .drop_nulls(subset=["ls_ret"])
        .sort("DATE")
    )

    return daily.select(["DATE", "long_ret", "short_ret", "ls_ret"])

# ---------- Fama-French read ----------

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
    return pl.read_csv(
        io.StringIO(header + "\n" + "\n".join(rows)),
        dtypes=dtypes,
    )

def read_ff5_market(path: Path) -> pl.DataFrame:
    """
    Returns DATE (pl.Date) and mkt_ret (decimal daily return).
    We interpret Fama-French 'Mkt_RF' (market minus RF) as the market factor.
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

# ---------- Stats / plotting helpers ----------

def _compute_stats(arr: np.ndarray):
    mu_d  = float(np.mean(arr)) if arr.size else float("nan")
    sig_d = float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")
    ann_sharpe = (
        (mu_d / sig_d) * math.sqrt(252)
        if (sig_d and sig_d > 0)
        else float("nan")
    )
    return mu_d, sig_d, ann_sharpe

def _sortino_ratio(arr: np.ndarray):
    if arr.size == 0:
        return float("nan")
    downside = np.where(arr < 0.0, arr, 0.0)
    down_std = np.std(downside, ddof=1) if arr.size > 1 else float("nan")
    if down_std and down_std > 0:
        return (np.mean(arr) / down_std) * math.sqrt(252)
    return float("nan")

def _info_ratio(arr: np.ndarray, bench: np.ndarray):
    if arr.size == 0 or bench.size == 0:
        return float("nan")
    active = arr - bench
    act_std = np.std(active, ddof=1) if active.size > 1 else float("nan")
    if act_std and act_std > 0:
        return (np.mean(active) / act_std) * math.sqrt(252)
    return float("nan")

def _max_drawdown_with_dates(arr: np.ndarray, dates_list):
    """
    Compute max drawdown (most negative peak-to-trough return),
    and return its start (peak) date and end (trough) date.
    Returns (max_dd, dd_start_date, dd_end_date).
    """
    if arr.size == 0:
        return float("nan"), None, None

    wealth = np.cumprod(1.0 + arr)
    high_watermark = wealth[0]
    high_idx = 0

    max_dd = 0.0
    dd_start_idx = 0
    dd_end_idx = 0

    for i in range(1, wealth.size):
        if wealth[i] > high_watermark:
            high_watermark = wealth[i]
            high_idx = i

        cur_dd = wealth[i] / high_watermark - 1.0

        if cur_dd < max_dd:
            max_dd = cur_dd
            dd_start_idx = high_idx
            dd_end_idx = i

    start_date = dates_list[dd_start_idx]
    end_date   = dates_list[dd_end_idx]
    return float(max_dd), start_date, end_date

def _plot_vami_single(dates, rets, out_path: Path, title: str):
    vami = 100.0 * np.cumprod(1.0 + rets)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dates, vami, linewidth=1.6)
    ax.set_title(title)
    ax.set_ylabel("Base 100")
    ax.grid(True, alpha=0.3)
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)

def _plot_vami_dual(dates, ls_rets, mkt_rets, out_path: Path, title: str):
    vami_ls  = 100.0 * np.cumprod(1.0 + ls_rets)
    vami_mkt = 100.0 * np.cumprod(1.0 + mkt_rets)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dates, vami_ls,  color="#8B0000", linewidth=1.6, label="L / S")
    ax.plot(dates, vami_mkt, color="black",  linewidth=1.2, linestyle="--", label="US Mkt FF")
    ax.set_title(title)
    ax.set_ylabel("Base 100")
    ax.grid(True, alpha=0.3)
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)

# ---------- Per-model runner ----------

def run_for_model(spec: dict, mkt_df: pl.DataFrame) -> None:
    name          = spec["name"]
    tb_dir        = spec["tb_dir"]
    OUT_RET       = spec["out_ret"]
    OUT_CSV       = spec["out_csv"]
    OUT_IMG       = spec["out_img"]
    OUT_LONG_RET  = spec["out_long_ret"]
    OUT_SHORT_RET = spec["out_short_ret"]
    OUT_LONG_IMG  = spec["out_long_img"]
    OUT_SHORT_IMG = spec["out_short_img"]

    print(f"\n=== Processing model: {name} ===")

    tb_files = list_tb_files(tb_dir)
    if not tb_files:
        print(f"[{name}] No cohort files found in {tb_dir}.")
        return

    month_idx = build_month_index(tb_files)
    daily = build_daily_ls(tb_files, month_idx)
    if daily.is_empty():
        print(f"[{name}] No daily L–S series.")
        return

    # Save L/S series
    daily.select(["DATE", "ls_ret"]).write_parquet(str(OUT_RET))

    # Leg return series (P&L conventions)
    long_df  = daily.select(["DATE", "long_ret"])
    short_df = daily.select(["DATE", (-pl.col("short_ret")).alias("short_ret")])
    long_df.write_parquet(str(OUT_LONG_RET))
    short_df.write_parquet(str(OUT_SHORT_RET))

    # Align with market factor
    mkt_df = mkt_df.sort("DATE")
    aligned = daily.join(mkt_df, on="DATE", how="inner").sort("DATE")
    if aligned.is_empty():
        print(f"[{name}] No overlap between strategy and Fama-French market factor.")
        return

    dates_aligned = aligned["DATE"].to_list()
    ls_arr        = aligned["ls_ret"].to_numpy()
    mkt_arr       = aligned["mkt_ret"].to_numpy()

    lng_arr = (
        long_df.sort("DATE")
        .join(aligned.select("DATE"), on="DATE", how="inner")["long_ret"]
        .to_numpy()
    )
    sht_arr = (
        short_df.sort("DATE")
        .join(aligned.select("DATE"), on="DATE", how="inner")["short_ret"]
        .to_numpy()
    )

    # Stats
    mu_ls,  sig_ls,  shr_ls  = _compute_stats(ls_arr)
    mu_mkt, sig_mkt, shr_mkt = _compute_stats(mkt_arr)
    mu_lng, sig_lng, shr_lng = _compute_stats(lng_arr)
    mu_sht, sig_sht, shr_sht = _compute_stats(sht_arr)

    sor_ls  = _sortino_ratio(ls_arr)
    sor_mkt = _sortino_ratio(mkt_arr)

    ir_ls   = _info_ratio(ls_arr,  mkt_arr)
    ir_mkt  = _info_ratio(mkt_arr, mkt_arr)  # NaN, benchmark vs itself

    dd_ls,  dd_ls_start,  dd_ls_end   = _max_drawdown_with_dates(ls_arr,  dates_aligned)
    dd_mkt, dd_mkt_start, dd_mkt_end = _max_drawdown_with_dates(mkt_arr, dates_aligned)

    start_d = dates_aligned[0] if dates_aligned else None
    end_d   = dates_aligned[-1] if dates_aligned else None
    n_days  = len(dates_aligned)

    # Preserve legacy column name "avg_daily_ls"
    perf_df = pl.DataFrame({
        "series":       ["LS", "Mkt"],
        "model":        [name, name],
        "ann_sharpe":   [shr_ls,        shr_mkt],
        "sortino":      [sor_ls,        sor_mkt],
        "info_ratio":   [ir_ls,         ir_mkt],
        "avg_daily_ls": [mu_ls,         mu_mkt],
        "daily_vol":    [sig_ls,        sig_mkt],
        "max_drawdown": [dd_ls,         dd_mkt],
        "dd_start":     [dd_ls_start,   dd_mkt_start],
        "dd_end":       [dd_ls_end,     dd_mkt_end],
        "start_date":   [start_d,       start_d],
        "end_date":     [end_d,         end_d],
        "n_days":       [n_days,        n_days],
    })
    perf_df.write_csv(str(OUT_CSV))

    # Console report
    print(
        f"[{name}] L/S   | Sharpe {shr_ls:.4f} | Sortino {sor_ls:.4f} | IR {ir_ls:.4f} "
        f"| μ_d {mu_ls:.6f} | σ_d {sig_ls:.6f} | DD {dd_ls:.4f} "
        f"({dd_ls_start} -> {dd_ls_end}) | n={n_days}"
    )
    print(
        f"[{name}] Mkt   | Sharpe {shr_mkt:.4f} | Sortino {sor_mkt:.4f} | IR {ir_mkt} "
        f"| μ_d {mu_mkt:.6f} | σ_d {sig_mkt:.6f} | DD {dd_mkt:.4f} "
        f"({dd_mkt_start} -> {dd_mkt_end}) | n={n_days}"
    )
    print(
        f"[{name}] Long  | Sharpe {shr_lng:.4f} | μ_d {mu_lng:.6f} | σ_d {sig_lng:.6f} | n={len(lng_arr)}"
    )
    print(
        f"[{name}] Short | Sharpe {shr_sht:.4f} | μ_d {mu_sht:.6f} | σ_d {sig_sht:.6f} | n={len(sht_arr)}"
    )

    # Plots
    _plot_vami_dual(
        dates_aligned,
        ls_arr,
        mkt_arr,
        OUT_IMG,
        f"{name} L / S vs US Mkt FF VAMI",
    )
    _plot_vami_single(
        dates_aligned,
        lng_arr,
        OUT_LONG_IMG,
        f"{name} Long Leg VAMI (start=100)",
    )
    _plot_vami_single(
        dates_aligned,
        sht_arr,
        OUT_SHORT_IMG,
        f"{name} Short Leg VAMI (start=100)",
    )

    print(f"[{name}] Saved: {OUT_IMG}, {OUT_CSV}, {OUT_RET}")
    print(f"[{name}] Saved: {OUT_LONG_IMG}, {OUT_SHORT_IMG}, {OUT_LONG_RET}, {OUT_SHORT_RET}")

# ---------- main ----------

def main():
    # Read market factor once, reuse for every model
    mkt_df = read_ff5_market(FF5_CSV)
    for spec in MODEL_SPECS:
        run_for_model(spec, mkt_df)

if __name__ == "__main__":
    main()
