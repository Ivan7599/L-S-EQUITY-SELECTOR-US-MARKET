# _0002_RatiosCalculation.py — final, with NOA_FLEX two-pass fix (valuation flipped to price/multiples)
from __future__ import annotations
from pathlib import Path
from datetime import date, timedelta
import polars as pl

BASE = Path(".")
FUND_DS = BASE / "FilteredRawData" / "Fundamentals_ds"
RET_DS  = BASE / "FilteredRawData" / "Returns_ds"
OUTDIR  = BASE / "Ratios"
OUTDIR.mkdir(parents=True, exist_ok=True)

ANCHORS = [(4,15),(5,20),(8,19),(11,19)]
YEARS   = range(1980, 2026)

def year_paths(ds_dir: Path, years) -> list[str]:
    paths = []
    for y in years:
        p = ds_dir / f"year={y}"
        if p.exists():
            paths += [str(pp) for pp in p.glob("*.parquet")]
    return paths

def parse_yyyymmdd(col: str) -> pl.Expr:
    c = pl.col(col)
    return pl.coalesce([
        c.cast(pl.Date, strict=False),
        c.cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False),
        c.cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False),
    ])

def safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return pl.when(den.is_finite() & (den.abs() > 0)).then(num/den).otherwise(None)

def pos_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return pl.when(den.is_finite() & (den > 0)).then(num/den).otherwise(None)

def next_trading_day(anchor: date) -> date | None:
    files = year_paths(RET_DS, [anchor.year-1, anchor.year, anchor.year+1])
    if not files: return None
    cal = (pl.scan_parquet(files)
           .select(DATE=parse_yyyymmdd("DATE"), VWRETD=pl.col("VWRETD"))
           .filter(pl.col("VWRETD").is_not_null())
           .select("DATE").unique().sort("DATE")).collect()
    for d in cal.to_series().to_list():
        if d >= anchor: return d
    return None

def prior_trading_day_approx_1y(anchor_td: date) -> date | None:
    target = anchor_td - timedelta(days=365)
    files = year_paths(RET_DS, [target.year-1, target.year, target.year+1])
    if not files: return None
    cal = (pl.scan_parquet(files)
           .select(DATE=parse_yyyymmdd("DATE"), VWRETD=pl.col("VWRETD"))
           .filter(pl.col("VWRETD").is_not_null())
           .select("DATE").unique().sort("DATE")).collect()
    days = cal.to_series().to_list()
    prev = [d for d in days if d <= target]
    return prev[-1] if prev else None

def crsp_snapshot_lazy(at_date: date | None) -> pl.LazyFrame:
    if at_date is None:
        return pl.DataFrame({"PERMNO":pl.Series([],dtype=pl.Int64),
                             "DATE":pl.Series([],dtype=pl.Date),
                             "PRC":pl.Series([],dtype=pl.Float64),
                             "SHROUT":pl.Series([],dtype=pl.Float64),
                             "CFACSHR":pl.Series([],dtype=pl.Float64)}).lazy()
    files = year_paths(RET_DS, [at_date.year])
    return (pl.scan_parquet(files)
            .select(PERMNO=pl.col("PERMNO").cast(pl.Int64),
                    DATE=parse_yyyymmdd("DATE"),
                    PRC=pl.col("PRC"),
                    SHROUT=pl.col("SHROUT"),
                    CFACSHR=pl.col("CFACSHR"))
            .filter(pl.col("DATE")==pl.lit(at_date)))

def compute_ratios_for_anchor(asof_day: date) -> pl.DataFrame:
    y = asof_day.year
    snap_t_lf = crsp_snapshot_lazy(asof_day)
    prev_day  = prior_trading_day_approx_1y(asof_day)
    snap_prev_lf = crsp_snapshot_lazy(prev_day)

    iss_lf = (snap_t_lf.select("PERMNO","CFACSHR","SHROUT")
              .join(snap_prev_lf.select(pl.col("PERMNO"),
                                        pl.col("CFACSHR").alias("CFACSHR_prev"),
                                        pl.col("SHROUT").alias("SHROUT_prev")),
                    on="PERMNO", how="left")
              .with_columns(ns_12m=pl.when(
                    pl.all_horizontal(pl.col("CFACSHR_prev").is_not_null(),
                                      pl.col("SHROUT_prev").is_not_null(),
                                      pl.col("CFACSHR").is_not_null(),
                                      pl.col("SHROUT").is_not_null()))
                    .then((pl.col("SHROUT")*pl.col("CFACSHR")).log() -
                          (pl.col("SHROUT_prev")*pl.col("CFACSHR_prev")).log())
                    .otherwise(None))
              .select("PERMNO","ns_12m"))

    fpaths = year_paths(FUND_DS, [y-2,y-1,y])
    lf0 = pl.scan_parquet(fpaths, low_memory=True)
    present = set(lf0.collect_schema().names())

    FUND_COLS = ["GVKEY","LPERMNO","DATADATE","RDQ","SALEQ","COGSQ","OIBDPQ","XSGAQ","XINTQ",
                 "NIQ","DPQ","ACTQ","CHEQ","LCTQ","DLCQ","TXPQ","ATQ","IVSTQ","DLTTQ","MIBQ",
                 "LTQ","SEQQ","TXDITCQ","PSTKRQ","PSTKRV","PSTKQ","LINKDT","LINKENDDT"]
    use_cols = [c for c in FUND_COLS if c in present]
    pstk_order = [c for c in ["PSTKRQ","PSTKRV","PSTKQ"] if c in present]
    pstk_expr = pl.coalesce([pl.col(c) for c in pstk_order]) if pstk_order else pl.lit(0.0)

    base = (lf0.select(use_cols)
            .with_columns(
                DATADATE_dt=parse_yyyymmdd("DATADATE") if "DATADATE" in present else pl.lit(None).cast(pl.Date),
                RDQ_dt=parse_yyyymmdd("RDQ") if "RDQ" in present else pl.lit(None).cast(pl.Date),
                GVKEY=pl.col("GVKEY").cast(pl.Utf8) if "GVKEY" in present else pl.lit(None),
                LPERMNO=pl.col("LPERMNO").cast(pl.Int64) if "LPERMNO" in present else pl.lit(None),
            )
            .filter(pl.when(pl.col("RDQ_dt").is_not_null())
                        .then(pl.col("RDQ_dt")<=pl.lit(asof_day))
                        .otherwise((pl.col("DATADATE_dt")+pl.duration(days=90))<=pl.lit(asof_day)))
            .pipe(lambda lf: lf.with_columns(
                    LINKDT_dt=parse_yyyymmdd("LINKDT"),
                    LINKENDDT_dt=parse_yyyymmdd("LINKENDDT"),
                 ).filter(
                    (pl.col("LINKDT_dt").is_null() | (pl.col("LINKDT_dt")<=pl.lit(asof_day))) &
                    (pl.col("LINKENDDT_dt").is_null() | (pl.col("LINKENDDT_dt")>=pl.lit(asof_day)))
                 ) if {"LINKDT","LINKENDDT"}.issubset(present) else lf)
            .sort(["GVKEY","RDQ_dt","DATADATE_dt"])
            .with_columns(((pl.col("ACTQ")-pl.col("CHEQ")) - (pl.col("LCTQ")-pl.col("DLCQ")-pl.col("TXPQ"))).alias("WC"))
            .with_columns([
                pl.col("NIQ").rolling_sum(4).over("GVKEY").alias("TTM_NIQ"),
                pl.col("DPQ").rolling_sum(4).over("GVKEY").alias("TTM_DPQ"),
                pl.col("OIBDPQ").rolling_sum(4).over("GVKEY").alias("TTM_OIBDP"),
                pl.col("ATQ").rolling_mean(4).over("GVKEY").alias("AVG4_ATQ"),
                (pl.col("WC")-pl.col("WC").shift(4).over("GVKEY")).alias("WC_D4"),
                pl.col("ATQ").shift(4).over("GVKEY").alias("ATQ_lag4"),
            ])
            .with_columns(
                CFO_TTM = pl.col("TTM_NIQ")+pl.col("TTM_DPQ")-pl.col("WC_D4"),
                BE      = pl.col("SEQQ")+pl.col("TXDITCQ")-pstk_expr,
                GP_num  = pl.coalesce([pl.col("SALEQ")-pl.col("COGSQ"), pl.col("OIBDPQ")+pl.col("XSGAQ")]),
                OP_num  = pl.coalesce([pl.col("SALEQ")-pl.col("COGSQ")-pl.col("XSGAQ")-pl.col("XINTQ"),
                                       pl.col("OIBDPQ")-pl.col("XINTQ")]),
                NOA_full = (pl.col("ATQ")-pl.col("CHEQ")-pl.col("IVSTQ"))
                           - (pl.col("LTQ")-pl.col("DLCQ")-pl.col("DLTTQ")),
                ACC_num = ((pl.col("ACTQ")-pl.col("ACTQ").shift(1).over("GVKEY"))
                           - (pl.col("CHEQ")-pl.col("CHEQ").shift(1).over("GVKEY"))
                           - (pl.col("LCTQ")-pl.col("LCTQ").shift(1).over("GVKEY"))
                           + (pl.col("DLCQ")-pl.col("DLCQ").shift(1).over("GVKEY"))
                           + (pl.col("TXPQ")-pl.col("TXPQ").shift(1).over("GVKEY"))
                           - pl.col("DPQ")),
            )
            .group_by("GVKEY").tail(1)
            .filter(pl.col("LPERMNO").is_not_null()))

    price_lf = snap_t_lf.select(PERMNO=pl.col("PERMNO").cast(pl.Int64),
                                PRC=pl.col("PRC"), SHROUT=pl.col("SHROUT"))

    latest_lf = (base.join(price_lf, left_on="LPERMNO", right_on="PERMNO", how="left")
                 .join(iss_lf, left_on="LPERMNO", right_on="PERMNO", how="left")
                 .with_columns(MC_mn=(pl.col("PRC").abs()*pl.col("SHROUT")/1000.0))
                 .with_columns(
                     EV_A_raw = pl.col("MC_mn")+pl.col("DLTTQ")+pl.col("DLCQ")+pstk_expr
                                +pl.col("MIBQ")-pl.col("CHEQ")-pl.col("IVSTQ"),
                     EV_B_raw = pl.col("MC_mn")+pl.coalesce([pl.col("DLTTQ"),pl.lit(0.0)]) \
                                +pl.coalesce([pl.col("DLCQ"),pl.lit(0.0)]) \
                                -pl.coalesce([pl.col("CHEQ"),pl.lit(0.0)]),
                     EV_C_raw = pl.col("MC_mn")+pl.coalesce([pl.col("LTQ"),pl.lit(0.0)]) \
                                -(pl.coalesce([pl.col("LCTQ"),pl.lit(0.0)])-pl.coalesce([pl.col("DLCQ"),pl.lit(0.0)])) \
                                -pl.coalesce([pl.col("CHEQ"),pl.lit(0.0)]),
                     EV_D_raw = pl.col("MC_mn"),
                 )
                 .with_columns(
                     goodA = pl.col("EV_A_raw").is_finite() & (pl.col("EV_A_raw")>0),
                     goodB = pl.col("EV_B_raw").is_finite() & (pl.col("EV_B_raw")>0),
                     goodC = pl.col("EV_C_raw").is_finite() & (pl.col("EV_C_raw")>0),
                     goodD = pl.col("EV_D_raw").is_finite() & (pl.col("EV_D_raw")>0),
                 )
                 .with_columns(
                     EV_FLEX = (pl.when(pl.col("goodA")).then(pl.col("EV_A_raw"))
                                  .when(pl.col("goodB")).then(pl.col("EV_B_raw"))
                                  .when(pl.col("goodC")).then(pl.col("EV_C_raw"))
                                  .when(pl.col("goodD")).then(pl.col("EV_D_raw"))
                                  .otherwise(None)),
                     EV_src  = (pl.when(pl.col("goodA")).then(pl.lit("A"))
                                  .when(pl.col("goodB")).then(pl.lit("B"))
                                  .when(pl.col("goodC")).then(pl.lit("C"))
                                  .when(pl.col("goodD")).then(pl.lit("D"))
                                  .otherwise(pl.lit("NA"))),
                 )
                 .with_columns(
                     WC_B=(pl.col("ACTQ")-pl.col("CHEQ"))-(pl.col("LCTQ")-pl.col("DLCQ")),
                     WC_C=(pl.col("ACTQ")-pl.col("CHEQ"))-pl.col("LCTQ"),
                 )
                 .with_columns(
                     WC_B_D4=pl.col("WC_B")-pl.col("WC_B").shift(4).over("GVKEY"),
                     WC_C_D4=pl.col("WC_C")-pl.col("WC_C").shift(4).over("GVKEY"),
                 )
                 .with_columns(
                     CFO_TTM_B=pl.col("TTM_NIQ")+pl.col("TTM_DPQ")-pl.col("WC_B_D4"),
                     CFO_TTM_C=pl.col("TTM_NIQ")+pl.col("TTM_DPQ")-pl.col("WC_C_D4"),
                     CFO_TTM_D=pl.col("TTM_NIQ")+pl.col("TTM_DPQ"),
                 )
                 .with_columns(
                     has_A=pl.col("CFO_TTM").is_finite() & pl.col("CFO_TTM").is_not_null(),
                     has_B=pl.col("CFO_TTM_B").is_finite() & pl.col("CFO_TTM_B").is_not_null(),
                     has_C=pl.col("CFO_TTM_C").is_finite() & pl.col("CFO_TTM_C").is_not_null(),
                     has_D=pl.col("CFO_TTM_D").is_finite() & pl.col("CFO_TTM_D").is_not_null(),
                 )
                 .with_columns(
                     CFO_TTM_FLEX=(pl.when(pl.col("has_A")).then(pl.col("CFO_TTM"))
                                     .when(pl.col("has_B")).then(pl.col("CFO_TTM_B"))
                                     .when(pl.col("has_C")).then(pl.col("CFO_TTM_C"))
                                     .when(pl.col("has_D")).then(pl.col("CFO_TTM_D"))
                                     .otherwise(None)),
                     CFO_src=(pl.when(pl.col("has_A")).then(pl.lit("A"))
                                .when(pl.col("has_B")).then(pl.lit("B"))
                                .when(pl.col("has_C")).then(pl.lit("C"))
                                .when(pl.col("has_D")).then(pl.lit("D"))
                                .otherwise(pl.lit("NA"))),
                 )
                 .with_columns(
                     GP_AT       = pos_div(pl.col("GP_num"),  pl.col("ATQ")),
                     OP_BE       = pos_div(pl.col("OP_num"),  pl.col("BE")),
                     ROA_TTM     = pos_div(pl.col("TTM_NIQ"), pl.col("AVG4_ATQ")),
                     CFO_AT      = pos_div(pl.col("CFO_TTM"), pl.col("ATQ")),
                     SloanAcc_AT = safe_div(pl.col("ACC_num"), pl.col("AVG4_ATQ")),
                     AssetGrowth = safe_div(pl.col("ATQ")-pl.col("ATQ_lag4"), pl.col("ATQ_lag4")),
                     NOA_AT      = pos_div(pl.col("NOA_full"), pl.col("ATQ")),
                     NetShareIss = pl.col("ns_12m"),
                     # ---- Valuation flipped to price/multiples ----
                     MB          = pos_div(pl.col("MC_mn"),  pl.col("BE")),        # Market-to-Book
                     PE          = pos_div(pl.col("MC_mn"),  pl.col("TTM_NIQ")),   # Price/Earnings
                     PC          = pos_div(pl.col("MC_mn"),  pl.col("CFO_TTM")),   # Price/Cash Flow
                 )
                 .with_columns(  # build NOA_* and quality flags
                     SloanAcc_AT_FLEX = safe_div(pl.col("TTM_NIQ")-pl.col("CFO_TTM_FLEX"), pl.col("AVG4_ATQ")),
                     NOA_B = pl.col("ATQ")-pl.col("CHEQ")-(pl.coalesce([pl.col("DLTTQ"),pl.lit(0.0)])+pl.coalesce([pl.col("DLCQ"),pl.lit(0.0)])),
                     NOA_C = pl.col("ATQ")-pl.col("CHEQ")-pl.coalesce([pl.col("LTQ"),pl.lit(0.0)]),
                     NOA_D = (pl.col("ACTQ")-pl.col("CHEQ"))-(pl.col("LCTQ")-pl.col("DLCQ")),
                 )
                 .with_columns(
                     goodN_A = pl.col("NOA_full").is_finite() & pl.col("NOA_full").is_not_null(),
                     goodN_B = pl.col("NOA_B").is_finite()    & pl.col("NOA_B").is_not_null(),
                     goodN_C = pl.col("NOA_C").is_finite()    & pl.col("NOA_C").is_not_null(),
                     goodN_D = pl.col("NOA_D").is_finite()    & pl.col("NOA_D").is_not_null(),
                 )
                 # >>> two-pass fix: make NOA_FLEX first, then use it
                 .with_columns(
                     NOA_FLEX = (pl.when(pl.col("goodN_A")).then(pl.col("NOA_full"))
                                   .when(pl.col("goodN_B")).then(pl.col("NOA_B"))
                                   .when(pl.col("goodN_C")).then(pl.col("NOA_C"))
                                   .when(pl.col("goodN_D")).then(pl.col("NOA_D"))
                                   .otherwise(None)),
                     NOA_src  = (pl.when(pl.col("goodN_A")).then(pl.lit("A"))
                                   .when(pl.col("goodN_B")).then(pl.lit("B"))
                                   .when(pl.col("goodN_C")).then(pl.lit("C"))
                                   .when(pl.col("goodN_D")).then(pl.lit("D"))
                                   .otherwise(pl.lit("NA"))),
                 )
                 .with_columns(  # now allowed to reference NOA_FLEX
                     NOA_AT_FLEX = pos_div(pl.col("NOA_FLEX"), pl.col("ATQ")),
                     CFO_AT_FLEX = pos_div(pl.col("CFO_TTM_FLEX"), pl.col("ATQ")),
                     PC_FLEX     = pos_div(pl.col("MC_mn"), pl.col("CFO_TTM_FLEX")),  # Price/Cash Flow (flex)
                     # ---- EV/EBITDA (STRICT uses EV_A when valid) ----
                     EV_EBITDA_STRICT = pos_div(
                         pl.when(pl.col("goodA")).then(pl.col("EV_A_raw")).otherwise(None),
                         pl.col("TTM_OIBDP")),
                     EV_EBITDA_FLEX   = pos_div(pl.col("EV_FLEX"), pl.col("TTM_OIBDP")),
                 )
                 .with_columns(core_non_null = pl.sum_horizontal([
                     pl.col("GP_AT").is_not_null(), pl.col("MB").is_not_null(), pl.col("PE").is_not_null(),
                     pl.col("ROA_TTM").is_not_null(), pl.col("AssetGrowth").is_not_null(),
                     pl.col("NetShareIss").is_not_null(), pl.col("OP_BE").is_not_null(),
                 ]))
                 .select(
                     pl.lit(asof_day).alias("asof_date"),
                     pl.col("LPERMNO").alias("permno"),
                     pl.col("GVKEY").alias("gvkey"),
                     "GP_AT","OP_BE","ROA_TTM","CFO_AT","SloanAcc_AT","AssetGrowth","NOA_AT","NetShareIss",
                     # ---- flipped valuation outputs ----
                     "MB","PE","PC",
                     "EV_EBITDA_STRICT",
                     "CFO_AT_FLEX","PC_FLEX","SloanAcc_AT_FLEX","NOA_AT_FLEX","EV_EBITDA_FLEX",
                     pl.col("EV_EBITDA_FLEX").alias("EV_EBITDA"),
                     "EV_src","CFO_src","NOA_src","MC_mn","EV_FLEX","core_non_null",
                 ))
    return latest_lf.collect(engine="streaming")

def main():
    for y in YEARS:
        outy = OUTDIR / f"year={y}"
        outy.mkdir(parents=True, exist_ok=True)
        for (m,d) in ANCHORS:
            anchor = date(y,m,d)
            asof = next_trading_day(anchor)
            if asof is None:
                continue
            df = compute_ratios_for_anchor(asof)
            if df.is_empty():
                continue
            out_path = outy / f"ratios_{asof.strftime('%Y%m%d')}.parquet"
            df.write_parquet(str(out_path))
            print(f"[OK] {asof} -> {out_path} ({df.height:,} rows)")

if __name__ == "__main__":
    main()
