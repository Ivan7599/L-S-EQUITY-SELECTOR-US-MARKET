# _0007_AllStrategiesRollingAlpha_QuarterPeriods.py
from __future__ import annotations
from pathlib import Path
import io, re
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ----------------- Paths & config -----------------
RAW_DIR = Path("./RawDatabase")
FF5_CSV = RAW_DIR / "Factors5.csv"
MOM_CSV = RAW_DIR / "Momentum.csv"

STRATEGY_PATHS = {
    "Base":     Path("./FilteredRawData/BaseLS_Returns.parquet"),
    "LightGBM": Path("./FilteredRawData/LightGBMLS_Returns.parquet"),
    "Ridge":    Path("./FilteredRawData/RidgeLS_Returns.parquet"),
}

ROLL_WIN_DAYS = 252 * 5          # 5-year rolling window
OUT_IMG = Path("./RollingAlpha_3Strategies.jpg")
OUT_ALPHA_TABLE = Path("./AlphaSubperiods.csv")


# ----------------- Helpers to read FF data -----------------
def _read_ff_like(path: Path, header: str) -> pl.DataFrame:
    patt = re.compile(r"^\s*\d{8},")  # keep only YYYYMMDD lines
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if patt.match(line):
                rows.append(line.rstrip("\n"))
    if not rows:
        raise RuntimeError(f"No data lines detected in {path}")
    csv_text = header + "\n" + "\n".join(rows)
    dtypes = {c: pl.Utf8 for c in header.split(",")}
    return pl.read_csv(io.StringIO(csv_text), dtypes=dtypes)


def read_ff5(path: Path) -> pl.DataFrame:
    hdr = "DATE,Mkt_RF,SMB,HML,RMW,CMA,RF"
    df = _read_ff_like(path, hdr)
    num_cols = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "RF"]
    exprs = [
        pl.col("DATE")
        .str.strip_chars()
        .str.strptime(pl.Date, "%Y%m%d", strict=False)
        .alias("DATE")
    ]
    exprs += [
        (pl.col(c).str.strip_chars().cast(pl.Float64) / 100.0).alias(c)
        for c in num_cols
    ]
    return df.with_columns(exprs).drop_nulls(subset=["DATE"])


def read_mom(path: Path) -> pl.DataFrame:
    hdr = "DATE,Mom"
    df = _read_ff_like(path, hdr)
    return (
        df.with_columns(
            [
                pl.col("DATE")
                .str.strip_chars()
                .str.strptime(pl.Date, "%Y%m%d", strict=False)
                .alias("DATE"),
                (pl.col("Mom").str.strip_chars().cast(pl.Float64) / 100.0).alias("Mom"),
            ]
        )
        .drop_nulls(subset=["DATE"])
    )


# ----------------- Strategy returns & OLS helpers -----------------
def read_strategy_returns(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(str(path))
    if "DATE" not in df.columns:
        raise RuntimeError(f"{path.name} must contain a DATE column.")
    if "ls_ret" in df.columns:
        out = df.select(["DATE", "ls_ret"])
    else:
        # Derive LS returns from a NAV/wealth column if needed
        num_cols = [
            c for c, dt in df.schema.items() if c != "DATE" and pl.datatypes.is_numeric(dt)
        ]
        if not num_cols:
            raise RuntimeError(f"No numeric columns to infer returns in {path.name}.")
        level = num_cols[0]
        out = (
            df.sort("DATE")
            .with_columns((pl.col(level) / pl.col(level).shift(1) - 1.0).alias("ls_ret"))
            .select(["DATE", "ls_ret"])
        )
    if out.schema["DATE"] != pl.Date:
        out = out.with_columns(pl.col("DATE").cast(pl.Date, strict=False))
    return out.drop_nulls(subset=["DATE", "ls_ret"])


def ols_with_tstats(y: np.ndarray, X: np.ndarray):
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    n, k = X.shape
    resid = y - X @ beta
    s2 = (resid @ resid) / (n - k)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(s2 * XtX_inv))
    tstats = beta / se
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - (resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if n > k else float("nan")
    return beta, tstats, r2, adj_r2


def rolling_alpha_series(
    y: np.ndarray, factors_mat: np.ndarray, dates_list, window: int
):
    roll_alpha = []
    roll_dates = []
    if len(y) < window:
        return roll_dates, roll_alpha

    for end_idx in range(window - 1, len(y)):
        y_win = y[end_idx - window + 1 : end_idx + 1]
        X_win = np.column_stack(
            [
                np.ones(window),
                factors_mat[end_idx - window + 1 : end_idx + 1, :],
            ]
        )
        beta_win, _, _, _ = ols_with_tstats(y_win, X_win)
        alpha_daily = beta_win[0]
        alpha_ann = alpha_daily * 252.0
        roll_alpha.append(alpha_ann)
        roll_dates.append(dates_list[end_idx])

    return roll_dates, roll_alpha


# ----------------- Rolling alpha + subperiod alphas -----------------
def compute_rolling_alpha_for_strategy(
    ls_path: Path, ff: pl.DataFrame, window: int
):
    strat = read_strategy_returns(ls_path)  # DATE, ls_ret
    data = strat.join(ff, on="DATE", how="inner").sort("DATE")
    if data.height < window:
        raise RuntimeError(f"Not enough data for 5y rolling alpha: {ls_path.name}")

    y = data["ls_ret"].to_numpy()
    factors_mat = np.column_stack(
        [
            data["Mkt_RF"].to_numpy(),
            data["SMB"].to_numpy(),
            data["HML"].to_numpy(),
            data["RMW"].to_numpy(),
            data["CMA"].to_numpy(),
            data["Mom"].to_numpy(),
        ]
    )
    dates_list = data["DATE"].to_list()
    return rolling_alpha_series(y, factors_mat, dates_list, window)


def compute_subperiod_alphas(
    model_name: str, ls_path: Path, ff: pl.DataFrame
) -> list[dict]:
    """
    Split the full sample into 4 equal subperiods (by number of days),
    and compute annualised alpha (and t-stat) for each model in each subperiod.
    """
    strat = read_strategy_returns(ls_path)
    data = strat.join(ff, on="DATE", how="inner").sort("DATE")

    y = data["ls_ret"].to_numpy()
    factors_mat = np.column_stack(
        [
            data["Mkt_RF"].to_numpy(),
            data["SMB"].to_numpy(),
            data["HML"].to_numpy(),
            data["RMW"].to_numpy(),
            data["CMA"].to_numpy(),
            data["Mom"].to_numpy(),
        ]
    )
    dates = data["DATE"].to_list()
    n = len(y)
    if n < 40:
        raise RuntimeError(f"Sample too short for subperiod splits: {ls_path.name}")

    # indices for 4 quartiles
    idx = [0, n // 4, n // 2, 3 * n // 4, n]

    records: list[dict] = []
    for p in range(4):
        start_i, end_i = idx[p], idx[p + 1]
        if end_i - start_i < 30:  # safety
            continue

        y_win = y[start_i:end_i]
        X_win = np.column_stack(
            [np.ones(end_i - start_i), factors_mat[start_i:end_i, :]]
        )
        beta, tstats, _, _ = ols_with_tstats(y_win, X_win)
        alpha_daily = beta[0]
        alpha_ann = alpha_daily * 252.0
        t_alpha = tstats[0]

        records.append(
            {
                "Model": model_name,
                "Period": p + 1,
                "StartDate": dates[start_i],
                "EndDate": dates[end_i - 1],
                "AnnAlpha": alpha_ann,
                "tstatAlpha": t_alpha,
                "NObs": end_i - start_i,
            }
        )
    return records


# ----------------- Plotting -----------------
def plot_multi_rolling_alpha(df: pl.DataFrame, out_path: Path):
    dates = df["DATE"].to_list()
    alpha_cols = [c for c in df.columns if c != "DATE"]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for col in alpha_cols:
        ax.plot(dates, df[col].to_numpy(), linewidth=1.4, label=col)

    ax.set_title("5Y Rolling Alpha (FF5 + Mom) — Base vs LightGBM vs Ridge")
    ax.set_ylabel("Annualized alpha")
    ax.grid(True, alpha=0.3)

    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.legend()
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


# ----------------- Main -----------------
def main():
    # Fama–French + momentum
    ff5 = read_ff5(FF5_CSV)
    mom = read_mom(MOM_CSV)
    ff = ff5.join(mom, on="DATE", how="inner").select(
        ["DATE", "Mkt_RF", "SMB", "HML", "RMW", "CMA", "RF", "Mom"]
    )

    # 5Y rolling alphas
    roll_dfs = []
    subperiod_records: list[dict] = []

    for name, parq in STRATEGY_PATHS.items():
        # Rolling alpha series
        roll_dates, roll_alpha = compute_rolling_alpha_for_strategy(
            parq, ff, ROLL_WIN_DAYS
        )
        roll_df = pl.DataFrame({"DATE": roll_dates, name: roll_alpha})
        roll_dfs.append(roll_df)

        # Subperiod alphas
        subperiod_records.extend(compute_subperiod_alphas(name, parq, ff))

    # Align rolling alphas across models (inner join on DATE)
    merged = roll_dfs[0]
    for df in roll_dfs[1:]:
        merged = merged.join(df, on="DATE", how="inner")

    if merged.height == 0:
        raise RuntimeError("No overlapping 5Y rolling alpha dates across strategies.")

    plot_multi_rolling_alpha(merged, OUT_IMG)
    print(f"Saved combined rolling alpha plot to {OUT_IMG}")

    # Build subperiod alpha table
    alpha_table = pl.DataFrame(subperiod_records).sort(["Period", "Model"])
    print(alpha_table)
    alpha_table.write_csv(OUT_ALPHA_TABLE)
    print(f"Saved subperiod alpha table to {OUT_ALPHA_TABLE}")


if __name__ == "__main__":
    main()

