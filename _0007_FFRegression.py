# _0007_FFRegression_AllModels.py
# OLS of daily strategy vs FF5+Mom (robust parsing, alignment)
# Runs for multiple models (Base, LightGBM, Ridge) in one pass.

from __future__ import annotations
from pathlib import Path
import io, re
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

RAW_DIR = Path("./RawDatabase")
FF5_CSV = RAW_DIR / "Factors5.csv"
MOM_CSV = RAW_DIR / "Momentum.csv"

# -------- Model specs: LS input + outputs per model --------
MODEL_SPECS = [
    {
        "name": "Base",
        "ls_parq": Path("./FilteredRawData/BaseLS_Returns.parquet"),
        "out_csv": Path("./BaseOLSFama.csv"),
        "out_roll_img": Path("./BaseRollingAlpha.jpg"),
    },
    {
        "name": "LightGBM",
        "ls_parq": Path("./FilteredRawData/LightGBMLS_Returns.parquet"),
        "out_csv": Path("./LightGBMOLSFama.csv"),
        "out_roll_img": Path("./LightGBMRollingAlpha.jpg"),
    },
    {
        "name": "Ridge",
        "ls_parq": Path("./FilteredRawData/RidgeLS_Returns.parquet"),
        "out_csv": Path("./RidgeOLSFama.csv"),
        "out_roll_img": Path("./RidgeRollingAlpha.jpg"),
    },
]

ROLL_WIN_DAYS = 252 * 5  # ~5y of daily trading data

# -------------------- FF5 & momentum readers --------------------


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
    # Ken French daily 5 factors: DATE,Mkt_RF,SMB,HML,RMW,CMA,RF (percent units)
    hdr = "DATE,Mkt_RF,SMB,HML,RMW,CMA,RF"
    df = _read_ff_like(path, hdr)
    num_cols = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "RF"]
    exprs = []
    exprs.append(
        pl.col("DATE")
        .str.strip_chars()
        .str.strptime(pl.Date, "%Y%m%d", strict=False)
        .alias("DATE")
    )
    for c in num_cols:
        exprs.append(
            (pl.col(c).str.strip_chars().cast(pl.Float64) / 100.0).alias(c)
        )
    return df.with_columns(exprs).drop_nulls(subset=["DATE"])


def read_mom(path: Path) -> pl.DataFrame:
    # DATE,Mom (percent units)
    hdr = "DATE,Mom"
    df = _read_ff_like(path, hdr)
    return (
        df.with_columns(
            [
                pl.col("DATE")
                .str.strip_chars()
                .str.strptime(pl.Date, "%Y%m%d", strict=False)
                .alias("DATE"),
                (pl.col("Mom").str.strip_chars().cast(pl.Float64) / 100.0).alias(
                    "Mom"
                ),
            ]
        )
        .drop_nulls(subset=["DATE"])
    )


# -------------------- Strategy returns reader --------------------


def read_strategy_returns(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(str(path))
    if "DATE" not in df.columns:
        raise RuntimeError(f"{path} must contain a DATE column.")
    if "ls_ret" in df.columns:
        out = df.select(["DATE", "ls_ret"])
    else:
        num_cols = [
            c
            for c, dt in df.schema.items()
            if c != "DATE" and pl.datatypes.is_numeric(dt)
        ]
        if not num_cols:
            raise RuntimeError(
                f"No numeric columns found to infer returns from in {path}."
            )
        level = num_cols[0]
        out = (
            df.sort("DATE")
            .with_columns(
                (pl.col(level) / pl.col(level).shift(1) - 1.0).alias("ls_ret")
            )
            .select(["DATE", "ls_ret"])
        )
    if out.schema["DATE"] != pl.Date:
        out = out.with_columns(pl.col("DATE").cast(pl.Date, strict=False))
    return out.drop_nulls(subset=["ls_ret", "DATE"])


# -------------------- OLS & rolling alpha helpers --------------------


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
    """
    For each trailing `window` days, run OLS:
    y_t = alpha + b*Mkt_RF + ... + Mom
    Store annualized alpha (daily alpha * 252).
    Returns (roll_dates, roll_alpha_ann).
    """
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
        alpha_ann = alpha_daily * 252.0  # annualize
        roll_alpha.append(alpha_ann)
        roll_dates.append(dates_list[end_idx])

    return roll_dates, roll_alpha


def plot_rolling_alpha(dates, alpha_vals, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dates, alpha_vals, color="#8B0000", linewidth=1.6, label="5Y Rolling Alpha")
    ax.set_title(title)
    ax.set_ylabel("Annualized alpha")
    ax.grid(True, alpha=0.3)
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


# -------------------- Per-model runner --------------------


def run_for_model(
    name: str,
    ls_parq: Path,
    ff: pl.DataFrame,
    out_csv: Path,
    out_roll_img: Path,
) -> None:
    print(f"\n=== Running FF5+Mom OLS for model: {name} ===")

    strat = read_strategy_returns(ls_parq)  # DATE, ls_ret (decimal)
    data = strat.join(ff, on="DATE", how="inner").sort("DATE")
    if data.height < 30:
        print(f"[{name}] Too few overlapping trading days to run OLS.")
        return

    # --- Full-sample OLS ---
    y = data["ls_ret"].to_numpy()
    X = np.column_stack(
        [
            np.ones(len(y)),
            data["Mkt_RF"].to_numpy(),
            data["SMB"].to_numpy(),
            data["HML"].to_numpy(),
            data["RMW"].to_numpy(),
            data["CMA"].to_numpy(),
            data["Mom"].to_numpy(),
        ]
    )

    beta, tstats, r2, adj_r2 = ols_with_tstats(y, X)

    out = pl.DataFrame(
        {
            "model": [name],
            "sample_start": [data["DATE"].min()],
            "sample_end": [data["DATE"].max()],
            "N": [len(y)],
            "alpha": [beta[0]],
            "t_alpha": [tstats[0]],
            "beta_mkt": [beta[1]],
            "t_mkt": [tstats[1]],
            "beta_smb": [beta[2]],
            "t_smb": [tstats[2]],
            "beta_hml": [beta[3]],
            "t_hml": [tstats[3]],
            "beta_rmw": [beta[4]],
            "t_rmw": [tstats[4]],
            "beta_cma": [beta[5]],
            "t_cma": [tstats[5]],
            "beta_mom": [beta[6]],
            "t_mom": [tstats[6]],
            "R2": [r2],
            "Adj_R2": [adj_r2],
        }
    )
    out.write_csv(str(out_csv))
    print(f"[{name}] Saved OLS results to {out_csv}")

    # --- 5Y rolling alpha series ---
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
    roll_dates, roll_alpha = rolling_alpha_series(
        y, factors_mat, dates_list, ROLL_WIN_DAYS
    )

    if len(roll_dates) > 0:
        plot_rolling_alpha(
            roll_dates,
            roll_alpha,
            out_roll_img,
            title=f"{name} 5Y Rolling Alpha (FF5+Mom)",
        )
        print(f"[{name}] Saved rolling alpha plot to {out_roll_img}")
    else:
        print(f"[{name}] Not enough data for 5Y rolling alpha window.")


# -------------------- main --------------------


def main():
    ff5 = read_ff5(FF5_CSV)
    mom = read_mom(MOM_CSV)
    ff = ff5.join(mom, on="DATE", how="inner").select(
        ["DATE", "Mkt_RF", "SMB", "HML", "RMW", "CMA", "RF", "Mom"]
    )

    for spec in MODEL_SPECS:
        run_for_model(
            name=spec["name"],
            ls_parq=spec["ls_parq"],
            ff=ff,
            out_csv=spec["out_csv"],
            out_roll_img=spec["out_roll_img"],
        )


if __name__ == "__main__":
    main()
