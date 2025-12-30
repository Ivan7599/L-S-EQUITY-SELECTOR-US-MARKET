# L-S-EQUITY-SELECTOR-US-MARKET

Systematic **US equity long/short stock selection** with an explicit research question:

> **Is machine learning stronger than economical logic** when both use the same fundamental information set under strict anti-overfitting constraints?

This repository implements an end-to-end pipeline using:

* **CRSP daily** prices/returns (US common stocks; **financials excluded**)
* **Compustat quarterly** fundamentals (US firms; **financials excluded**)
* **Fama–French 5 factors (daily)** + **Momentum (daily)** for alpha attribution

We build **profitability / valuation / quality** characteristics, produce **five investable rankings** (Base + 4 ML variants), form **top/bottom decile long–short portfolios** at quarterly investment dates, and evaluate:

* **L/S and leg performance** (Sharpe, Sortino, drawdown, etc.)
* **FF5+Mom alphas** (full-sample + rolling)
* **VAMI charts** (single-model and multi-model comparisons)
* **Hit rates** (L/S, Long, Short)

> ⚠ **Important:** This project requires a valid **WRDS license**.
> CRSP / Compustat data are proprietary and **cannot be included** in this repository.

---

## 1) Data requirements

You must source raw data from **WRDS**:

1. Download from WRDS:

* **CRSP** → US Stock → Daily Stock File
* **Compustat North America** → Fundamentals Quarterly

2. Recommended universe filters:

* US **common stocks only**
* **Exclude financial institutions** (SIC/GICS or WRDS filter)
* Date range (e.g. 1980–today)

3. Export as **SAS** files named:

* `RawReturns.sas7bdat` (CRSP daily)
* `RawFundamentals.sas7bdat` (Compustat quarterly)

4. Create this folder in the project root:

```text
RawDatabase/
    RawReturns.sas7bdat
    RawFundamentals.sas7bdat
    Factors5.csv      # Fama–French 5 factors (daily)
    Momentum.csv      # Momentum factor (daily)
```

`Factors5.csv` and `Momentum.csv` should be the standard **Kenneth French daily factor files** saved as CSV.

---

## 2) Repository structure

After running the pipeline, the project will contain (high-level):

```text
FilteredRawData/
    Returns_ds/               # CRSP daily returns, Parquet, partitioned by year=YYYY
    Fundamentals_q/           # Compustat quarterly, Parquet, partitioned by year=YYYY
    [Model]LS_Returns.parquet
    [Model]Long_Returns.parquet
    [Model]Short_Returns.parquet

Ratios/
    year=YYYY/ratios_YYYYMMDD.parquet

BaseAnchorRanking/
    year=YYYY/anchor_YYYYMMDD.parquet

BaseInvestmentRanking/
LightGBMInvestmentRanking/
LightGBMScoreRanking/
RidgeRanking/
RidgeScoreRanking/
    year=YYYY/*rank_YYYYMMDD.parquet

BaseTopBottom10/
LightGBMTopBottom10/
LightGBMScoreTopBottom10/
RidgeTopBottom10/
RidgeScoreTopBottom10/
    year=YYYY/topbottom10_YYYYMMDD.parquet
```

---

## 3) Models and feature sets

All models ultimately produce a **cross-sectional ranking at each investment date**. Portfolios are then formed by:

* **Long**: top **10%**
* **Short**: bottom **10%**
* Signals: `signal = +1` (long), `signal = -1` (short)
* Always **1 USD long / 1 USD short** (dollar-neutral by construction)

### 3.1 Base model (economical logic)

A transparent baseline grounded in economic intuition:

* Build a **composite score** from profitability, valuation, and quality pillars at each anchor date.
* Convert the score into a persistent signal using **3-year dynamics**:

  * **Average (level)**
  * **Slope (trend)**
  * **Stability** (volatility / variability of score changes)
* Combine these into a single investment score and rank stocks cross-sectionally.

### 3.2 Machine-learning models (two feature configurations)

Each ML family is evaluated under two feature sets:

**A) Raw ratios as features**

* Uses the cross-sectional **raw fundamental ratios** directly as predictors.

**B) “Score features” (same information content as Base dynamics)**

* Uses **only engineered features derived from the composite score** over the last 3 years:

  * slope / average / stdev (trend, level, stability)

Models:

* **LightGBMInvestmentRanking** → LightGBM with **raw ratios**
* **LightGBMScoreRanking** → LightGBM with **score features**
* **RidgeInvestmentRanking** → Ridge regression with **raw ratios**
* **RidgeScoreRanking** → Ridge regression with **score features**

---

## 4) Pipeline (run in order)

### Step 1 — WRDS SAS → Parquet (year-partitioned)

**`_0001_RawDataProcessing.py`**

* Reads `RawReturns.sas7bdat` and `RawFundamentals.sas7bdat`
* Cleans identifiers and dates
* Excludes financials
* Writes year-partitioned Parquet datasets:

```text
FilteredRawData/Returns_ds/year=YYYY/*.parquet
FilteredRawData/Fundamentals_q/year=YYYY/*.parquet
```

---

### Step 2 — Ratios computation (profitability / valuation / quality)

**`_0002_RatiosCalculation.py`**

* Aligns quarterly fundamentals to investment “as-of” dates (quarterly anchors)
* Computes a set of **fundamental ratios** spanning:

  * Profitability
  * Valuation
  * Quality
* Cleans / winsorizes / standardizes directions (“higher is better”)
* Writes one snapshot per anchor:

```text
Ratios/year=YYYY/ratios_YYYYMMDD.parquet
```

---

### Step 3 — Anchor composite ranking (Base score per anchor)

**`_0003_AnchorRanking.py`**

* Aggregates ratios into a **composite anchor score**
* Writes per-anchor snapshots:

```text
BaseAnchorRanking/year=YYYY/anchor_YYYYMMDD.parquet
```

---

### Step 4 — Investment-date rankings (Base + 4 ML variants)

These scripts produce investable rankings at each investment date:

* **`_0004_BaseInvestmentRanking.py`**

  * Features: score **slope / average / stability** over a 3-year lookback
  * Output: `BaseInvestmentRanking/year=*/invrank_YYYYMMDD.parquet`

* **`_0004_LightGBMInvestmentRanking.py`**

  * Features: **raw ratios**
  * Output: `LightGBMInvestmentRanking/year=*/LightGBMrank_YYYYMMDD.parquet`

* **`_0004_LightGBMScoreRanking.py`**

  * Features: **slope / average / stdev** of composite scores
  * Output: `LightGBMScoreRanking/year=*/LightGBMscore_rank_YYYYMMDD.parquet`

* **`_0004_RidgeInvestmentRanking.py`**

  * Features: **raw ratios**
  * Output: `RidgeRanking/year=*/ridgerank_YYYYMMDD.parquet`

* **`_0004_RidgeScoreRanking.py`**

  * Features: **slope / average / stdev** of composite scores
  * Output: `RidgeScoreRanking/year=*/ridgescore_rank_YYYYMMDD.parquet`

---

### Step 5 — Top/bottom decile selection (all models)

**`_0005_TopBottom10.py`**

* Reads each model’s ranking snapshots
* Selects:

  * **Top 10%** → `signal = +1`
  * **Bottom 10%** → `signal = -1`
  * Ensures long/short sets are **disjoint**
* Writes per-model snapshots:

```text
BaseTopBottom10/year=*/topbottom10_YYYYMMDD.parquet
LightGBMTopBottom10/year=*/topbottom10_YYYYMMDD.parquet
LightGBMScoreTopBottom10/year=*/topbottom10_YYYYMMDD.parquet
RidgeTopBottom10/year=*/topbottom10_YYYYMMDD.parquet
RidgeScoreTopBottom10/year=*/topbottom10_YYYYMMDD.parquet
```

---

### Step 6 — Daily performance (L/S + legs) + per-model VAMI

**`_0006_Performance.py`**

* Builds **overlapping 12-month cohorts** (quarterly rebalancing, 1-year holding)
* Produces daily series:

  * Long leg returns
  * Short leg returns (**P&L convention**)
  * Long–short returns
* Saves:

  * `FilteredRawData/[Model]LS_Returns.parquet`
  * `FilteredRawData/[Model]Long_Returns.parquet`
  * `FilteredRawData/[Model]Short_Returns.parquet`
  * `[Model]PerfMeasure.csv`
  * `[Model]VAMI.jpg`, `[Model]VAMI_Long.jpg`, `[Model]VAMI_Short.jpg`

---

### Step 7 — FF5 + Momentum regression on L/S (all models)

**`_0007_FFRegression.py`**

* Regresses daily **L/S returns** on **FF5 + Mom**
* Outputs:

  * `[Model]OLSFama.csv`
  * `[Model]RollingAlpha.jpg` (5-year rolling alpha)

---

### Step 8 — FF5 + Momentum regressions on long & short legs (all models)

**`_0008_FFLegsRegression.py`**

* Separate regressions for:

  * Long leg
  * Short leg (P&L convention)
* Outputs:

  * `[Model]OLSFamaLong.csv`
  * `[Model]OLSFamaShort.csv`

---

### Step 9 — Consolidated VAMI (multi-model comparison)

**`_0009_VAMI.py`**

* Builds multi-line VAMI charts comparing:

  * All L/S models vs market factor
  * (optionally) legs across models
* Useful for a single “dashboard” view beyond per-model charts.

---

### Step 10 — Hit rates

**`_0010_HitRates.py`**

* Computes hit rates for:

  * L/S
  * Long leg
  * Short leg
* Produces summary tables (CSV) for quick model comparison.

---

## 5) How to run (step-by-step)

From the project root:

```bash
# 1) Raw SAS -> Parquet
python _0001_RawDataProcessing.py

# 2) Fundamental ratios
python _0002_RatiosCalculation.py

# 3) Base anchor composite ranking
python _0003_AnchorRanking.py

# 4) Investment rankings (Base + 4 ML variants)
python _0004_BaseInvestmentRanking.py
python _0004_LightGBMInvestmentRanking.py
python _0004_LightGBMScoreRanking.py
python _0004_RidgeInvestmentRanking.py
python _0004_RidgeScoreRanking.py

# 5) Top/bottom deciles (all models)
python _0005_TopBottom10.py

# 6) Daily performance + VAMI per model
python _0006_Performance.py

# 7) FF5+Mom regression on L/S (all models)
python _0007_FFRegression.py

# 8) FF5+Mom regressions on legs (all models)
python _0008_FFLegsRegression.py

# 9) Consolidated VAMI
python _0009_VAMI.py

# 10) Hit rates
python _0010_HitRates.py
```

---

## 6) Dependencies

Python ≥ 3.10 recommended.

Core stack:

* `polars`
* `numpy`
* `pandas`
* `pyarrow`
* `matplotlib`

Modeling:

* `lightgbm`
* `scikit-learn`

Install:

```bash
pip install polars numpy pandas pyarrow matplotlib lightgbm scikit-learn
```

---

## 7) Reproducibility and no-look-ahead

This project is designed to be **strictly out-of-sample**:

* Fundamentals are aligned to investment dates with reporting/availability logic
* ML training uses rolling history and only outcomes that have fully elapsed by the training cut
* Comparisons are fair: all variants are fed the **same underlying information set**, either as raw ratios or as Base-style engineered score dynamics

---

## 8) Report / methodology note

A full write-up of the methodology, modeling choices, and results is provided in:

* `Nikonov_Ivan_Report-ADA.pdf`

---

## 9) License & data disclaimer

* **Code:** choose an OSS license (e.g. MIT) and add it to the repository.
* **Data:** CRSP / Compustat are proprietary (WRDS). This repo contains **no licensed data**. Users must obtain their own access and comply with WRDS / CRSP / S&P Global terms.
