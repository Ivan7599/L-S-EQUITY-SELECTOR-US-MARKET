# L-S-EQUITY-SELECTOR-US-MARKET

Systematic **US equity long/short** stock-selection pipeline using **CRSP daily** market data and **Compustat quarterly** fundamentals (non-financials). The project’s central question is:

> **Is machine learning stronger than economical logic?**  
> We compare a transparent, economically structured baseline signal to ML models trained under strict anti-overfitting constraints and the **same information set**. <!-- :contentReference[oaicite:0]{index=0} -->

This repository implements the full workflow: **data → ratios → rankings → portfolios → performance → FF5+Mom alphas → VAMI → hit rates**.

> ⚠ **WRDS license required.** CRSP/Compustat data are proprietary and cannot be included in this repository.

---

## What this project does

At each quarterly investment date, we **rank the US stock universe** and form a **1 USD long / 1 USD short** portfolio by going:
- **Long** the **top 10%** of stocks
- **Short** the **bottom 10%** of stocks

Portfolios are held in **overlapping 12-month cohorts** (new cohort each quarter; multiple cohorts coexist). <!-- :contentReference[oaicite:1]{index=1} -->

Models implemented:

1. **Base (Economical Logic)**  
   - Build a quarterly **composite score** from 3 pillars: profitability, valuation, quality  
   - Convert the last ~3 years of composite history into 3 features: **slope (trend), average (level), stdev of changes (stability)**  
   - Final score = simple combination of those components (interpretable). <!-- :contentReference[oaicite:2]{index=2} -->

2. **LightGBM (Raw ratios as features)**  
   - Predict forward returns (3m/6m/12m) from **raw ratios** at the investment date, then blend into a single ranking score. <!-- :contentReference[oaicite:3]{index=3} -->

3. **LightGBMScore (Base-style engineered features)**  
   - Same LightGBM framework, but features are **(slope, average, stdev)** of composite scores (Base information content). <!-- :contentReference[oaicite:4]{index=4} -->

4. **Ridge (Raw ratios as features)**  
   - Linear shrinkage model predicting forward returns (3m/6m/12m) from **raw ratios** (with strong regularization). <!-- :contentReference[oaicite:5]{index=5} -->

5. **RidgeScore (Base-style engineered features)**  
   - Same Ridge framework, but features are **(slope, average, stdev)** of composite scores. <!-- :contentReference[oaicite:6]{index=6} -->

---

## 1) Data requirements (WRDS / RawDatabase)

You must source all raw data from WRDS.

### 1.1 WRDS extracts (SAS)
Export the following files as SAS datasets:

- CRSP daily returns: `RawReturns.sas7bdat`
- Compustat quarterly fundamentals: `RawFundamentals.sas7bdat`

Recommended universe filters:
- US **common stocks**
- **Exclude financial institutions** (SIC/GICS, or WRDS filters)
- Large date coverage (e.g., 1980–today) <!-- :contentReference[oaicite:7]{index=7} -->

### 1.2 Factor files
Download daily factors and save as CSV:
- `Factors5.csv` (daily FF5)
- `Momentum.csv` (daily momentum)

### 1.3 Folder layout (required)
Create this folder in the project root:

```text
RawDatabase/
    RawReturns.sas7bdat
    RawFundamentals.sas7bdat
    Factors5.csv
    Momentum.csv
2) Pipeline (scripts must be run in order)
_0001_RawDataProcessing.py — Raw WRDS → Parquet (year-partitioned)
Reads SAS files from RawDatabase/

Cleans identifiers and dates

Excludes financials

Writes year-partitioned Parquet datasets:

text
Copier le code
FilteredRawData/Returns_ds/year=YYYY/*.parquet
FilteredRawData/Fundamentals_q/year=YYYY/*.parquet
This design enables fast, low-memory processing on large panels. <!-- :contentReference[oaicite:8]{index=8} -->

_0002_RatiosCalculation.py — Fundamental ratios (Profitability / Valuation / Quality)
Builds quarterly anchor / as-of dates that approximate information availability (SEC filing lags)

Joins CRSP market data with Compustat quarterly fundamentals available by the as-of date

Computes a curated set of firm characteristics (ratios) across the three pillars

Writes ratio snapshots:

text
Copier le code
Ratios/year=YYYY/ratios_YYYYMMDD.parquet
(Implementation respects an “available information” constraint by filtering fundamentals using announcement dates or lag rules.) <!-- :contentReference[oaicite:9]{index=9} -->

_0003_AnchorRanking.py — Base composite score at each anchor
Standardizes ratio directions so “higher = better”

Aggregates the 3 pillars into a single composite anchor score

Writes snapshots:

text
Copier le code
BaseAnchorRanking/year=YYYY/anchor_YYYYMMDD.parquet
Investment-date ranking scripts (3-year rolling dynamics)
All ranking scripts use a rolling historical window (≈ 3 years / ~12 anchors) to build an investment-date score.

_0004_BaseInvestmentRanking.py
Uses the Base composite score history

Builds three interpretable features per stock:

slope (trend)

average (level)

stdev of changes (stability)

Writes:

text
Copier le code
BaseInvestmentRanking/year=YYYY/invrank_YYYYMMDD.parquet
<!-- :contentReference[oaicite:10]{index=10} -->
_0004_LightGBMInvestmentRanking.py (features = raw ratios)
Features: raw ratios

Targets: forward returns at 3m / 6m / 12m; blended into one ranking score

Writes:

text
Copier le code
LightGBMInvestmentRanking/year=YYYY/LightGBMrank_YYYYMMDD.parquet
<!-- :contentReference[oaicite:11]{index=11} -->
_0004_LightGBMScoreRanking.py (features = Base engineered features)
Features: (slope, average, stdev) of Base composite scores

Same LightGBM training / prediction framework

Writes:

text
Copier le code
LightGBMScoreRanking/year=YYYY/LightGBMscore_rank_YYYYMMDD.parquet
<!-- :contentReference[oaicite:12]{index=12} -->
_0004_RidgeInvestmentRanking.py (features = raw ratios)
Features: raw ratios

Targets: forward returns at 3m / 6m / 12m; blended into one ranking score

Writes:

text
Copier le code
RidgeRanking/year=YYYY/ridgerank_YYYYMMDD.parquet
<!-- :contentReference[oaicite:13]{index=13} -->
_0004_RidgeScoreRanking.py (features = Base engineered features)
Features: (slope, average, stdev) of Base composite scores

Same Ridge training / prediction framework

Writes:

text
Copier le code
RidgeScoreRanking/year=YYYY/ridgescore_rank_YYYYMMDD.parquet
<!-- :contentReference[oaicite:14]{index=14} -->
_0005_TopBottom10.py — Top/bottom decile selection (ALL models)
For each investment snapshot:

takes top 10% (signal = +1)

takes bottom 10% (signal = −1), disjoint from longs

Writes:

text
Copier le code
BaseTopBottom10/year=YYYY/topbottom10_YYYYMMDD.parquet
LightGBMTopBottom10/year=YYYY/topbottom10_YYYYMMDD.parquet
LightGBMScoreTopBottom10/year=YYYY/topbottom10_YYYYMMDD.parquet
RidgeTopBottom10/year=YYYY/topbottom10_YYYYMMDD.parquet
RidgeScoreTopBottom10/year=YYYY/topbottom10_YYYYMMDD.parquet
_0006_Performance.py — Daily performance, cohort aggregation, VAMI per model
Builds overlapping 12-month cohorts rebalanced quarterly (new cohort each quarter)

Produces daily series for:

Long leg returns

Short leg returns (P&L convention)

Long–short (L/S)

Outputs per model:

text
Copier le code
FilteredRawData/[Model]LS_Returns.parquet
FilteredRawData/[Model]Long_Returns.parquet
FilteredRawData/[Model]Short_Returns.parquet
[Model]PerfMeasure.csv
[Model]VAMI.jpg
[Model]VAMI_Long.jpg
[Model]VAMI_Short.jpg
Performance metrics include Sharpe/Sortino/max drawdown and other diagnostics. <!-- :contentReference[oaicite:15]{index=15} -->

_0007_FFRegression.py — FF5+Momentum regression on L/S (ALL models)
OLS regression: L/S daily returns on FF5 + Momentum

Writes:

text
Copier le code
[Model]OLSFama.csv
[Model]RollingAlpha.jpg    # 5-year rolling alpha
Alpha decomposition and rolling alpha are used to compare model robustness across regimes. <!-- :contentReference[oaicite:16]{index=16} -->

_0008_FFLegsRegression.py — FF5+Momentum regressions on Long and Short legs (ALL models)
OLS regressions:

Long leg returns on FF5+Mom

Short leg returns on FF5+Mom

Writes:

text
Copier le code
[Model]OLSFamaLong.csv
[Model]OLSFamaShort.csv
_0009_VAMI.py — Global VAMI (ALL models on one chart)
Produces a single chart comparing cumulative performance across models (and optionally vs market factor, depending on your implementation).

Output typically saved as a .jpg/.png figure.

_0010_HitRates.py — Hit rates (ALL models)
Computes hit rates (directional accuracy) for:

L/S

Long leg

Short leg

Outputs a summary table (CSV) with hit rates per model. <!-- :contentReference[oaicite:17]{index=17} -->

3) How to run (step-by-step)
From the project root:

bash
Copier le code
# 1) Raw SAS -> Parquet
python _0001_RawDataProcessing.py

# 2) Fundamental ratios
python _0002_RatiosCalculation.py

# 3) Base anchor rankings
python _0003_AnchorRanking.py

# 4) Investment rankings (5 variants)
python _0004_BaseInvestmentRanking.py
python _0004_LightGBMInvestmentRanking.py
python _0004_LightGBMScoreRanking.py
python _0004_RidgeInvestmentRanking.py
python _0004_RidgeScoreRanking.py

# 5) Top/bottom deciles
python _0005_TopBottom10.py

# 6) Performance + per-model VAMI
python _0006_Performance.py

# 7) FF regressions on LS + rolling alpha
python _0007_FFRegression.py

# 8) FF regressions on legs
python _0008_FFLegsRegression.py

# 9) One chart: VAMI of all models
python _0009_VAMI.py

# 10) Hit rates per model
python _0010_HitRates.py
4) Dependencies
Python ≥ 3.10 recommended.

Core:

polars

numpy

pandas

pyarrow

matplotlib

Models:

lightgbm

scikit-learn

Example:

bash
Copier le code
pip install polars numpy pandas pyarrow matplotlib lightgbm scikit-learn
5) License & data disclaimer
Code: choose a license (e.g., MIT) and add a LICENSE file.

Data: CRSP and Compustat are proprietary (WRDS/S&P Global licensing).
This repository does not include any raw WRDS data. Users are responsible for obtaining access and complying with licensing terms.

6) Reference report (methodology + results)
A full write-up of the methodology, modeling choices, and empirical results is available in the project report. <!-- :contentReference[oaicite:18]{index=18} -->
