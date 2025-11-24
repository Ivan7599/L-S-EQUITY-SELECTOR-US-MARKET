# L-S-EQUITY-SELECTOR-US-MARKET
Here’s a README you can drop into `README.md` and tweak if you want to change paths or script names.

---

# Systematic US Equity Long/Short – CRSP & Compustat

This repository implements an end-to-end **US equity long/short stock-picking backtest** using:

* **CRSP daily** prices/returns for US common stocks (financials excluded)
* **Compustat quarterly** fundamentals for US firms (financials excluded)

We build **profitability / valuation / quality** factors, derive three stock-picking models (Base, LightGBM, Ridge), form quarterly top/bottom decile long–short portfolios, and evaluate performance and FF5+Momentum alphas over 1980s–today.

> ⚠ **Important:** This project requires a valid **WRDS license**.
> The raw CRSP and Compustat data **cannot be shared** in this repo for licensing reasons.

---

## 1. Data requirements (WRDS / RawDatabase)

You must source all raw data yourself from WRDS.

1. Log into **WRDS** and go to:

   * **CRSP** → US Stock → Daily Stock File
   * **Compustat North America** → Fundamentals Quarterly

2. Universe / filters (recommended):

   * US **common stocks only**
   * **Exclude financial institutions** (e.g. using SIC / GICS or WRDS filters)
   * Reasonable date range (e.g. 1980–today)

3. Export as **SAS** files with the following names:

   * CRSP daily returns:
     `RawReturns.sas7bdat`
   * Compustat quarterly fundamentals:
     `RawFundamentals.sas7bdat`

4. Create the following folder in the project root:

   ```text
   RawDatabase/
       RawReturns.sas7bdat
       RawFundamentals.sas7bdat
       Factors5.csv      # Fama–French 5 factors (daily)
       Momentum.csv      # Daily momentum factor
   ```

   * `Factors5.csv` and `Momentum.csv` are the standard **Kenneth French daily factor files** (download them from the French website and save as CSV).

Once this folder exists, the rest of the pipeline runs locally on your machine.

---

## 2. Pipeline overview

The code is organised as a **sequential pipeline**. The scripts are meant to be run **in order**:

1. **Raw data → Parquet**
   `_0001_RawDataProcessing.py`

   * Reads `RawReturns.sas7bdat` (CRSP) and `RawFundamentals.sas7bdat` (Compustat)
   * Cleans dates and identifiers
   * Excludes financials
   * Writes year-partitioned **Parquet** datasets in `FilteredRawData/Returns_ds/` and `FilteredRawData/Fundamentals_q/`

2. **Fundamental ratios (profitability / valuation / quality)**
   `_0002_RatiosCalculation.py`

   * Aligns quarterly fundamentals to **anchor dates** that respect SEC filing lags (Q4, Q1, Q2, Q3)
   * Constructs a large set of profitability, valuation and quality ratios
   * Applies winsorisation and basic cleaning
   * Writes ratios in `Ratios/year=*/ratios_YYYYMMDD.parquet`

3. **Base model anchor ranking (per quarter)**
   `_0003_AnchorRanking.py`

   * Builds a **cross-sectional composite score** at each anchor from the three pillars
   * Stores one Parquet snapshot per anchor in `BaseAnchorRanking/year=*/anchor_YYYYMMDD.parquet`

4. **Investment-date ranking (3-year dynamics)**

   * Base: `_0004_BaseInvestmentRanking.py`
   * LightGBM: `_0004_LightGBMInvestmentRanking.py`
   * Ridge: `_0004_RidgeInvestmentRanking.py`

   These scripts:

   * Look back over the **past 3 years of anchors**
   * For the Base model, compute level / slope / stability of the composite score
   * For LightGBM and Ridge, fit return-prediction models on the same ratios and horizons (3, 6, 12 months)
   * Produce **investment scores / ranks** and write them to:

     ```text
     BaseInvestmentRanking/year=*/invrank_YYYYMMDD.parquet
     LightGBMInvestmentRanking/year=*/LightGBMrank_YYYYMMDD.parquet
     RidgeRanking/year=*/ridgerank_YYYYMMDD.parquet
     ```

5. **Top / bottom decile selection (all models)**
   `TopBottom_AllModels.py` (the unified script you have)

   * Reads the investment ranking snapshots for **Base, LightGBM, Ridge**
   * For each anchor date:

     * Sorts stocks by score
     * Takes **top 10%** as longs (`signal = +1`)
     * Takes **bottom 10%** as shorts (`signal = −1`), ensuring long/short sets are disjoint
   * Writes snapshots to:

     ```text
     BaseTopBottom10/year=*/topbottom10_YYYYMMDD.parquet
     LightGBMTopBottom10/year=*/topbottom10_YYYYMMDD.parquet
     RidgeTopBottom10/year=*/topbottom10_YYYYMMDD.parquet
     ```

6. **Daily performance and VAMI (all models)**
   `_0006_Performance_AllModels.py`

   * Reads TopBottom snapshots, CRSP daily returns, and FF market factor
   * Constructs **overlapping 12-month cohorts**:

     * Each quarter, new longs & shorts are selected and held for 12 months
     * Multiple cohorts coexist; weights are proportional to the number of names in each sleeve
   * Produces daily time series for:

     * Long leg
     * Short leg (P&L convention)
     * Long–short (L/S)
   * Outputs:

     * `FilteredRawData/[Model]LS_Returns.parquet`
     * `FilteredRawData/[Model]Long_Returns.parquet`
     * `FilteredRawData/[Model]Short_Returns.parquet`
     * Performance summary CSV `[Model]PerfMeasure.csv`
     * VAMI charts for L/S and each leg

7. **FF5+Momentum regression on L/S (all models)**
   `_0007_FFRegression_AllModels.py`

   * Reads `[Model]LS_Returns.parquet` and the FF5+Mom factors
   * Runs **OLS** of L/S daily returns on FF5 + Momentum
   * Saves full-sample alpha, factor loadings, t-stats, R² in:

     * `[Model]OLSFama.csv`
   * Also computes and plots **5-year rolling alpha** for each model:

     * `[Model]RollingAlpha.jpg`

8. **FF5+Momentum regression on long & short legs (all models)**
   `_0008_FFLegsRegression_AllModels.py`

   * Reads `[Model]Long_Returns.parquet` and `[Model]Short_Returns.parquet`
   * Runs separate FF5+Mom regressions for long and short legs
   * Saves:

     * `[Model]OLSFamaLong.csv`
     * `[Model]OLSFamaShort.csv`

---

## 3. How to run (step by step)

From the project root:

```bash
# 1) Raw SAS -> Parquet
python _0001_RawDataProcessing.py

# 2) Fundamental ratios
python _0002_RatiosCalculation.py

# 3) Base anchor rankings
python _0003_AnchorRanking.py

# 4) Investment rankings (Base, LightGBM, Ridge)
python _0004_BaseInvestmentRanking.py
python _0004_LightGBMInvestmentRanking.py
python _0004_RidgeInvestmentRanking.py

# 5) Top/bottom deciles for all models
python TopBottom_AllModels.py

# 6) Daily LS/legs and performance for all models
python _0006_Performance_AllModels.py

# 7) FF5+Mom regression on LS for all models
python _0007_FFRegression_AllModels.py

# 8) FF5+Mom regression on long & short legs for all models
python _0008_FFLegsRegression_AllModels.py
```

After this, you will have:

* Daily long–short and leg return series
* Performance tables and VAMI plots
* Factor regression tables and rolling alpha plots for Base, LightGBM, and Ridge

---

## 4. Dependencies

Typical Python stack (Python ≥ 3.10):

* `polars`
* `numpy`
* `pandas`
* `pyarrow`
* `matplotlib`
* `lightgbm` (for the LightGBM model)
* `scikit-learn` (if used elsewhere)

Example:

```bash
pip install polars numpy pandas pyarrow matplotlib lightgbm scikit-learn
```

---

## 5. License & data disclaimer

* **Code**: choose and specify your preferred open-source license (e.g. MIT).
* **Data**: CRSP and Compustat data are **proprietary**.

  * This repository does **not** include any WRDS data.
  * Users are responsible for obtaining their own access and complying with WRDS / CRSP / S&P Global licensing terms.
