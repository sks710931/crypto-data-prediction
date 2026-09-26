# Crypto Data Prediction — Data Pipeline

An end-to-end, high-performance historical market data pipeline to download, merge, normalize, audit, resample, and compute technical indicators on cryptocurrency candlestick (kline) data from [Binance Vision](https://data.binance.vision) into optimized Parquet feature stores for quantitative trading and machine learning model training.

See [BTCUSDT_5m_AUDIT.md](BTCUSDT_5m_AUDIT.md) for the verified 2017–2026 BTCUSDT 5m data audit and remaining limitations.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Prerequisites & Environment Setup](#prerequisites--environment-setup)
- [1. Downloading Data (`download.py`)](#1-downloading-data-downloadpy)
- [2. Processing & Merging Data (`merger.py`)](#2-processing--merging-data-mergerpy)
- [3. Pipeline Orchestrator (`main.py`)](#3-pipeline-orchestrator-mainpy)
  - [Step 3.1: Parquet Conversion (`src/storage.py`)](#step-31-parquet-conversion-srcstoragepy)
  - [Step 3.2: Data Quality & Integrity Validation (`src/validation.py`)](#step-32-data-quality--integrity-validation-srcvalidationpy)
  - [Step 3.3: Multi-Timeframe Resampling (`src/resampling.py`)](#step-33-multi-timeframe-resampling-srcresamplingpy)
  - [Step 3.4: Technical Indicators Engine (`src/indicators.py`)](#step-34-technical-indicators-engine-srcindicatorspy)
  - [Orchestrator Options & Usage](#orchestrator-options--usage)
- [Feature Store Schema (65 Columns for 5m)](#feature-store-schema-65-columns-for-5m)
- [Git & Data Handling](#git--data-handling)

---

## Project Overview

This repository provides a modular, production-ready pipeline designed to prepare multi-year crypto market datasets (such as BTCUSDT from August 2017 to present):

1. **`download.py`**: Concurrently downloads historical monthly spot kline ZIP archives directly from Binance Vision.
2. **`merger.py`**: Extracts CSVs from monthly archives, normalizes timestamps to seconds, validates continuity, and merges them into a single consolidated CSV.
3. **`main.py` (Orchestrator)**:
   * **Converts CSV to Parquet**: Achieves ~2.5x disk compression and sub-second load times via PyArrow.
   * **Audits Data Quality**: Checks for nulls, price anomalies ($H \ge L, H \ge O, H \ge C$), volume violations, and exchange downtime gaps.
   * **5m source selection**: Uses complete official Binance Vision 5m monthly archives when present. Otherwise, resamples only complete, clock-aligned 1m bins. Source exclusions are recorded in a quality JSON file.
   * **Computes 53 Technical Indicators**: Vectorised calculations across **Trend**, **Momentum**, **Volatility**, and **Volume** families.

---

## Directory Structure

```
crypto-data-prediction/
├── download.py                  # Downloads monthly kline archives from Binance
├── merger.py                    # Merges & extracts raw archives into processed CSV
├── main.py                      # Master pipeline orchestrator
├── requirements.txt             # Python dependencies
├── README.md                    # Pipeline documentation
├── .gitignore                   # Ignores large raw/processed datasets & virtual envs
│
├── src/                         # Modular pipeline components
│   ├── __init__.py
│   ├── config.py                # Global configurations, schemas, and timeframe maps
│   ├── validation.py            # Comprehensive data integrity & gap audit
│   ├── storage.py               # High-speed PyArrow CSV <-> Parquet I/O
│   ├── resampling.py            # Financial OHLCV multi-timeframe resampling
│   ├── indicators.py            # 53 technical indicators & derived features
│   └── training_readiness.py    # Point-in-time label and split audit
│
├── raw/                         # (Ignored by Git) Downloaded monthly zip archives
│   └── BTCUSDT/
│       ├── BTCUSDT-1m-2017-08.zip
│       └── ...
│
├── processed/                   # (Ignored by Git) Merged and intermediate market data
│   ├── BTCUSDT.csv              # Full merged 1m CSV
│   └── parquet/
│       ├── BTCUSDT_1m.parquet   # Source 1m candles
│       └── BTCUSDT_5m.parquet   # Valid observed 5m candles
└── train-data/                  # (Ignored by Git) Training features and audit artifacts
    ├── BTCUSDT_5m_features.parquet
    ├── BTCUSDT_5m_training_rows.parquet
    ├── BTCUSDT_5m_training_readiness.md
    ├── BTCUSDT_5m_snapshot.json
    └── snapshots/              # Frozen, content-addressed feature and row files
```

---

## Prerequisites & Environment Setup

- **Python**: 3.12 (Recommended for latest NumPy and Pandas compatibility)

### Setting up Virtual Environment

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Windows (CMD):
.\.venv\Scripts\activate.bat

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 1. Downloading Data (`download.py`)

Downloads monthly spot kline ZIP archives from Binance Vision into `raw/{pair}/`.

```bash
python download.py --pair BTCUSDT
```

---

## 2. Processing & Merging Data (`merger.py`)

Extracts CSVs from `raw/{pair}/*.zip`, normalizes timestamps to Unix epoch seconds, validates continuity, and outputs `processed/{pair}.csv`.

```bash
python merger.py --pair BTCUSDT
```

To publish a **uniform, synthetic clock-grid view** and a per-gap filling report, run:

```bash
python merger.py --pair BTCUSDT --interval 1m --fill-missing --clock-grid --output-file processed/BTCUSDT_1m_grid.csv
python merger.py --pair BTCUSDT --interval 5m --fill-missing --clock-grid --output-file processed/BTCUSDT_5m_grid.csv
```

Every generated candle has `is_synthetic=1`, flat OHLC equal to the last observed close, and zero volume and trades. The report is written beside the CSV as `.fill_report.json`. Clock-shifted or incomplete exchange bars are preserved in `.excluded_source.csv`. These grid files give uniform timestamps for calendar analysis; synthetic rows are **not observed market data** and must not be used as training labels or indicator continuity. The observed-only 5m feature pipeline still uses the direct archives and remains gap-aware. See [BTCUSDT_FILLING_REPORT.md](BTCUSDT_FILLING_REPORT.md) for actual counts and limitations.

---

## 3. Pipeline Orchestrator (`main.py`)

The orchestrator transforms raw merged data into validated Parquet datasets and generates a complete technical indicator feature store:

```bash
python main.py --pair BTCUSDT --interval 5m
```

### Step 3.1: Parquet Conversion (`src/storage.py`)
* Converts the large CSV (645+ MB) into column-oriented Snappy-compressed Parquet.
* Implements a strict PyArrow schema (Timestamps $\to$ `int64`, Prices $\to$ `float64`, Volumes $\to$ `float64`, Trades $\to$ `int32`).
* Cuts loading time from **~25s down to 0.37s**.

### Step 3.2: Data Quality & Integrity Validation (`src/validation.py`)
* **Price Sanity:** Verifies $\text{High} \ge \max(\text{Open}, \text{Close}, \text{Low})$ and $\text{Low} \le \min(\text{Open}, \text{Close}, \text{High})$ and all prices $> 0$.
* **Volume Sanity:** Verifies $\text{Volume} \ge 0$ and $\text{Taker Buy Volume} \le \text{Total Volume}$.
* **Chronological Ordering:** Verifies strictly monotonic timestamps ($t_{i} > t_{i-1}$) with zero duplicates.
* **Gap Report:** Detects and ranks unavailable clock intervals without attributing their cause.

### Step 3.3: Multi-Timeframe Resampling (`src/resampling.py`)
Resamples 1-minute base data into higher timeframes using proper financial aggregation:
* $\text{Open} = \text{first}, \quad \text{High} = \max, \quad \text{Low} = \min, \quad \text{Close} = \text{last}$
* $\text{Volume} = \sum, \quad \text{Quote Volume} = \sum, \quad \text{Trades} = \sum, \quad \text{Taker Buy Volume} = \sum$
* Left-closed, left-labeled $[T, T + \Delta t)$ intervals.
* Rejects incomplete bins anywhere in history, including bins with shifted or truncated 1m source candles.

### Step 3.4: Technical Indicators Engine (`src/indicators.py`)
Computes **53 indicators and scale-invariant derived features** across 4 categories:

#### 1. Trend Indicators (19 features)
- **EMAs**: 9, 21, 50, 200 (`ema_9`, `ema_21`, `ema_50`, `ema_200`)
- **SMAs**: 20, 50 (`sma_20`, `sma_50`)
- **Price Distance from EMA 21 (%)**: `price_dist_ema21_pct = (close - ema_21) / ema_21 * 100`
- **EMA 9 vs EMA 21 Spread (%)**: `ema9_ema21_spread_pct = (ema_9 - ema_21) / ema_21 * 100`
- **EMA 21 Slope (%)**: `ema21_slope_pct = ema_21.pct_change() * 100`
- **MACD (12, 26, 9)**: Line, signal, histogram, and histogram rate of change (`macd`, `macd_signal`, `macd_hist`, `macd_hist_change`)
- **ADX & Directional Movement (14)**: `adx_14`, `plus_di_14`, `minus_di_14`, and spread `di_diff_14` (`plus_di_14 - minus_di_14`)
- **Supertrend (10, 3)**: Price band (`supertrend_10_3`) and direction (`supertrend_dir_10_3`: +1.0 bullish, -1.0 bearish)

#### 2. Momentum Indicators (10 features)
- **RSI**: 7, 14 (`rsi_7`, `rsi_14`)
- **RSI Momentum**: `rsi_14_change = rsi_14.diff()`
- **Rate of Change (ROC %)**: 5, 10, 20 (`roc_5`, `roc_10`, `roc_20`)
- **Stochastic RSI (14, 3, 3)**: `%K` and `%D` (`stochrsi_k_14_3_3`, `stochrsi_d_14_3_3`)
- **Commodity Channel Index (CCI 20)**: `cci_20`
- **Williams %R (14)**: `williams_r_14`

#### 3. Volatility Indicators (15 features)
- **ATR & Normalized ATR (NATR %)**: `atr_14`, `natr_14 = (atr_14 / close) * 100`
- **Bollinger Bands (20, 2)**: Lower, middle, upper bands, bandwidth (`bb_bandwidth_20_2`), and position `%b` (`bb_pct_b_20_2`)
- **Rolling Return Volatility (%)**: 5, 15, 30, 60 periods (`volatility_returns_5`, `15`, `30`, `60`)
- **Parkinson Volatility (20)**: High/Low range volatility estimator (`parkinson_vol_20`)
- **Keltner Channels (20, 2)**: Lower, middle, upper channels (`keltner_lower_20_2`, `mid`, `upper`)

#### 4. Volume Indicators (9 features)
- **Relative Volume (RVOL 20)**: `rvol_20 = volume / rolling_mean(20)`
- **Volume Z-Score (20)**: Standardized volume deviation (`volume_zscore_20`)
- **On-Balance Volume (OBV)**: `obv`
- **Money Flow Index (MFI 14)**: Volume-weighted RSI (`mfi_14`)
- **Chaikin Money Flow (CMF 20)**: Accumulation/Distribution oscillator (`cmf_20`)
- **Rolling VWAP (20, 60)**: `vwap_20`, `vwap_60`
- **Price Distance from VWAP (%)**: `dist_vwap_20_pct`, `dist_vwap_60_pct`

---

### Orchestrator Options & Usage

```bash
# 1. Audit 1m directly (stops on BTCUSDT's historical off-grid source bars)
python main.py --pair BTCUSDT

# 2. Use complete official 5m archives when available; otherwise strict resampling
python main.py --pair BTCUSDT --interval 5m

# 3. Higher timeframes (e.g., 15m, 1h)
python main.py --pair BTCUSDT --interval 15m
```

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--pair` | `str` | `BTCUSDT` | Trading pair symbol (e.g. `BTCUSDT`, `ETHUSDT`) |
| `--interval` | `str` | `1m` | Candle interval (e.g. `1m`, `5m`, `15m`, `1h`). |
| `--source` | `auto`, `1m`, `5m` | `auto` | 5m source selection; direct 5m requires complete monthly archives. |
| `--train-end` | UTC date | `2026-03-01` | First day of validation; the preceding next-candle label is embargoed. |
| `--validation-end` | UTC date | `2026-06-01` | First day of the untouched test period. |
| `--color` | `auto`, `always`, `never` | `auto` | Console colors; `always` overrides `NO_COLOR`. |

For the primary 5m dataset, download official 5m archives with `python download.py --pair BTCUSDT --interval 5m --start 2017-08 --end 2026-08`. `--source auto` (the default) uses them only when every month covering the local 1m range is present. Use `--source 1m` to force strict 1m resampling, or `--source 5m` to require direct archives. The pipeline never fills gaps with synthetic candles. It drops and counts the first 200 rows of **each continuous segment** so rolling indicators and EMA state cannot cross a gap. `train-data/BTCUSDT_5m_quality.json` records source exclusions, gaps, warmup exclusions, and unavailable labels.

Each retained 5m row has a nullable `target_next_up`: 1 when the **next contiguous 5m candle** closes above its open, 0 when it closes below, and null for a gap, the dataset end, or a flat next candle. Row T's features are available when candle T closes. Exclude `target_next_up` from the XGBoost feature matrix, select only rows with a non-null target for supervised training, and use chronological validation splits.

The sixth pipeline stage produces a training-readiness report in `train-data/BTCUSDT_5m_training_readiness.md` and JSON. It independently checks every target by timestamp, checks three causal feature prefixes, profiles each year and split, declares a fixed 62-column input whitelist, and records row-level inclusion or exclusion in `BTCUSDT_5m_training_rows.parquet`. It also records SHA-256 hashes of source, code, and output files in `BTCUSDT_5m_snapshot.json` and keeps content-addressed copies of the feature data and row ledger in `train-data/snapshots/`. Training uses no feature scaling or imputation. The report computes a training-prevalence baseline on validation and reserves test metrics for final model evaluation. This is a probability-forecasting workflow; trading costs and rules are outside its scope.
The readiness status can pass with modeling warnings. In particular, the report compares later-period feature distributions with the training period and flags substantial drift; this does not change or drop the existing feature definitions.
The default split trains through February 2026, validates on March-May 2026, and reserves June-August 2026 for one final test. After a model has passed that test, a separate final fit can include all observations through August 2026; its performance must then be monitored on newly arriving candles.

---

## Feature Store Schema (65 Columns for 5m)

The output Parquet feature store (`train-data/BTCUSDT_5m_features.parquet`) contains:

| Category | Columns |
| :--- | :--- |
| **Base OHLCV** (11) | `open_time`, `open`, `high`, `low`, `close`, `volume`, `close_time`, `quote_asset_volume`, `number_of_trades`, `taker_buy_base_asset_volume`, `taker_buy_quote_asset_volume` |
| **Trend** (19) | `ema_9`, `ema_21`, `ema_50`, `ema_200`, `sma_20`, `sma_50`, `price_dist_ema21_pct`, `ema9_ema21_spread_pct`, `ema21_slope_pct`, `macd`, `macd_hist`, `macd_signal`, `macd_hist_change`, `adx_14`, `plus_di_14`, `minus_di_14`, `di_diff_14`, `supertrend_10_3`, `supertrend_dir_10_3` |
| **Momentum** (10) | `rsi_7`, `rsi_14`, `rsi_14_change`, `roc_5`, `roc_10`, `roc_20`, `stochrsi_k_14_3_3`, `stochrsi_d_14_3_3`, `cci_20`, `williams_r_14` |
| **Volatility** (15) | `atr_14`, `natr_14`, `bb_lower_20_2`, `bb_mid_20_2`, `bb_upper_20_2`, `bb_bandwidth_20_2`, `bb_pct_b_20_2`, `volatility_returns_5`, `volatility_returns_15`, `volatility_returns_30`, `volatility_returns_60`, `parkinson_vol_20`, `keltner_lower_20_2`, `keltner_mid_20_2`, `keltner_upper_20_2` |
| **Volume** (9) | `rvol_20`, `volume_zscore_20`, `obv`, `mfi_14`, `cmf_20`, `vwap_20`, `dist_vwap_20_pct`, `vwap_60`, `dist_vwap_60_pct` |
| **Target** (1) | `target_next_up` (nullable 0/1, never an input feature) |

Williams %R is null when its 14-candle high-low range is zero; no arbitrary oscillator value is inserted. Volume ratios and rolling VWAP remain null when their denominator is zero. Validation separates these mathematical exceptions from critical missing values and reports unavailable clock slots without assuming the cause.

---

## Git & Data Handling

The `.gitignore` is configured to prevent committing large data files and runtime artifacts:
- Raw ZIP files (`raw/`, `*.zip`)
- Processed and training data (`processed/`, `train-data/`, `*.csv`, `*.parquet`)
- Virtual environments (`.venv/`, `venv/`)
- Python bytecode and cache (`__pycache__/`)

Only source code, pipelines, and documentation are tracked by Git.
