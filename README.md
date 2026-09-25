# Crypto Data Prediction — Data Pipeline

Pipeline scripts to download, extract, normalize, validate, and merge historical cryptocurrency kline (candlestick) market data from [Binance Vision](https://data.binance.vision).

---

## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Prerequisites & Environment Setup](#prerequisites--environment-setup)
- [1. Downloading Data (`download.py`)](#1-downloading-data-downloadpy)
- [2. Processing & Merging Data (`merger.py`)](#2-processing--merging-data-mergerpy)
- [Processed Data Schema](#processed-data-schema)
- [Git & Data Handling](#git--data-handling)

---

## Project Overview

This repository provides two high-performance, standalone Python scripts designed to prepare historical market datasets for quantitative analysis and machine learning model training:

1. **`download.py`**: Concurrently downloads historical monthly spot kline ZIP archives directly from Binance Vision.
2. **`merger.py`**: Extracts the CSVs from downloaded archives, normalizes all timestamps to integer seconds, validates continuity / detects missing candles, and merges them into a clean, chronological CSV file.

Zero external Python dependencies are required—both scripts rely exclusively on Python standard library modules (`urllib`, `concurrent.futures`, `csv`, `zipfile`, `pathlib`, `argparse`).

---

## Directory Structure

```
crypto-data-prediction/
│
├── download.py             # Script to download monthly kline archives
├── merger.py               # Script to merge, normalize, and validate klines
├── README.md               # Pipeline documentation
├── .gitignore              # Ignores large raw/processed datasets & virtual envs
│
├── raw/                    # (Ignored by Git) Downloaded monthly zip archives
│   ├── BTCUSDT/
│   │   ├── BTCUSDT-1m-2017-08.zip
│   │   └── ...
│   └── ETHUSDT/
│       ├── ETHUSDT-1m-2017-08.zip
│       └── ...
│
└── processed/              # (Ignored by Git) Merged, validated CSV files
    ├── BTCUSDT.csv
    └── ETHUSDT.csv
```

---

## Prerequisites & Environment Setup

- **Python**: 3.10 or later

### Setting up Virtual Environment (Optional but Recommended)

```powershell
# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate on Windows (CMD)
.\.venv\Scripts\activate.bat

# Activate on Linux / macOS
source .venv/bin/activate
```

---

## 1. Downloading Data (`download.py`)

Downloads monthly spot kline ZIP archives from Binance Vision into `raw/{pair}/`.

### Basic Usage

Download default range (August 2017 to August 2026) for a trading pair:

```powershell
python download.py --pair BTCUSDT
```

Download for another pair (e.g., `ETHUSDT`, `SOLUSDT`):

```powershell
python download.py --pair ETHUSDT
```

### Options & Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--pair` | `str` | *(Required)* | Trading pair symbol (e.g. `BTCUSDT`, `ETHUSDT`) |
| `--interval` | `str` | `1m` | Kline interval (e.g. `1m`, `5m`, `15m`, `1h`, `1d`) |
| `--start` | `YYYY-MM` | `2017-08` | Start year and month |
| `--end` | `YYYY-MM` | `2026-08` | End year and month |
| `--threads` | `int` | `4` | Number of concurrent download worker threads |
| `--output-dir` | `path` | `raw/{pair}` | Destination directory for downloaded ZIP archives |

### Advanced Examples

```powershell
# Custom date range
python download.py --pair BTCUSDT --start 2021-01 --end 2023-12

# 1-hour interval candles with 8 download threads
python download.py --pair BTCUSDT --interval 1h --threads 8

# Custom output destination
python download.py --pair SOLUSDT --output-dir data/raw_sol
```

*Note: Files that already exist locally are automatically skipped.*

---

## 2. Processing & Merging Data (`merger.py`)

Takes the downloaded monthly `.zip` files from `raw/{pair}/`, extracts each CSV, normalizes timestamps, validates data integrity, and outputs `./processed/{pair}.csv`.

### Key Validation & Normalization Rules:
- **Timestamp Format**: Converts timestamps (`open_time` and `close_time`) from milliseconds (`13` digits), microseconds (`16` digits), or nanoseconds (`19` digits) into standard integer **seconds** (`10` digits).
- **Chronological Sorting**: Ensures rows are strictly sorted from earliest to latest.
- **Continuity & Gap Detection**: Checks every consecutive candle for missing timestamps ($\Delta t > \text{interval}$). Detects and logs exchange downtime / maintenance periods.
- **Deduplication**: Identifies and discards duplicate rows.

### Basic Usage

```powershell
python merger.py --pair BTCUSDT
```

```powershell
python merger.py --pair ETHUSDT
```

### Options & Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--pair` | `str` | *(Required)* | Trading pair symbol (e.g. `BTCUSDT`, `ETHUSDT`) |
| `--interval` | `str` | `1m` | Candle interval (`1s`, `1m`, `3m`, `5m`, `15m`, `1h`, `1d`, etc.) |
| `--raw-dir` | `path` | `raw/{pair}` | Directory containing raw ZIP archives |
| `--output-dir` | `path` | `processed` | Target folder for the merged CSV |
| `--output-file` | `path` | `processed/{pair}.csv` | Explicit destination file path |
| `--fill-missing` | `flag` | `False` | Forward-fill missing candles (carries forward previous close with 0 volume) |
| `--no-header` | `flag` | `False` | Exclude CSV column header row |

### Advanced Examples

```powershell
# Merge and forward-fill any missing candles during exchange maintenance
python merger.py --pair BTCUSDT --fill-missing

# Merge higher timeframe candles (e.g., 1h)
python merger.py --pair BTCUSDT --interval 1h
```

---

## Processed Data Schema

The merged CSV in `./processed/{pair}.csv` adheres to the following column specification:

| # | Column Name | Type | Description |
| :-: | :--- | :--- | :--- |
| 1 | `open_time` | `int` | Kline open timestamp in **seconds** (Unix epoch) |
| 2 | `open` | `float` | Open price |
| 3 | `high` | `float` | Highest price during interval |
| 4 | `low` | `float` | Lowest price during interval |
| 5 | `close` | `float` | Close price |
| 6 | `volume` | `float` | Base asset trading volume |
| 7 | `close_time` | `int` | Kline close timestamp in **seconds** (Unix epoch) |
| 8 | `quote_asset_volume` | `float` | Quote asset trading volume |
| 9 | `number_of_trades` | `int` | Total number of trades executed |
| 10 | `taker_buy_base_asset_volume` | `float` | Taker buy base asset volume |
| 11 | `taker_buy_quote_asset_volume` | `float` | Taker buy quote asset volume |
| 12 | `ignore` | `float` | Binance internal field (usually 0) |

---

## Git & Data Handling

The `.gitignore` is configured to prevent committing large data files and runtime artifacts:
- Raw ZIP files (`raw/`, `*.zip`)
- Processed CSV files (`processed/`, `*.csv`)
- Virtual environments (`.venv/`, `venv/`)
- Python bytecode and cache (`__pycache__/`)

Only source code, pipelines, and documentation are tracked by Git.
