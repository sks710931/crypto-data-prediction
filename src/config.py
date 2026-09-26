"""
Configuration and constants for data pipeline and feature engineering.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "processed"
PARQUET_DIR = PROCESSED_DIR / "parquet"
TRAIN_DATA_DIR = BASE_DIR / "train-data"

# Ensure directories exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Standard Binance 1m Kline Columns
RAW_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]

# Optimized Column Dtypes for Storage and Computation
COLUMN_DTYPES = {
    "open_time": "int64",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "close_time": "int64",
    "quote_asset_volume": "float64",
    "number_of_trades": "int32",
    "taker_buy_base_asset_volume": "float64",
    "taker_buy_quote_asset_volume": "float64",
    "ignore": "float32",
}

# Standard timeframe string to pandas frequency mapping
TIMEFRAME_MAP = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1d": "1D",
}

# Timeframe to duration in seconds
TIMEFRAME_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "1d": 86400,
}

# Contract for the engineered feature store. The target is checked separately.
FEATURE_COLUMNS = [
    "ema_9", "ema_21", "ema_50", "ema_200", "sma_20", "sma_50",
    "price_dist_ema21_pct", "ema9_ema21_spread_pct", "ema21_slope_pct",
    "macd", "macd_hist", "macd_signal", "macd_hist_change",
    "adx_14", "plus_di_14", "minus_di_14", "di_diff_14",
    "supertrend_10_3", "supertrend_dir_10_3",
    "rsi_7", "rsi_14", "rsi_14_change", "roc_5", "roc_10", "roc_20",
    "stochrsi_k_14_3_3", "stochrsi_d_14_3_3", "cci_20", "williams_r_14",
    "atr_14", "natr_14", "bb_lower_20_2", "bb_mid_20_2", "bb_upper_20_2",
    "bb_bandwidth_20_2", "bb_pct_b_20_2", "volatility_returns_5",
    "volatility_returns_15", "volatility_returns_30", "volatility_returns_60",
    "parkinson_vol_20", "keltner_lower_20_2", "keltner_mid_20_2", "keltner_upper_20_2",
    "rvol_20", "volume_zscore_20", "obv", "mfi_14", "cmf_20",
    "vwap_20", "dist_vwap_20_pct", "vwap_60", "dist_vwap_60_pct",
]
