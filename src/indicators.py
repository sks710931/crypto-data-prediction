"""
Technical Indicators & Feature Engineering Module.

Implements all 4 core families of market indicators:
  1. Trend Indicators (EMA, SMA, MACD, ADX, DI, Supertrend + derived spreads/slopes)
  2. Momentum Indicators (RSI, ROC, StochRSI, CCI, Williams %R + RSI momentum)
  3. Volatility Indicators (ATR, NATR, Bollinger Bands, Return Volatility, Parkinson Volatility, Keltner)
  4. Volume Indicators (RVOL, Volume Z-Score, OBV, MFI, CMF, Rolling VWAP + VWAP distance)
"""

import time
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import pandas_ta as ta

from .config import TIMEFRAME_SECONDS, TRAIN_DATA_DIR
from .storage import load_parquet


# =====================================================================
# 1. Trend Indicators
# =====================================================================

def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Trend indicators and derived features:
      - EMAs: 9, 21, 50, 200
      - SMAs: 20, 50
      - Price distance from EMA 21 (%)
      - EMA 9 vs EMA 21 spread (%)
      - EMA 21 slope (%)
      - MACD (12, 26, 9) + Histogram + Histogram rate of change
      - ADX (14), +DI, -DI + (+DI - -DI) difference
      - Supertrend (10, 3) + trend direction
    """
    res = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # EMAs
    res["ema_9"] = ta.ema(close, length=9)
    res["ema_21"] = ta.ema(close, length=21)
    res["ema_50"] = ta.ema(close, length=50)
    res["ema_200"] = ta.ema(close, length=200)

    # SMAs
    res["sma_20"] = ta.sma(close, length=20)
    res["sma_50"] = ta.sma(close, length=50)

    # Derived EMA features (scale-invariant %)
    res["price_dist_ema21_pct"] = (close - res["ema_21"]) / res["ema_21"] * 100.0
    res["ema9_ema21_spread_pct"] = (res["ema_9"] - res["ema_21"]) / res["ema_21"] * 100.0
    res["ema21_slope_pct"] = res["ema_21"].pct_change() * 100.0

    # MACD (12, 26, 9)
    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        # MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        res["macd"] = macd_df.iloc[:, 0]
        res["macd_hist"] = macd_df.iloc[:, 1]
        res["macd_signal"] = macd_df.iloc[:, 2]
        res["macd_hist_change"] = res["macd_hist"].diff()

    # ADX and Directional Indicators (14)
    adx_df = ta.adx(high, low, close, length=14)
    if adx_df is not None and not adx_df.empty:
        # ADX_14, DMP_14, DMN_14
        res["adx_14"] = adx_df["ADX_14"]
        res["plus_di_14"] = adx_df["DMP_14"]
        res["minus_di_14"] = adx_df["DMN_14"]
        res["di_diff_14"] = res["plus_di_14"] - res["minus_di_14"]

    # Supertrend (10, 3)
    st_df = ta.supertrend(high, low, close, length=10, multiplier=3.0)
    if st_df is not None and not st_df.empty:
        # SUPERT_10_3.0, SUPERTd_10_3.0 (1.0 for bull, -1.0 for bear)
        res["supertrend_10_3"] = st_df.iloc[:, 0].astype("float64")
        res["supertrend_dir_10_3"] = st_df.iloc[:, 1].astype("float32")

    return res


# =====================================================================
# 2. Momentum Indicators
# =====================================================================

def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Momentum indicators:
      - RSI: 7, 14
      - RSI recent change (momentum of RSI)
      - Rate of Change (ROC): 5, 10, 20
      - Stochastic RSI: (14, 3, 3) %K and %D
      - CCI: 20
      - Williams %R: 14
    """
    res = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # RSI
    res["rsi_7"] = ta.rsi(close, length=7)
    res["rsi_14"] = ta.rsi(close, length=14)
    res["rsi_14_change"] = res["rsi_14"].diff()

    # ROC (Rate of Change %)
    res["roc_5"] = close.pct_change(5) * 100.0
    res["roc_10"] = close.pct_change(10) * 100.0
    res["roc_20"] = close.pct_change(20) * 100.0

    # Stochastic RSI (14, 3, 3)
    stochrsi_df = ta.stochrsi(close, length=14, rsi_length=14, k=3, d=3)
    if stochrsi_df is not None and not stochrsi_df.empty:
        res["stochrsi_k_14_3_3"] = stochrsi_df.iloc[:, 0]
        res["stochrsi_d_14_3_3"] = stochrsi_df.iloc[:, 1]

    # CCI (20)
    res["cci_20"] = ta.cci(high, low, close, length=20)

    # Williams %R (14)
    res["williams_r_14"] = ta.willr(high, low, close, length=14)

    return res


# =====================================================================
# 3. Volatility Indicators
# =====================================================================

def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Volatility indicators:
      - ATR: 14
      - Normalized ATR (NATR %): 14
      - Bollinger Bands: (20, 2) + %b + bandwidth
      - Rolling return volatility: 5, 15, 30, 60
      - Parkinson volatility: 20
      - Keltner Channels: (20, 2)
    """
    res = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # ATR & Normalized ATR (NATR)
    res["atr_14"] = ta.atr(high, low, close, length=14)
    res["natr_14"] = (res["atr_14"] / close) * 100.0

    # Bollinger Bands (20, 2)
    bb_df = ta.bbands(close, length=20, std=2.0)
    if bb_df is not None and not bb_df.empty:
        # BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0 (bandwidth), BBP_20_2.0 (%b)
        res["bb_lower_20_2"] = bb_df.iloc[:, 0]
        res["bb_mid_20_2"] = bb_df.iloc[:, 1]
        res["bb_upper_20_2"] = bb_df.iloc[:, 2]
        res["bb_bandwidth_20_2"] = bb_df.iloc[:, 3]
        res["bb_pct_b_20_2"] = bb_df.iloc[:, 4]

    # Rolling Return Volatility (%)
    returns = close.pct_change()
    for w in [5, 15, 30, 60]:
        res[f"volatility_returns_{w}"] = returns.rolling(window=w).std() * 100.0

    # Parkinson Volatility (20): High/Low based volatility estimator
    hl_ratio = np.log(high / low)
    res["parkinson_vol_20"] = np.sqrt(
        (hl_ratio ** 2).rolling(window=20).mean() / (4.0 * np.log(2.0))
    ) * 100.0

    # Keltner Channels (20, 2)
    kc_df = ta.kc(high, low, close, length=20, scalar=2.0)
    if kc_df is not None and not kc_df.empty:
        # KCLe_20_2.0, KCBe_20_2.0, KCUe_20_2.0
        res["keltner_lower_20_2"] = kc_df.iloc[:, 0]
        res["keltner_mid_20_2"] = kc_df.iloc[:, 1]
        res["keltner_upper_20_2"] = kc_df.iloc[:, 2]

    return res


# =====================================================================
# 4. Volume Indicators
# =====================================================================

def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Volume indicators:
      - Relative Volume (RVOL): 20
      - Volume Z-Score: 20
      - OBV (On-Balance Volume)
      - MFI: 14
      - Chaikin Money Flow (CMF): 20
      - Rolling VWAP: 20, 60
      - Price distance from VWAP (%)
    """
    res = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # RVOL & Volume Z-Score (20)
    vol_mean_20 = volume.rolling(window=20).mean()
    vol_std_20 = volume.rolling(window=20).std()
    res["rvol_20"] = volume / vol_mean_20.where(vol_mean_20 != 0)
    res["volume_zscore_20"] = (volume - vol_mean_20) / vol_std_20.where(vol_std_20 != 0)

    # OBV
    res["obv"] = ta.obv(close, volume)

    # MFI (14)
    res["mfi_14"] = ta.mfi(high, low, close, volume, length=14)

    # CMF (20)
    res["cmf_20"] = ta.cmf(high, low, close, volume, length=20)

    # Rolling VWAP (20, 60)
    typical_price = (high + low + close) / 3.0
    price_volume = typical_price * volume

    for w in [20, 60]:
        pv_sum = price_volume.rolling(window=w).sum()
        v_sum = volume.rolling(window=w).sum()
        vwap = pv_sum / v_sum.where(v_sum != 0)
        res[f"vwap_{w}"] = vwap
        res[f"dist_vwap_{w}_pct"] = (close - vwap) / vwap * 100.0

    return res


# =====================================================================
# 5. Master Feature Generation Pipeline
# =====================================================================

def compute_all_indicators(
    df: pd.DataFrame,
    drop_warmup: bool = True,
    warmup_period: int = 200,
    interval: str = "1m",
) -> pd.DataFrame:
    """
    Computes all 4 families of technical indicators and joins them with base OHLCV data.

    Parameters:
      df: Base OHLCV DataFrame (must include open_time, open, high, low, close, volume)
      drop_warmup: If True, drops the first `warmup_period` rows (due to EMA 200 warmup)
      warmup_period: Number of initial rows to drop if drop_warmup is True (default: 200)

    Returns:
      DataFrame containing base OHLCV + all engineered features.
    """
    if interval not in TIMEFRAME_SECONDS:
        raise ValueError(f"Unsupported interval: {interval}")
    if not df.open_time.is_monotonic_increasing or df.open_time.duplicated().any():
        raise ValueError("Candles must have unique, increasing open_time values")
    if (df.open_time % TIMEFRAME_SECONDS[interval] != 0).any():
        raise ValueError("Candles must be aligned to the requested interval")

    start_time = time.time()
    print("Computing technical indicators across 4 categories...")
    step = TIMEFRAME_SECONDS[interval]
    starts = np.r_[0, np.flatnonzero(np.diff(df.open_time.to_numpy()) != step) + 1]
    ends = np.r_[starts[1:], len(df)]
    next_open = df.open_time.shift(-1)
    next_candle_open = df.open.shift(-1)
    next_candle_close = df.close.shift(-1)
    label_valid = (next_open - df.open_time == step) & (next_candle_close != next_candle_open)
    labels = pd.Series((next_candle_close > next_candle_open).astype("int8"), index=df.index).astype("Int8")
    labels.loc[~label_valid] = pd.NA

    blocks = []
    dropped = 0
    for start, end in zip(starts, ends):
        segment = df.iloc[start:end].reset_index(drop=True)
        if drop_warmup and len(segment) <= warmup_period:
            dropped += len(segment)
            continue
        trend = add_trend_indicators(segment)
        momentum = add_momentum_indicators(segment)
        volatility = add_volatility_indicators(segment)
        volume = add_volume_indicators(segment)
        block = pd.concat([segment, trend, momentum, volatility, volume], axis=1)
        block["target_next_up"] = pd.array(labels.iloc[start:end].to_numpy(), dtype="Int8")
        if drop_warmup:
            block = block.iloc[warmup_period:]
            dropped += warmup_period
        blocks.append(block)
    if not blocks:
        raise ValueError("No segment contains enough candles to pass the warmup period")
    features_df = pd.concat(blocks, ignore_index=True)
    features_df.attrs["warmup_rows_dropped"] = dropped
    features_df.attrs["segment_count"] = len(starts)
    print(f"Segments: {len(starts):,}; explicitly excluded warmup rows: {dropped:,}.")
    total_features = len(features_df.columns) - len(df.columns) - 1
    print(f"Feature engineering complete: {total_features} new features and one target in {time.time() - start_time:.2f}s.")
    return features_df


def generate_and_save_features(
    input_parquet: Path | str,
    output_parquet: Optional[Path | str] = None,
    drop_warmup: bool = True,
    warmup_period: int = 200,
    interval: str = "1m",
) -> Path:
    """
    Loads OHLCV Parquet file, computes all technical indicators, and saves to Parquet.
    """
    input_parquet = Path(input_parquet)
    if not input_parquet.exists():
        raise FileNotFoundError(f"Input Parquet not found: {input_parquet.resolve()}")

    if output_parquet is None:
        # e.g., BTCUSDT_5m_features.parquet
        output_parquet = TRAIN_DATA_DIR / f"{input_parquet.stem}_features.parquet"
    else:
        output_parquet = Path(output_parquet)

    print(f"\nLoading: {input_parquet.name}...")
    df = load_parquet(input_parquet)
    print(f"Loaded {len(df):,} candles.")

    features_df = compute_all_indicators(
        df=df,
        drop_warmup=drop_warmup,
        warmup_period=warmup_period,
        interval=interval,
    )

    print(f"Writing features to: {output_parquet.name}...")
    features_df.to_parquet(output_parquet, compression="snappy", engine="pyarrow")
    size_mb = output_parquet.stat().st_size / (1024 * 1024)
    print(f"Saved: {output_parquet.name} ({size_mb:.2f} MB, {len(features_df.columns)} total columns)")

    return output_parquet
