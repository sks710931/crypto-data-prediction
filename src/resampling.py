"""
OHLCV Resampling Engine.
Aggregates 1-minute base data into higher timeframes (5m, 15m, 1h, etc.)
strictly adhering to financial candle conventions and preventing lookahead bias.
"""

import time
from pathlib import Path
from typing import Optional
import pandas as pd

from .config import PARQUET_DIR, TIMEFRAME_MAP, TIMEFRAME_SECONDS
from .storage import load_parquet


def resample_ohlcv(
    df: pd.DataFrame,
    target_timeframe: str,
    drop_incomplete_last: bool = True,
) -> pd.DataFrame:
    """
    Resamples 1-minute OHLCV DataFrame into a higher timeframe (e.g., '5m', '15m', '1h').

    Aggregation Rules:
      - open: first
      - high: max
      - low: min
      - close: last
      - volume: sum
      - quote_asset_volume: sum
      - number_of_trades: sum
      - taker_buy_base_asset_volume: sum
      - taker_buy_quote_asset_volume: sum
    """
    if target_timeframe not in TIMEFRAME_MAP:
        raise ValueError(
            f"Unsupported timeframe '{target_timeframe}'. Supported: {list(TIMEFRAME_MAP.keys())}"
        )

    pandas_freq = TIMEFRAME_MAP[target_timeframe]
    target_seconds = TIMEFRAME_SECONDS[target_timeframe]

    # Create temporary UTC datetime index for accurate resampling
    temp_df = df.copy()
    temp_df["dt"] = pd.to_datetime(temp_df["open_time"], unit="s", utc=True)
    temp_df.set_index("dt", inplace=True)

    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    optional_sum_cols = [
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    for col in optional_sum_cols:
        if col in temp_df.columns:
            agg_dict[col] = "sum"

    # Resample with left-closed, left-labeled intervals: [T, T + interval)
    resampled = temp_df.resample(pandas_freq, closed="left", label="left").agg(agg_dict)

    # Drop intervals with zero observations (exchange maintenance / gaps)
    resampled.dropna(subset=["close"], inplace=True)

    # Reconstruct integer second timestamps
    resampled["open_time"] = resampled.index.tz_localize(None).astype("datetime64[s]").astype("int64")
    resampled["close_time"] = resampled["open_time"] + target_seconds - 1

    # Check and optionally drop the last candle if incomplete
    if drop_incomplete_last and len(resampled) > 0:
        last_open_ts = resampled["open_time"].iloc[-1]
        raw_last_open_ts = df["open_time"].iloc[-1]
        # If the latest 1m timestamp is before the candle's end, drop incomplete bar
        if (last_open_ts + target_seconds - 60) > raw_last_open_ts:
            resampled = resampled.iloc[:-1]

    # Organize column ordering
    ordered_cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
    ]
    for col in optional_sum_cols:
        if col in resampled.columns:
            ordered_cols.append(col)

    resampled = resampled.reset_index(drop=True)[ordered_cols]
    return resampled


def resample_and_save(
    input_parquet_path: Path | str,
    timeframes: list[str],
    output_dir: Optional[Path | str] = None,
    pair: str = "BTCUSDT",
) -> dict[str, Path]:
    """
    Resamples a base 1-minute Parquet file into multiple timeframes and saves each.
    """
    input_parquet_path = Path(input_parquet_path)
    output_dir = Path(output_dir) if output_dir else PARQUET_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading base 1m Parquet: {input_parquet_path.name}...")
    load_start = time.time()
    df_1m = load_parquet(input_parquet_path)
    print(f"Base data loaded: {len(df_1m):,} candles in {time.time() - load_start:.2f}s.")

    saved_paths: dict[str, Path] = {}

    for tf in timeframes:
        if tf == "1m":
            continue

        print(f"\nResampling to {tf} ({TIMEFRAME_SECONDS.get(tf, 0)}s)...")
        resample_start = time.time()
        df_resampled = resample_ohlcv(df_1m, target_timeframe=tf)
        resample_elapsed = time.time() - resample_start

        out_path = output_dir / f"{pair}_{tf}.parquet"
        df_resampled.to_parquet(out_path, compression="snappy", engine="pyarrow")

        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"  -> Generated {len(df_resampled):,} candles in {resample_elapsed:.2f}s.")
        print(f"  -> Saved: {out_path.name} ({size_mb:.2f} MB)")
        saved_paths[tf] = out_path

    return saved_paths
