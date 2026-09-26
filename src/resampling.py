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
    return_excluded: bool = False,
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
    if not drop_incomplete_last:
        raise ValueError("Incomplete candles cannot enter a training dataset")

    pandas_freq = TIMEFRAME_MAP[target_timeframe]
    target_seconds = TIMEFRAME_SECONDS[target_timeframe]
    if target_seconds <= 60 or target_seconds % 60:
        raise ValueError("Target timeframe must contain whole 1m candles")

    # Create temporary UTC datetime index for accurate resampling
    temp_df = df.copy()
    temp_df["dt"] = pd.to_datetime(temp_df["open_time"], unit="s", utc=True)
    temp_df.set_index("dt", inplace=True)
    temp_df["source_valid"] = (
        (temp_df["open_time"] % 60 == 0)
        & (temp_df["close_time"] - temp_df["open_time"] == 59)
    )
    if "is_synthetic" in temp_df:
        temp_df["source_valid"] &= ~temp_df["is_synthetic"].astype(bool)

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
    source_groups = temp_df.resample(pandas_freq, closed="left", label="left")
    counts = source_groups["open_time"].count()
    first = source_groups["open_time"].min()
    last = source_groups["open_time"].max()
    unique = source_groups["open_time"].nunique()
    source_valid = source_groups["source_valid"].all()
    # Pandas can retain second-resolution datetime indexes; their int64 values
    # are already seconds, so do not assume nanoseconds here.
    starts = resampled.index.tz_localize(None).astype("datetime64[s]").astype("int64")
    complete = (
        (counts == target_seconds // 60)
        & (unique == target_seconds // 60)
        & (first == starts)
        & (last == starts + target_seconds - 60)
        & source_valid
    )
    excluded_mask = ((counts > 0) & ~complete).to_numpy()
    excluded = pd.DataFrame({
        "open_time": starts.to_numpy()[excluded_mask],
        "source_count": counts.to_numpy()[excluded_mask],
        "reason": "incomplete_or_irregular_source_minutes",
    })
    resampled = resampled.loc[complete]

    # Reconstruct integer second timestamps
    resampled["open_time"] = resampled.index.tz_localize(None).astype("datetime64[s]").astype("int64")
    resampled["close_time"] = resampled["open_time"] + target_seconds - 1

    # All partial bins, including the last one, were removed above.

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
    if return_excluded:
        return resampled, excluded.reset_index(drop=True)
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
