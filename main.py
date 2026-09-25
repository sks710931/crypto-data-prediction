#!/usr/bin/env python3
"""
Streamlined Market Data Pipeline Orchestrator.

Usage:
  python main.py --pair BTCUSDT
  python main.py --pair BTCUSDT --interval 5m

Pipeline Steps:
  1. Convert merged CSV to optimized 1m Parquet (if not already done).
  2. Resample to target interval at start (skipped if interval is 1m).
  3. Validate OHLCV data integrity & continuity.
  4. Compute 53 technical indicators & derived features (drops warmup by default).
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import PARQUET_DIR, PROCESSED_DIR
from src.indicators import compute_all_indicators
from src.resampling import resample_ohlcv
from src.storage import csv_to_parquet, load_parquet
from src.validation import validate_ohlcv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Streamlined Crypto Market Data Pipeline Orchestrator"
    )
    parser.add_argument(
        "--pair",
        type=str,
        default="BTCUSDT",
        help="Trading pair symbol (default: BTCUSDT)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1m",
        help="Candle interval (default: 1m, e.g. 1m, 5m, 15m, 1h)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pair = args.pair.strip().upper()
    interval = args.interval.strip().lower()

    input_csv = PROCESSED_DIR / f"{pair}.csv"
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    parquet_1m = PARQUET_DIR / f"{pair}_1m.parquet"

    total_start = time.time()

    print("=" * 68)
    print("        STREAMLINED CRYPTO DATA PIPELINE ORCHESTRATOR")
    print("=" * 68)
    print(f" Pair:       {pair}")
    print(f" Interval:   {interval}")
    print("=" * 68)

    # -------------------------------------------------------------
    # Step 1: Convert CSV to Base 1m Parquet
    # -------------------------------------------------------------
    if not parquet_1m.exists():
        if not input_csv.exists():
            print(f"Error: Neither {parquet_1m.name} nor {input_csv.resolve()} found.", file=sys.stderr)
            print("Please run 'merger.py' first.", file=sys.stderr)
            sys.exit(1)
        print("\n[1/4] Converting CSV to 1m Parquet...")
        csv_to_parquet(csv_path=input_csv, parquet_path=parquet_1m)
    else:
        print(f"\n[1/4] Found base Parquet: {parquet_1m.name} ({parquet_1m.stat().st_size / (1024 * 1024):.1f} MB)")

    # -------------------------------------------------------------
    # Step 2: Resampling (At Start if interval != 1m)
    # -------------------------------------------------------------
    if interval != "1m":
        parquet_interval = PARQUET_DIR / f"{pair}_{interval}.parquet"
        print(f"\n[2/4] Resampling base 1m data to {interval} at start...")
        df_1m = load_parquet(parquet_1m)
        working_df = resample_ohlcv(df_1m, target_timeframe=interval)
        working_df.to_parquet(parquet_interval, compression="snappy", engine="pyarrow")
        print(f"  -> Resampled to {len(working_df):,} candles.")
        print(f"  -> Saved: {parquet_interval.name}")
    else:
        print("\n[2/4] Interval is 1m. Resampling skipped.")
        working_df = load_parquet(parquet_1m)

    # -------------------------------------------------------------
    # Step 3: Data Quality & Integrity Validation
    # -------------------------------------------------------------
    print(f"\n[3/4] Validating {pair} ({interval}) data...")
    val_start = time.time()
    report = validate_ohlcv(working_df, interval=interval)
    print(report.summary())
    print(f"Validation completed in {time.time() - val_start:.2f}s.")

    # -------------------------------------------------------------
    # Step 4: Technical Indicators & Feature Addition (Warmup Dropped)
    # -------------------------------------------------------------
    print(f"\n[4/4] Computing Technical Indicators for {pair} ({interval})...")
    features_df = compute_all_indicators(
        df=working_df,
        drop_warmup=True,
        warmup_period=200,
    )

    output_features_parquet = PARQUET_DIR / f"{pair}_{interval}_features.parquet"
    print(f"Writing features to: {output_features_parquet.name}...")
    features_df.to_parquet(output_features_parquet, compression="snappy", engine="pyarrow")
    size_mb = output_features_parquet.stat().st_size / (1024 * 1024)

    # -------------------------------------------------------------
    # Pipeline Summary
    # -------------------------------------------------------------
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 68)
    print("                 PIPELINE EXECUTION COMPLETE")
    print("=" * 68)
    print(f" Total Elapsed Time:   {total_elapsed:.2f}s")
    print(f" Output Dataset:       {output_features_parquet.name}")
    print(f" Rows (Post-Warmup):   {len(features_df):,}")
    print(f" Columns:              {len(features_df.columns)} (11 OHLCV + 53 Indicators)")
    print(f" File Size:            {size_mb:.2f} MB")
    print(f" Saved Location:       {output_features_parquet.resolve()}")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
