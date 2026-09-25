#!/usr/bin/env python3
"""
Binance Vision Klines Merger & Validator
Extracts monthly klines zip files from raw/{pair}/, normalizes timestamps
to seconds, validates chronological order and continuity, and merges into
processed/{pair}.csv.
"""

import argparse
import csv
import glob
import io
import os
import re
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path


# Standard Binance spot kline header
HEADER = [
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

INTERVAL_SECONDS = {
    "1s": 1,
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
    "3d": 259200,
    "1w": 604800,
}


def parse_interval_to_seconds(interval_str: str) -> int:
    """Convert interval string (e.g. 1m, 1h, 1d) to seconds."""
    interval_str = interval_str.lower()
    if interval_str in INTERVAL_SECONDS:
        return INTERVAL_SECONDS[interval_str]

    # Try regex parsing e.g. 10m -> 600
    match = re.match(r"^(\d+)([smhd])$", interval_str)
    if match:
        val, unit = int(match.group(1)), match.group(2)
        multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
        return val * multipliers[unit]

    # Fallback default 1m
    print(f"Warning: Unknown interval '{interval_str}', defaulting to 60 seconds (1m).")
    return 60


def to_seconds(ts_val: str | int | float) -> int:
    """
    Normalizes timestamp to seconds integer.
    Handles nanoseconds (19 digits), microseconds (16 digits),
    milliseconds (13 digits), and seconds (10 digits).
    """
    ts = int(float(ts_val))
    if ts >= 10**17:  # Nanoseconds (~1e18)
        return ts // 10**9
    elif ts >= 10**14:  # Microseconds (~1e15)
        return ts // 10**6
    elif ts >= 10**11:  # Milliseconds (~1e12)
        return ts // 10**3
    return ts


def format_utc(ts_seconds: int) -> str:
    """Convert epoch timestamp in seconds to formatted UTC string."""
    return datetime.fromtimestamp(ts_seconds, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def extract_year_month_from_filename(filename: str) -> tuple[int, int]:
    """Extracts (year, month) from filename like BTCUSDT-1m-2017-08.zip."""
    match = re.search(r"(\d{4})-(\d{2})", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0


def get_sorted_zip_files(raw_dir: Path, pair: str, interval: str) -> list[Path]:
    """Find and chronologically sort all zip files for the pair."""
    patterns = [
        f"{pair}-{interval}-*.zip",
        f"{pair}-*.zip",
        "*.zip",
    ]

    files: set[Path] = set()
    for pattern in patterns:
        matched = list(raw_dir.glob(pattern))
        if matched:
            files.update(matched)
            break

    if not files:
        # Check case-insensitive
        for f in raw_dir.iterdir():
            if f.suffix.lower() == ".zip" and pair.lower() in f.name.lower():
                files.add(f)

    # Sort files chronologically by extracted year-month and name
    sorted_files = sorted(
        list(files),
        key=lambda p: (extract_year_month_from_filename(p.name), p.name)
    )
    return sorted_files


def merge_and_validate(
    pair: str,
    raw_dir: Path,
    output_file: Path,
    interval_str: str = "1m",
    fill_missing: bool = False,
    include_header: bool = True,
):
    step_seconds = parse_interval_to_seconds(interval_str)

    if not raw_dir.exists():
        print(f"Error: Raw directory not found: {raw_dir.resolve()}", file=sys.stderr)
        sys.exit(1)

    zip_files = get_sorted_zip_files(raw_dir, pair, interval_str)
    if not zip_files:
        print(f"Error: No zip files found in {raw_dir.resolve()} for pair '{pair}'", file=sys.stderr)
        sys.exit(1)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_output_file = output_file.with_suffix(".tmp")

    print("=" * 70)
    print(f" Binance Klines Merger & Validator")
    print(f" Pair:            {pair}")
    print(f" Interval:        {interval_str} ({step_seconds} seconds)")
    print(f" Raw Directory:   {raw_dir.resolve()}")
    print(f" Output File:     {output_file.resolve()}")
    print(f" Zip Files Found: {len(zip_files)} files")
    print("=" * 70)

    total_rows = 0
    duplicate_count = 0
    gaps: list[dict] = []
    prev_open_ts: int | None = None
    prev_close_price: str = "0.0"
    first_open_ts: int | None = None
    last_open_ts: int | None = None

    start_time = time.time()

    with open(temp_output_file, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        if include_header:
            writer.writerow(HEADER)

        for file_idx, zip_path in enumerate(zip_files, start=1):
            print(f"Processing [{file_idx:3d}/{len(zip_files):3d}]: {zip_path.name}...", end="\r", flush=True)

            with zipfile.ZipFile(zip_path, "r") as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_names:
                    continue

                for csv_name in csv_names:
                    with zf.open(csv_name, "r") as csv_file:
                        text_wrapper = io.TextIOWrapper(csv_file, encoding="utf-8")
                        reader = csv.reader(text_wrapper)

                        for row in reader:
                            if not row or not row[0].strip():
                                continue

                            # Check if header row
                            try:
                                open_ts_sec = to_seconds(row[0])
                            except (ValueError, IndexError):
                                # Skip header row if present in CSV
                                continue

                            # Convert close_time (column 6) if present
                            if len(row) > 6:
                                try:
                                    close_ts_sec = to_seconds(row[6])
                                    row[6] = str(close_ts_sec)
                                except ValueError:
                                    row[6] = str(open_ts_sec + step_seconds - 1)
                            else:
                                close_ts_sec = open_ts_sec + step_seconds - 1

                            row[0] = str(open_ts_sec)
                            close_price = row[4] if len(row) > 4 else prev_close_price

                            if first_open_ts is None:
                                first_open_ts = open_ts_sec

                            # Continuity and gap validation
                            if prev_open_ts is not None:
                                diff = open_ts_sec - prev_open_ts

                                if diff == 0:
                                    # Duplicate timestamp
                                    duplicate_count += 1
                                    continue
                                elif diff < 0:
                                    print(f"\n[WARNING] Out of order timestamp detected: {prev_open_ts} -> {open_ts_sec} in {zip_path.name}")
                                elif diff > step_seconds:
                                    missing_candles = (diff // step_seconds) - 1
                                    gap_info = {
                                        "start_ts": prev_open_ts + step_seconds,
                                        "end_ts": open_ts_sec - step_seconds,
                                        "start_utc": format_utc(prev_open_ts + step_seconds),
                                        "end_utc": format_utc(open_ts_sec - step_seconds),
                                        "missing_candles": missing_candles,
                                        "duration_sec": diff - step_seconds,
                                    }
                                    gaps.append(gap_info)

                                    if fill_missing:
                                        # Fill missing rows with forward fill
                                        curr_fill_ts = prev_open_ts + step_seconds
                                        while curr_fill_ts < open_ts_sec:
                                            fill_row = [
                                                str(curr_fill_ts),          # open_time
                                                prev_close_price,           # open
                                                prev_close_price,           # high
                                                prev_close_price,           # low
                                                prev_close_price,           # close
                                                "0.00000000",               # volume
                                                str(curr_fill_ts + step_seconds - 1),  # close_time
                                                "0.00000000",               # quote_asset_volume
                                                "0",                        # number_of_trades
                                                "0.00000000",               # taker_buy_base_asset_volume
                                                "0.00000000",               # taker_buy_quote_asset_volume
                                                "0",                        # ignore
                                            ]
                                            writer.writerow(fill_row)
                                            total_rows += 1
                                            curr_fill_ts += step_seconds

                            # Write validated row
                            # Ensure row has 12 columns
                            while len(row) < 12:
                                row.append("0")

                            writer.writerow(row[:12])
                            total_rows += 1
                            prev_open_ts = open_ts_sec
                            prev_close_price = close_price
                            last_open_ts = open_ts_sec

    # Rename temp file to final destination
    temp_output_file.replace(output_file)
    elapsed = time.time() - start_time
    file_size_mb = output_file.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 70)
    print(" Merge & Validation Summary:")
    print(f"  - Output File:       {output_file.resolve()}")
    print(f"  - Output File Size:  {file_size_mb:.2f} MB")
    print(f"  - Total Candles:     {total_rows:,}")
    if first_open_ts and last_open_ts:
        print(f"  - Start Time:        {format_utc(first_open_ts)} ({first_open_ts}s)")
        print(f"  - End Time:          {format_utc(last_open_ts)} ({last_open_ts}s)")
    print(f"  - Duplicates:        {duplicate_count}")
    print(f"  - Timestamp Format:  Seconds (Validated)")
    print(f"  - Gaps Detected:     {len(gaps)}")

    total_missing_candles = sum(g["missing_candles"] for g in gaps)
    print(f"  - Missing Candles:   {total_missing_candles:,}")
    print(f"  - Elapsed Time:      {elapsed:.2f}s")
    print("=" * 70)

    if gaps:
        print(f"\n[Validation Report] Timestamp Gaps Found ({len(gaps)} total):")
        print(f"{'#':<4} {'Start UTC':<22} {'End UTC':<22} {'Missing Candles':<16} {'Duration':<12}")
        print("-" * 78)
        for idx, g in enumerate(gaps[:10], start=1):
            dur_str = f"{g['duration_sec'] // 60}m" if g['duration_sec'] < 86400 else f"{g['duration_sec'] / 3600:.1f}h"
            print(f"{idx:<4} {g['start_utc']:<22} {g['end_utc']:<22} {g['missing_candles']:<16,d} {dur_str:<12}")

        if len(gaps) > 10:
            print(f"... and {len(gaps) - 10} more gaps (typically Binance maintenance windows).")
        print("-" * 78)
    else:
        print("\n[Validation Success] Perfect timestamp continuity! No missing timestamps.")


def main():
    parser = argparse.ArgumentParser(
        description="Merge and validate Binance monthly kline CSVs from raw/{pair}/ into processed/{pair}.csv"
    )
    parser.add_argument(
        "--pair",
        type=str,
        required=True,
        help="Trading pair symbol (e.g. BTCUSDT, ETHUSDT)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1m",
        help="Candle interval (default: 1m)",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Directory containing raw zip files (default: raw/{pair})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="processed",
        help="Directory for merged output CSV (default: processed)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Custom output CSV file path (overrides --output-dir)",
    )
    parser.add_argument(
        "--fill-missing",
        action="store_true",
        help="Fill missing candle gaps with previous close price and zero volume",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Do not include header row in output CSV",
    )

    args = parser.parse_args()

    pair = args.pair.strip().upper()
    interval = args.interval.strip()

    if args.raw_dir:
        raw_dir = Path(args.raw_dir)
    else:
        raw_dir = Path(f"raw/{pair}")

    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_dir = Path(args.output_dir)
        output_file = output_dir / f"{pair}.csv"

    merge_and_validate(
        pair=pair,
        raw_dir=raw_dir,
        output_file=output_file,
        interval_str=interval,
        fill_missing=args.fill_missing,
        include_header=not args.no_header,
    )


if __name__ == "__main__":
    main()
