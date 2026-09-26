#!/usr/bin/env python3
"""
Binance Vision Klines Merger & Validator
Extracts monthly klines zip files from raw/{pair}/, normalizes timestamps
to seconds, validates chronological order and continuity, and merges into
processed/{pair}.csv.
"""

import argparse
import csv
from contextlib import ExitStack
import io
import json
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
SYNTHETIC_COLUMN = "is_synthetic"

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

    raise ValueError(f"Unknown interval '{interval_str}'")


def to_seconds(ts_val: str | int | float) -> int:
    """
    Normalizes timestamp to seconds integer.
    Handles nanoseconds (19 digits), microseconds (16 digits),
    milliseconds (13 digits), and seconds (10 digits).
    """
    ts = int(ts_val)
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
    files = set(raw_dir.glob(f"{pair}-{interval}-*.zip"))

    # Sort files chronologically by extracted year-month and name
    sorted_files = sorted(
        list(files),
        key=lambda p: (extract_year_month_from_filename(p.name), p.name)
    )
    return sorted_files


def synthetic_gap_rows(previous_ts: int, next_ts: int, step_seconds: int, previous_close: str):
    """Yield marked flat placeholders; the price and zero activity are synthetic."""
    for ts in range(previous_ts + step_seconds, next_ts, step_seconds):
        yield [
            str(ts), previous_close, previous_close, previous_close, previous_close,
            "0", str(ts + step_seconds - 1), "0", "0", "0", "0", "0", "1",
        ]


def clock_grid_exclusion_reason(open_ts: int, close_ts: int, step_seconds: int) -> str | None:
    """Explain why a source candle cannot occupy a complete UTC clock slot."""
    if open_ts % step_seconds:
        return "unaligned_open_time"
    if close_ts - open_ts != step_seconds - 1:
        return "irregular_duration"
    return None


def merge_and_validate(
    pair: str,
    raw_dir: Path,
    output_file: Path,
    interval_str: str = "1m",
    fill_missing: bool = False,
    include_header: bool = True,
    clock_grid: bool = False,
):
    if clock_grid and not fill_missing:
        raise ValueError("--clock-grid requires --fill-missing")
    if fill_missing and not include_header:
        raise ValueError("--fill-missing requires a header so synthetic rows remain identifiable")
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
    exceptions_path = output_file.with_suffix(".excluded_source.csv")
    temp_exceptions = exceptions_path.with_suffix(".csv.tmp")

    print("=" * 70)
    print(f" Binance Klines Merger & Validator")
    print(f" Pair:            {pair}")
    print(f" Interval:        {interval_str} ({step_seconds} seconds)")
    print(f" Raw Directory:   {raw_dir.resolve()}")
    print(f" Output File:     {output_file.resolve()}")
    print(f" Zip Files Found: {len(zip_files)} files")
    print("=" * 70)

    total_rows = 0
    observed_rows = 0
    synthetic_rows = 0
    excluded_source_rows = 0
    duplicate_count = 0
    misaligned_count = 0
    irregular_duration_count = 0
    gaps: list[dict] = []
    prev_open_ts: int | None = None
    prev_row: list[str] | None = None
    prev_source_open_ts: int | None = None
    prev_source_row: list[str] | None = None
    prev_close_price: str | None = None
    previous_archive: str | None = None
    first_open_ts: int | None = None
    last_open_ts: int | None = None
    fills: list[dict] = []

    start_time = time.time()

    with ExitStack() as stack:
        out_f = stack.enter_context(open(temp_output_file, "w", newline="", encoding="utf-8"))
        writer = csv.writer(out_f)
        exceptions_writer = None
        if clock_grid:
            exceptions_f = stack.enter_context(open(temp_exceptions, "w", newline="", encoding="utf-8"))
            exceptions_writer = csv.writer(exceptions_f)
            exceptions_writer.writerow(HEADER + ["reason", "source_archive"])
        if include_header:
            writer.writerow(HEADER + ([SYNTHETIC_COLUMN] if fill_missing else []))

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
                            except (ValueError, IndexError) as exc:
                                if row[0].strip().lower() == "open_time":
                                    continue
                                raise ValueError(f"Malformed timestamp in {zip_path.name}: {row!r}") from exc

                            if len(row) != len(HEADER):
                                raise ValueError(f"Expected {len(HEADER)} fields in {zip_path.name}, found {len(row)}: {row!r}")

                            # Convert close_time (column 6) if present
                            if len(row) > 6:
                                try:
                                    close_ts_sec = to_seconds(row[6])
                                    row[6] = str(close_ts_sec)
                                except ValueError as exc:
                                    raise ValueError(f"Malformed close_time in {zip_path.name}: {row!r}") from exc
                            else:
                                close_ts_sec = open_ts_sec + step_seconds - 1

                            row[0] = str(open_ts_sec)
                            if prev_source_open_ts is not None:
                                if open_ts_sec < prev_source_open_ts:
                                    raise ValueError(f"Out of order source timestamp {prev_source_open_ts} -> {open_ts_sec} in {zip_path.name}")
                                if open_ts_sec == prev_source_open_ts:
                                    if row != prev_source_row:
                                        raise ValueError(f"Conflicting duplicate source candle at {open_ts_sec} in {zip_path.name}")
                                    duplicate_count += 1
                                    continue
                            prev_source_open_ts = open_ts_sec
                            prev_source_row = row.copy()
                            misaligned_count += int(open_ts_sec % step_seconds != 0)
                            irregular_duration_count += int(close_ts_sec - open_ts_sec != step_seconds - 1)
                            reason = clock_grid_exclusion_reason(open_ts_sec, close_ts_sec, step_seconds)
                            if clock_grid and reason is not None:
                                assert exceptions_writer is not None
                                exceptions_writer.writerow(row + [reason, zip_path.name])
                                excluded_source_rows += 1
                                continue
                            if first_open_ts is None:
                                first_open_ts = open_ts_sec

                            # Continuity and gap validation
                            if prev_open_ts is not None:
                                diff = open_ts_sec - prev_open_ts

                                if diff == 0:
                                    # Duplicate timestamp
                                    if row != prev_row:
                                        raise ValueError(f"Conflicting duplicate candle at {open_ts_sec} in {zip_path.name}")
                                    duplicate_count += 1
                                    continue
                                elif diff < 0:
                                    raise ValueError(f"Out of order timestamp {prev_open_ts} -> {open_ts_sec} in {zip_path.name}")
                                elif diff > step_seconds:
                                    missing_candles = (diff - 1) // step_seconds
                                    gap_info = {
                                        "previous_observed_ts": prev_open_ts,
                                        "next_observed_ts": open_ts_sec,
                                        "start_ts": prev_open_ts + step_seconds,
                                        "end_ts": prev_open_ts + missing_candles * step_seconds,
                                        "start_utc": format_utc(prev_open_ts + step_seconds),
                                        "end_utc": format_utc(prev_open_ts + missing_candles * step_seconds),
                                        "missing_candles": missing_candles,
                                        "duration_sec": diff - step_seconds,
                                        "boundary_residual_sec": diff - missing_candles * step_seconds,
                                    }
                                    gaps.append(gap_info)
                                    if fill_missing:
                                        assert prev_close_price is not None
                                        for synthetic in synthetic_gap_rows(prev_open_ts, open_ts_sec, step_seconds, prev_close_price):
                                            writer.writerow(synthetic)
                                            synthetic_rows += 1
                                            total_rows += 1
                                        fills.append({
                                            **gap_info,
                                            "synthetic_rows_inserted": missing_candles,
                                            "price_source": "previous observed close",
                                            "volume_and_trades": "synthetic zero",
                                            "source_archive_before": previous_archive,
                                            "source_archive_after": zip_path.name,
                                        })

                            # Write validated row
                            writer.writerow(row + (["0"] if fill_missing else []))
                            total_rows += 1
                            observed_rows += 1
                            prev_open_ts = open_ts_sec
                            prev_row = row.copy()
                            prev_close_price = row[4]
                            previous_archive = zip_path.name
                            last_open_ts = open_ts_sec

    # Rename temp file to final destination
    temp_output_file.replace(output_file)
    if clock_grid:
        temp_exceptions.replace(exceptions_path)
    if fill_missing:
        report_path = output_file.with_suffix(".fill_report.json")
        report = {
            "pair": pair,
            "interval": interval_str,
            "source": str(raw_dir.resolve()),
            "output_csv": str(output_file.resolve()),
            "synthetic_column": SYNTHETIC_COLUMN,
            "clock_grid": clock_grid,
            "observed_rows": observed_rows,
            "excluded_source_rows": excluded_source_rows,
            "excluded_source_csv": str(exceptions_path.resolve()) if clock_grid else None,
            "synthetic_rows": synthetic_rows,
            "total_rows": total_rows,
            "gap_intervals_filled": len(fills),
            "off_grid_observed_rows": misaligned_count,
            "irregular_observed_durations": irregular_duration_count,
            "nonuniform_gap_boundaries": sum(g["boundary_residual_sec"] != step_seconds for g in fills),
            "fills": fills,
            "warning": "Synthetic rows are placeholders, not observed trades. Exclude them from model training and indicator state.",
        }
        temp_report = report_path.with_suffix(".json.tmp")
        temp_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temp_report.replace(report_path)
    elapsed = time.time() - start_time
    file_size_mb = output_file.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 70)
    print(" Merge & Validation Summary:")
    print(f"  - Output File:       {output_file.resolve()}")
    print(f"  - Output File Size:  {file_size_mb:.2f} MB")
    print(f"  - Total Candles:     {total_rows:,}")
    if fill_missing:
        print(f"  - Observed Candles:  {observed_rows:,}")
        print(f"  - Synthetic Candles: {synthetic_rows:,}")
        if clock_grid:
            print(f"  - Excluded Source:   {excluded_source_rows:,} (saved to {exceptions_path.resolve()})")
        print(f"  - Filling Report:    {report_path.resolve()}")
    if first_open_ts and last_open_ts:
        print(f"  - Start Time:        {format_utc(first_open_ts)} ({first_open_ts}s)")
        print(f"  - End Time:          {format_utc(last_open_ts)} ({last_open_ts}s)")
    print(f"  - Duplicates:        {duplicate_count}")
    print(f"  - Misaligned Opens:  {misaligned_count:,}")
    print(f"  - Irregular Durations:{irregular_duration_count:,}")
    print(f"  - Timestamp Format:  Seconds (Validated)")
    print(f"  - Gaps Detected:     {len(gaps)}")

    total_missing_candles = sum(g["missing_candles"] for g in gaps)
    print(f"  - Gap Slots Found:   {total_missing_candles:,}")
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
            print(f"... and {len(gaps) - 10} more gaps (cause not inferred).")
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
        help="Insert marked synthetic flat candles and write a .fill_report.json; unsuitable as observed training data",
    )
    parser.add_argument(
        "--clock-grid",
        action="store_true",
        help="For fill mode, put only aligned, full-duration source bars on the uniform grid and save source exceptions separately",
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
        clock_grid=args.clock_grid,
    )


if __name__ == "__main__":
    main()
