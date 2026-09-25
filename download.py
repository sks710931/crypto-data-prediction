#!/usr/bin/env python3
"""
Binance Vision Monthly Klines Downloader
Downloads monthly kline data (August 2017 to July 2026 by default)
for a given trading pair into raw/{pair}/
"""

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"


def generate_month_list(start_year: int, start_month: int, end_year: int, end_month: int):
    """Generate list of (year, month) tuples within the specified range."""
    months = []
    current_year, current_month = start_year, start_month
    while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
        months.append((current_year, current_month))
        if current_month == 12:
            current_year += 1
            current_month = 1
        else:
            current_month += 1
    return months


def download_file(url: str, dest_path: Path, max_retries: int = 3, timeout: int = 30) -> tuple[bool, str]:
    """
    Downloads a single file from URL to dest_path.
    Returns (success: bool, status_message: str).
    """
    filename = dest_path.name

    # Check if already downloaded
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True, f"[SKIPPED] {filename} (already exists)"

    temp_path = dest_path.with_suffix(".tmp")

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status != 200:
                    return False, f"[FAILED] {filename} (HTTP {response.status})"

                with open(temp_path, "wb") as f:
                    while True:
                        chunk = response.read(64 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)

            # Move temporary file to final destination once download completes successfully
            temp_path.replace(dest_path)
            size_mb = dest_path.stat().st_size / (1024 * 1024)
            return True, f"[DOWNLOADED] {filename} ({size_mb:.2f} MB)"

        except urllib.error.HTTPError as e:
            if temp_path.exists():
                temp_path.unlink()
            if e.code == 404:
                return False, f"[NOT FOUND] {filename} (HTTP 404 - data not available)"
            if attempt == max_retries:
                return False, f"[ERROR] {filename} (HTTP {e.code}: {e.reason})"
            time.sleep(1 * attempt)

        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if temp_path.exists():
                temp_path.unlink()
            if attempt == max_retries:
                return False, f"[ERROR] {filename} (Failed after {max_retries} attempts: {e})"
            time.sleep(2 * attempt)

    return False, f"[FAILED] {filename} (Unknown error)"


def parse_year_month(date_str: str) -> tuple[int, int]:
    """Parse string formatted as YYYY-MM into (year, month)."""
    try:
        dt = datetime.strptime(date_str.strip(), "%Y-%m")
        return dt.year, dt.month
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: '{date_str}'. Expected format is YYYY-MM (e.g. 2017-08).")


def main():
    parser = argparse.ArgumentParser(
        description="Download Binance spot monthly klines from data.binance.vision"
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
        help="Kline interval (default: 1m)",
    )
    parser.add_argument(
        "--start",
        type=parse_year_month,
        default="2017-08",
        help="Start year-month (default: 2017-08)",
    )
    parser.add_argument(
        "--end",
        type=parse_year_month,
        default="2026-07",
        help="End year-month (default: 2026-07)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output directory (default: raw/{pair})",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of concurrent download threads (default: 4)",
    )

    args = parser.parse_args()

    pair = args.pair.strip().upper()
    interval = args.interval.strip()
    start_year, start_month = args.start
    end_year, end_month = args.end

    if args.output_dir:
        dest_dir = Path(args.output_dir)
    else:
        dest_dir = Path(f"raw/{pair}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    month_list = generate_month_list(start_year, start_month, end_year, end_month)
    total_files = len(month_list)

    print("=" * 65)
    print(" Binance Monthly Klines Downloader")
    print(f" Pair:         {pair}")
    print(f" Interval:     {interval}")
    print(f" Range:        {start_year}-{start_month:02d} to {end_year}-{end_month:02d} ({total_files} months)")
    print(f" Destination:  {dest_dir.resolve()}")
    print(f" Threads:      {args.threads}")
    print("=" * 65)

    tasks = []
    for year, month in month_list:
        filename = f"{pair}-{interval}-{year:04d}-{month:02d}.zip"
        url = f"{BASE_URL}/{pair}/{interval}/{filename}"
        target_file = dest_dir / filename
        tasks.append((url, target_file))

    success_count = 0
    not_found_count = 0
    error_count = 0
    skipped_count = 0

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        future_to_task = {
            executor.submit(download_file, url, target_file): (url, target_file)
            for url, target_file in tasks
        }

        for idx, future in enumerate(as_completed(future_to_task), start=1):
            success, message = future.result()
            print(f"[{idx:3d}/{total_files:3d}] {message}")

            if success:
                if "already exists" in message:
                    skipped_count += 1
                else:
                    success_count += 1
            else:
                if "404" in message or "not available" in message:
                    not_found_count += 1
                else:
                    error_count += 1

    elapsed = time.time() - start_time
    print("=" * 65)
    print(" Download Summary:")
    print(f"  - Downloaded: {success_count}")
    print(f"  - Skipped (already exists): {skipped_count}")
    print(f"  - Not Found / Future: {not_found_count}")
    print(f"  - Errors: {error_count}")
    print(f"  - Total Time: {elapsed:.2f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
