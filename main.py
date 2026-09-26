#!/usr/bin/env python3
"""
Streamlined Market Data Pipeline Orchestrator.

Usage:
  python main.py --pair BTCUSDT
  python main.py --pair BTCUSDT --interval 5m

Pipeline Steps:
  1. Convert merged CSV to optimized 1m Parquet (if not already done).
  2. Select complete official 5m archives when present; otherwise strictly resample 1m.
  3. Validate OHLCV data integrity & continuity.
  4. Compute 53 technical indicators, reset at gaps, and label the next candle.
  5. Validate post-feature dataset (missing candles, duplicate timestamps, NaNs, indicator calculations).
  6. Freeze source provenance and audit point-in-time labels, features, and training splits.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import BASE_DIR, PARQUET_DIR, PROCESSED_DIR, TIMEFRAME_SECONDS, TRAIN_DATA_DIR
from src.indicators import compute_all_indicators
from src.official import archives_cover_range, load_monthly_klines, separate_complete_candles
from src.resampling import resample_ohlcv
from src.storage import csv_to_parquet, load_parquet
from src.training_readiness import build_training_readiness
from src.validation import validate_features, validate_ohlcv


def _supports_color(mode: str = "auto") -> bool:
    """Use ANSI colors on interactive terminals, with an explicit override."""
    if mode == "never":
        return False
    if mode == "auto" and (os.environ.get("NO_COLOR") is not None or os.environ.get("TERM") == "dumb"):
        return False
    if mode == "auto" and not sys.stdout.isatty():
        return False
    if os.name == "nt" and sys.stdout.isatty():
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        console_mode = ctypes.c_ulong()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(console_mode)):
            return mode == "always"
        if not kernel32.SetConsoleMode(handle, console_mode.value | 0x0004):  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
            return mode == "always"
    return True


COLOR_ENABLED = _supports_color()


def _color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if COLOR_ENABLED else text


def _print_validation_summary(summary: str, is_valid: bool) -> None:
    for line in summary.splitlines():
        if "Overall Status:" in line:
            code = "1;31" if not is_valid else "1;33" if "historical gaps" in line else "1;32"
        elif "VALIDATION & INTEGRITY REPORT" in line:
            code = "1;36"
        elif any(label in line for label in (
            "Gaps Detected:", "Unavailable Clock Slots:", "Feature Discontinuities:",
            "Defined-Value NaNs:", "Unavailable Labels:", "Top 5 Unavailable",
        )):
            code = "1;33"
        elif "Columns with NaN Values" in line:
            code = "1;33"
        else:
            code = ""
        print(_color(line, code) if code else line)


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
    parser.add_argument(
        "--source", choices=("auto", "1m", "5m"), default="auto",
        help="For 5m, prefer complete official 5m archives when available; 1m forces strict resampling",
    )
    parser.add_argument(
        "--color", choices=("auto", "always", "never"), default="auto",
        help="Colorize console output (default: auto; always overrides NO_COLOR)",
    )
    parser.add_argument("--train-end", default="2026-03-01",
                        help="First UTC date in validation (default: 2026-03-01)")
    parser.add_argument("--validation-end", default="2026-06-01",
                        help="First UTC date in test (default: 2026-06-01)")
    return parser.parse_args()


def main():
    global COLOR_ENABLED
    args = parse_args()
    COLOR_ENABLED = _supports_color(args.color)
    pair = args.pair.strip().upper()
    interval = args.interval.strip().lower()
    if interval not in TIMEFRAME_SECONDS:
        raise ValueError(f"Unsupported interval: {interval}")
    if args.source == "5m" and interval != "5m":
        raise ValueError("--source 5m requires --interval 5m")

    input_csv = PROCESSED_DIR / f"{pair}.csv"
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    parquet_1m = PARQUET_DIR / f"{pair}_1m.parquet"

    total_start = time.time()

    print("=" * 68)
    print(_color("        STREAMLINED CRYPTO DATA PIPELINE ORCHESTRATOR", "1;36"))
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
        print(_color("\n[1/6] Converting CSV to 1m Parquet...", "1;36"))
        csv_to_parquet(csv_path=input_csv, parquet_path=parquet_1m)
    else:
        print(_color(f"\n[1/6] Found base Parquet: {parquet_1m.name} ({parquet_1m.stat().st_size / (1024 * 1024):.1f} MB)", "1;36"))

    # -------------------------------------------------------------
    # Step 2: Resampling (At Start if interval != 1m)
    # -------------------------------------------------------------
    excluded = None
    source_name = "1m parquet"
    source_paths = [parquet_1m]
    if interval != "1m":
        parquet_interval = PARQUET_DIR / f"{pair}_{interval}.parquet"
        df_1m = load_parquet(parquet_1m)
        raw_dir = BASE_DIR / "raw" / pair
        have_direct = interval == "5m" and archives_cover_range(
            raw_dir, pair, interval, int(df_1m.open_time.iloc[0]), int(df_1m.open_time.iloc[-1])
        )
        if args.source == "5m" and not have_direct:
            raise FileNotFoundError("Complete official BTCUSDT 5m monthly archives are required for --source 5m")
        if have_direct and args.source != "1m":
            print(_color("\n[2/6] Loading exchange-provided 5m candles...", "1;36"))
            direct = load_monthly_klines(raw_dir, pair, interval, int(df_1m.open_time.iloc[0]), int(df_1m.open_time.iloc[-1]))
            working_df, excluded = separate_complete_candles(direct, TIMEFRAME_SECONDS[interval])
            source_name = "official 5m monthly archives"
            months = pd.period_range(
                pd.Timestamp(int(df_1m.open_time.iloc[0]), unit="s", tz="UTC").tz_localize(None).to_period("M"),
                pd.Timestamp(int(df_1m.open_time.iloc[-1]), unit="s", tz="UTC").tz_localize(None).to_period("M"),
                freq="M",
            )
            source_paths.extend(raw_dir / f"{pair}-{interval}-{month}.zip" for month in months)
        else:
            print(_color(f"\n[2/6] Strictly resampling 1m data to {interval}...", "1;36"))
            working_df, excluded = resample_ohlcv(df_1m, target_timeframe=interval, return_excluded=True)
            source_name = "strict 1m resample"
        print(f"  -> Retained {len(working_df):,} complete candles; excluded {len(excluded):,} source bars/bins.")
    else:
        print(_color("\n[2/6] Interval is 1m. Resampling skipped.", "1;36"))
        working_df = load_parquet(parquet_1m)

    # -------------------------------------------------------------
    # Step 3: Data Quality & Integrity Validation
    # -------------------------------------------------------------
    print(_color(f"\n[3/6] Validating {pair} ({interval}) OHLCV data...", "1;36"))
    val_start = time.time()
    report = validate_ohlcv(working_df, interval=interval)
    _print_validation_summary(report.summary(), report.is_valid)
    print(f"OHLCV validation completed in {time.time() - val_start:.2f}s.")
    if not report.is_valid:
        raise ValueError("Critical OHLCV integrity errors; feature output was not written")
    if interval != "1m":
        temp_interval = parquet_interval.with_suffix(".parquet.tmp")
        working_df.to_parquet(temp_interval, compression="snappy", engine="pyarrow", index=False)
        temp_interval.replace(parquet_interval)

    # -------------------------------------------------------------
    # Step 4: Technical Indicators & Feature Addition (Warmup Dropped)
    # -------------------------------------------------------------
    print(_color(f"\n[4/6] Computing Technical Indicators for {pair} ({interval})...", "1;36"))
    features_df = compute_all_indicators(
        df=working_df,
        drop_warmup=True,
        warmup_period=200,
        interval=interval,
    )

    # -------------------------------------------------------------
    # Step 5: Post-Feature Engineering Validation
    # (Missing candles, duplicate timestamps, NaNs, indicator calculations)
    # -------------------------------------------------------------
    print(_color(f"\n[5/6] Validating {pair} ({interval}) Engineered Features & Indicators...", "1;36"))
    feat_val_start = time.time()
    feature_report = validate_features(
        features_df, interval=interval, source_report=report,
        warmup_rows_excluded=features_df.attrs["warmup_rows_dropped"],
    )
    _print_validation_summary(feature_report.summary(), feature_report.is_valid)
    print(f"Feature validation completed in {time.time() - feat_val_start:.2f}s.")
    if not feature_report.is_valid:
        raise ValueError("Critical feature integrity errors; feature output was not written")

    trainable_labels = int(features_df.target_next_up.notna().sum())
    up_labels = int((features_df.target_next_up == 1).sum())
    down_labels = int((features_df.target_next_up == 0).sum())
    up_pct = 100 * up_labels / trainable_labels if trainable_labels else 0.0
    down_pct = 100 * down_labels / trainable_labels if trainable_labels else 0.0

    output_features_parquet = TRAIN_DATA_DIR / f"{pair}_{interval}_features.parquet"
    print(f"\nWriting features to: {output_features_parquet.name}...")
    temp_features = output_features_parquet.with_suffix(".parquet.tmp")
    features_df.to_parquet(temp_features, compression="snappy", engine="pyarrow", index=False)
    temp_features.replace(output_features_parquet)
    audit_path = TRAIN_DATA_DIR / f"{pair}_{interval}_quality.json"
    retained = working_df.open_time.isin(features_df.open_time)
    next_is_contiguous = working_df.open_time.shift(-1) - working_df.open_time == TIMEFRAME_SECONDS[interval]
    next_is_flat = working_df.open.shift(-1) == working_df.close.shift(-1)
    audit = {
        "source": source_name,
        "source_candles_retained": len(working_df),
        "excluded_source_rows_or_bins": [] if excluded is None else excluded.to_dict(orient="records"),
        "historical_gaps": report.gaps,
        "warmup_rows_excluded": features_df.attrs["warmup_rows_dropped"],
        "segment_count": features_df.attrs["segment_count"],
        "feature_rows": len(features_df),
        "mathematically_undefined_features": feature_report.acceptable_null_counts,
        "unavailable_next_candle_labels": feature_report.unavailable_labels,
        "trainable_label_distribution": {
            "up": {"rows": up_labels, "percent": up_pct},
            "down": {"rows": down_labels, "percent": down_pct},
        },
        "labels_unavailable_at_gap_or_end": int((retained & ~next_is_contiguous).sum()),
        "labels_unavailable_for_flat_next_candle": int((retained & next_is_contiguous & next_is_flat).sum()),
    }
    temp_audit = audit_path.with_suffix(".json.tmp")
    temp_audit.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    temp_audit.replace(audit_path)
    print(_color(f"\n[6/6] Checking {pair} ({interval}) training readiness...", "1;36"))
    code_paths = [BASE_DIR / name for name in (
        "main.py", "merger.py", "src/config.py", "src/indicators.py", "src/official.py",
        "src/resampling.py", "src/storage.py", "src/training_readiness.py", "src/validation.py",
    )]
    readiness = build_training_readiness(
        features=features_df, source=working_df, interval=interval,
        train_end=args.train_end, validation_end=args.validation_end,
        feature_path=output_features_parquet, source_paths=source_paths,
        code_paths=code_paths, output_dir=TRAIN_DATA_DIR, pair=pair,
    )
    if not readiness["ready"]:
        print(_color(f" Training readiness requires review: {readiness['issues']}", "1;31"))
        raise ValueError(f"Training readiness failed; see {readiness['artifacts']['readiness_markdown']}")
    size_mb = output_features_parquet.stat().st_size / (1024 * 1024)

    # -------------------------------------------------------------
    # Pipeline Summary
    # -------------------------------------------------------------
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 68)
    print(_color("                 PIPELINE EXECUTION COMPLETE", "1;36"))
    print("=" * 68)
    print(f" Total Elapsed Time:   {total_elapsed:.2f}s")
    print(f" Output Dataset:       {output_features_parquet.name}")
    print(f" Rows (Post-Warmup):   {len(features_df):,}")
    print(f" Columns:              {len(features_df.columns)} ({len(working_df.columns)} source + 53 indicators + target)")
    print(f" Source:               {source_name}")
    print(f" Excluded Source Bars: {0 if excluded is None else len(excluded):,}")
    print(f" Warmup Rows Excluded: {features_df.attrs['warmup_rows_dropped']:,}")
    print(_color(f" Trainable Labels:     {trainable_labels:,}", "1;36"))
    print(_color(f" UP Labels:            {up_labels:,} ({up_pct:.2f}% of trainable)", "1;32"))
    print(_color(f" DOWN Labels:          {down_labels:,} ({down_pct:.2f}% of trainable)", "1;31"))
    print(_color(f" OHLCV Status:         {'PASSED [OK]' if report.is_valid else 'FAILED [WARNINGS FOUND]'}", "1;32" if report.is_valid else "1;31"))
    print(_color(f" Feature Status:       {'PASSED [OK]' if feature_report.is_valid else 'FAILED [WARNINGS FOUND]'}", "1;32" if feature_report.is_valid else "1;31"))
    readiness_status = "PASSED WITH WARNINGS" if readiness["warnings"] else "PASSED [OK]"
    print(_color(f" Training Readiness:   {readiness_status}", "1;33" if readiness["warnings"] else "1;32"))
    for warning in readiness["warnings"]:
        print(_color(f"  - {warning}", "1;33"))
    for name, profile in readiness["profiles_by_split"].items():
        print(f" {name.title():<20}{profile['included_rows']:,} labeled rows")
    print(f" File Size:            {size_mb:.2f} MB")
    print(f" Saved Location:       {output_features_parquet.resolve()}")
    print(f" Quality Audit:        {audit_path.resolve()}")
    print(f" Readiness Report:     {readiness['artifacts']['readiness_markdown']}")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
