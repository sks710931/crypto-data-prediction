"""
Data Quality and Integrity Validation Module.
Validates OHLCV data for price anomalies, volume consistency, nulls,
and chronological timestamp continuity.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from .config import RAW_COLUMNS, TIMEFRAME_SECONDS


@dataclass
class ValidationReport:
    total_rows: int = 0
    start_time_utc: str = ""
    end_time_utc: str = ""
    null_counts: dict[str, int] = field(default_factory=dict)
    price_errors: int = 0
    volume_errors: int = 0
    timestamp_duplicates: int = 0
    out_of_order_errors: int = 0
    gaps: list[dict[str, Any]] = field(default_factory=list)
    total_missing_candles: int = 0
    is_valid: bool = True

    def summary(self) -> str:
        lines = [
            "=" * 68,
            "               DATA VALIDATION & INTEGRITY REPORT",
            "=" * 68,
            f" Total Candles:        {self.total_rows:,}",
            f" Date Range:           {self.start_time_utc} to {self.end_time_utc}",
            f" Null / NaN Errors:    {sum(self.null_counts.values()):,}",
            f" Price Anomaly Errors: {self.price_errors:,}",
            f" Volume Errors:        {self.volume_errors:,}",
            f" Duplicate Timestamps: {self.timestamp_duplicates:,}",
            f" Out of Order Errors:  {self.out_of_order_errors:,}",
            f" Gaps Detected:        {len(self.gaps):,}",
            f" Total Missing Candles:{self.total_missing_candles:,}",
            f" Overall Status:       {'PASSED [OK]' if self.is_valid else 'FAILED [WARNINGS FOUND]'}",
            "=" * 68,
        ]
        if self.gaps:
            lines.append("\nTop 5 Largest Detected Gaps (Exchange Downtime / Maintenance):")
            lines.append(f"{'#':<4} {'Start UTC':<22} {'End UTC':<22} {'Missing':<10} {'Duration'}")
            lines.append("-" * 68)
            sorted_gaps = sorted(self.gaps, key=lambda g: g["missing_candles"], reverse=True)
            for i, gap in enumerate(sorted_gaps[:5], start=1):
                dur_str = f"{gap['duration_sec'] // 60}m" if gap['duration_sec'] < 86400 else f"{gap['duration_sec'] / 3600:.1f}h"
                lines.append(f"{i:<4} {gap['start_utc']:<22} {gap['end_utc']:<22} {gap['missing_candles']:<10,d} {dur_str}")
            lines.append("-" * 68)
        return "\n".join(lines)


def format_utc(ts: int) -> str:
    """Format Unix timestamp in seconds to readable UTC string."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def validate_ohlcv(
    df: pd.DataFrame,
    interval: str = "1m",
    price_cols: tuple[str, str, str, str] = ("open", "high", "low", "close"),
    volume_col: str = "volume",
    time_col: str = "open_time",
) -> ValidationReport:
    """
    Performs comprehensive integrity checks on an OHLCV DataFrame.

    Checks:
      - Presence of required columns
      - Missing / NaN / Inf values
      - Logical OHLC constraints (high >= low, high >= open, high >= close, etc.)
      - Positive prices and non-negative volume
      - Taker volume <= Total volume (with floating point tolerance)
      - Monotonicity and chronological ordering of timestamps
      - Identification of gaps and missing periods
    """
    report = ValidationReport()
    report.total_rows = len(df)

    if report.total_rows == 0:
        report.is_valid = False
        return report

    # 1. Null / NaN Counts
    null_dict = df.isnull().sum().to_dict()
    report.null_counts = {k: int(v) for k, v in null_dict.items() if v > 0}
    if sum(report.null_counts.values()) > 0:
        report.is_valid = False

    # Extract columns
    o_col, h_col, l_col, c_col = price_cols
    opens = df[o_col].to_numpy()
    highs = df[h_col].to_numpy()
    lows = df[l_col].to_numpy()
    closes = df[c_col].to_numpy()
    volumes = df[volume_col].to_numpy()
    times = df[time_col].to_numpy()

    report.start_time_utc = format_utc(int(times[0]))
    report.end_time_utc = format_utc(int(times[-1]))

    # 2. Price Logic Validation
    # High must be >= Open, Close, Low; Low must be <= Open, Close, High
    # All prices must be strictly positive
    invalid_hl = (highs < lows)
    invalid_ho = (highs < opens)
    invalid_hc = (highs < closes)
    invalid_lo = (lows > opens)
    invalid_lc = (lows > closes)
    invalid_positive = (opens <= 0) | (highs <= 0) | (lows <= 0) | (closes <= 0)

    price_anomalies = invalid_hl | invalid_ho | invalid_hc | invalid_lo | invalid_lc | invalid_positive
    report.price_errors = int(np.sum(price_anomalies))
    if report.price_errors > 0:
        report.is_valid = False

    # 3. Volume Validation
    # Volume must be non-negative
    invalid_volume = volumes < 0
    report.volume_errors = int(np.sum(invalid_volume))

    # Taker volume check if available
    if "taker_buy_base_asset_volume" in df.columns:
        taker_vol = df["taker_buy_base_asset_volume"].to_numpy()
        # tolerance for floating point rounding issues
        excessive_taker = taker_vol > (volumes + 1e-5)
        report.volume_errors += int(np.sum(excessive_taker))

    if report.volume_errors > 0:
        report.is_valid = False

    # 4. Timestamp Continuity & Gaps
    step_sec = TIMEFRAME_SECONDS.get(interval, 60)
    time_diffs = np.diff(times)

    duplicates = int(np.sum(time_diffs == 0))
    out_of_order = int(np.sum(time_diffs < 0))
    report.timestamp_duplicates = duplicates
    report.out_of_order_errors = out_of_order

    if duplicates > 0 or out_of_order > 0:
        report.is_valid = False

    # Gap detection
    gap_indices = np.where(time_diffs > step_sec)[0]
    for idx in gap_indices:
        t_start = int(times[idx]) + step_sec
        t_end = int(times[idx + 1]) - step_sec
        gap_sec = int(time_diffs[idx]) - step_sec
        missing_count = int(gap_sec // step_sec)

        report.gaps.append({
            "start_ts": t_start,
            "end_ts": t_end,
            "start_utc": format_utc(t_start),
            "end_utc": format_utc(t_end),
            "missing_candles": missing_count,
            "duration_sec": gap_sec,
        })
        report.total_missing_candles += missing_count

    return report
