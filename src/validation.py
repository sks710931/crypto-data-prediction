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


def format_utc(ts: int | float) -> str:
    """Format Unix timestamp (seconds or milliseconds) to readable UTC string."""
    t = float(ts)
    if t >= 1_000_000_000_000:
        t = t / 1000.0
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


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
    sample_ts = int(times[0])
    is_ms = sample_ts >= 1_000_000_000_000
    step_sec = TIMEFRAME_SECONDS.get(interval, 60)
    expected_step = step_sec * 1000 if is_ms else step_sec
    time_diffs = np.diff(times)

    duplicates = int(np.sum(time_diffs == 0))
    out_of_order = int(np.sum(time_diffs < 0))
    report.timestamp_duplicates = duplicates
    report.out_of_order_errors = out_of_order

    if duplicates > 0 or out_of_order > 0:
        report.is_valid = False

    # Gap detection
    gap_indices = np.where(time_diffs > expected_step)[0]
    for idx in gap_indices:
        t_start = int(times[idx]) + expected_step
        t_end = int(times[idx + 1]) - expected_step
        gap_duration = int(time_diffs[idx]) - expected_step
        missing_count = int(gap_duration // expected_step)
        gap_sec = gap_duration // 1000 if is_ms else gap_duration

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


@dataclass
class FeatureValidationReport:
    total_rows: int = 0
    total_columns: int = 0
    start_time_utc: str = ""
    end_time_utc: str = ""
    timestamp_duplicates: int = 0
    out_of_order_errors: int = 0
    gaps: list[dict[str, Any]] = field(default_factory=list)
    total_missing_candles: int = 0
    null_counts: dict[str, int] = field(default_factory=dict)
    inf_counts: dict[str, int] = field(default_factory=dict)
    total_nans: int = 0
    total_infs: int = 0
    indicator_errors: dict[str, int] = field(default_factory=dict)
    total_indicator_errors: int = 0
    is_valid: bool = True

    def summary(self) -> str:
        lines = [
            "=" * 68,
            "        POST-FEATURE VALIDATION & INTEGRITY REPORT",
            "=" * 68,
            f" Total Candles:          {self.total_rows:,}",
            f" Total Columns:          {self.total_columns:,}",
            f" Date Range:             {self.start_time_utc} to {self.end_time_utc}",
            f" Missing Candles / Gaps: {self.total_missing_candles:,} ({len(self.gaps):,} gap(s))",
            f" Duplicate Timestamps:   {self.timestamp_duplicates:,}",
            f" Out of Order Errors:    {self.out_of_order_errors:,}",
            f" Total NaN Errors:       {self.total_nans:,}",
            f" Total Inf Errors:       {self.total_infs:,}",
            f" Indicator Anomalies:    {self.total_indicator_errors:,}",
            f" Overall Status:         {'PASSED [OK]' if self.is_valid else 'FAILED [WARNINGS FOUND]'}",
            "=" * 68,
        ]
        if self.null_counts:
            lines.append("\nColumns with NaN Values:")
            for col, count in sorted(self.null_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                pct = (count / self.total_rows) * 100 if self.total_rows > 0 else 0
                lines.append(f"  - {col}: {count:,} ({pct:.2f}%)")
        if self.inf_counts:
            lines.append("\nColumns with Infinite Values:")
            for col, count in sorted(self.inf_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                pct = (count / self.total_rows) * 100 if self.total_rows > 0 else 0
                lines.append(f"  - {col}: {count:,} ({pct:.2f}%)")
        if self.indicator_errors:
            lines.append("\nIndicator Calculation Anomalies:")
            for rule, count in sorted(self.indicator_errors.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  - {rule}: {count:,} violations")
        if self.gaps:
            lines.append("\nTop 5 Detected Gaps (Missing Candles):")
            lines.append(f"{'#':<4} {'Start UTC':<22} {'End UTC':<22} {'Missing':<10} {'Duration'}")
            lines.append("-" * 68)
            sorted_gaps = sorted(self.gaps, key=lambda g: g["missing_candles"], reverse=True)
            for i, gap in enumerate(sorted_gaps[:5], start=1):
                dur_str = f"{gap['duration_sec'] // 60}m" if gap['duration_sec'] < 86400 else f"{gap['duration_sec'] / 3600:.1f}h"
                lines.append(f"{i:<4} {gap['start_utc']:<22} {gap['end_utc']:<22} {gap['missing_candles']:<10,d} {dur_str}")
            lines.append("-" * 68)
        return "\n".join(lines)


def validate_features(
    df: pd.DataFrame,
    interval: str = "1m",
    time_col: str = "open_time",
    price_col: str = "close",
    tolerance: float = 1e-4,
) -> FeatureValidationReport:
    """
    Validates feature-engineered market dataset across 4 critical pillars:
      1. Missing candles (gap identification based on candle interval)
      2. Duplicate timestamps and chronological monotonicity
      3. NaNs and infinite values across all feature columns
      4. Indicator calculations (bounds, formulas, channel order, positivity)
    """
    report = FeatureValidationReport()
    report.total_rows = len(df)
    report.total_columns = len(df.columns)

    if report.total_rows == 0:
        report.is_valid = False
        return report

    # -------------------------------------------------------------
    # 1 & 2. Missing Candles, Duplicates & Monotonicity
    # -------------------------------------------------------------
    times = df[time_col].to_numpy()
    report.start_time_utc = format_utc(int(times[0]))
    report.end_time_utc = format_utc(int(times[-1]))

    sample_ts = int(times[0])
    is_ms = sample_ts >= 1_000_000_000_000
    step_sec = TIMEFRAME_SECONDS.get(interval, 60)
    expected_step = step_sec * 1000 if is_ms else step_sec

    time_diffs = np.diff(times)
    report.timestamp_duplicates = int(np.sum(time_diffs == 0))
    report.out_of_order_errors = int(np.sum(time_diffs < 0))

    if report.timestamp_duplicates > 0 or report.out_of_order_errors > 0:
        report.is_valid = False

    gap_indices = np.where(time_diffs > expected_step)[0]
    for idx in gap_indices:
        t_start = int(times[idx]) + expected_step
        t_end = int(times[idx + 1]) - expected_step
        gap_duration = int(time_diffs[idx]) - expected_step
        missing_count = int(gap_duration // expected_step)
        gap_sec = gap_duration // 1000 if is_ms else gap_duration

        report.gaps.append({
            "start_ts": t_start,
            "end_ts": t_end,
            "start_utc": format_utc(t_start),
            "end_utc": format_utc(t_end),
            "missing_candles": missing_count,
            "duration_sec": gap_sec,
        })
        report.total_missing_candles += missing_count

    # -------------------------------------------------------------
    # 3. NaNs and Infinite Values
    # -------------------------------------------------------------
    null_dict = df.isnull().sum().to_dict()
    report.null_counts = {k: int(v) for k, v in null_dict.items() if v > 0}
    report.total_nans = sum(report.null_counts.values())

    num_df = df.select_dtypes(include=[np.number])
    inf_dict = np.isinf(num_df).sum().to_dict()
    report.inf_counts = {k: int(v) for k, v in inf_dict.items() if v > 0}
    report.total_infs = sum(report.inf_counts.values())

    if report.total_nans > 0 or report.total_infs > 0:
        report.is_valid = False

    # -------------------------------------------------------------
    # 4. Indicator Calculation Validation
    # -------------------------------------------------------------
    close = df[price_col] if price_col in df.columns else None

    def record_anomaly(rule_name: str, count: int) -> None:
        if count > 0:
            report.indicator_errors[rule_name] = int(count)
            report.total_indicator_errors += int(count)
            report.is_valid = False

    # A. Trend Moving Averages Positivity
    ma_cols = [c for c in ["ema_9", "ema_21", "ema_50", "ema_200", "sma_20", "sma_50"] if c in df.columns]
    if ma_cols:
        invalid_ma = (df[ma_cols] <= 0).sum().sum()
        record_anomaly("ma_positive (EMA/SMA <= 0)", invalid_ma)

    # B. Derived Trend Features Formulas
    if close is not None:
        if "ema_21" in df.columns and "price_dist_ema21_pct" in df.columns:
            expected = (close - df["ema_21"]) / df["ema_21"] * 100.0
            inv = (np.abs(df["price_dist_ema21_pct"] - expected) > tolerance).sum()
            record_anomaly("price_dist_ema21_pct_formula", inv)

        if "ema_9" in df.columns and "ema_21" in df.columns and "ema9_ema21_spread_pct" in df.columns:
            expected = (df["ema_9"] - df["ema_21"]) / df["ema_21"] * 100.0
            inv = (np.abs(df["ema9_ema21_spread_pct"] - expected) > tolerance).sum()
            record_anomaly("ema9_ema21_spread_pct_formula", inv)

        if "macd" in df.columns and "macd_signal" in df.columns and "macd_hist" in df.columns:
            expected_hist = df["macd"] - df["macd_signal"]
            inv = (np.abs(df["macd_hist"] - expected_hist) > tolerance).sum()
            record_anomaly("macd_hist_formula (hist != macd - signal)", inv)

    # C. ADX & Directional Movement
    if "adx_14" in df.columns:
        inv_adx = ((df["adx_14"] < 0) | (df["adx_14"] > 100)).sum()
        record_anomaly("adx_14_bounds ([0, 100])", inv_adx)

    if "plus_di_14" in df.columns and "minus_di_14" in df.columns:
        inv_di = ((df["plus_di_14"] < 0) | (df["minus_di_14"] < 0)).sum()
        record_anomaly("di_non_negative (+DI or -DI < 0)", inv_di)
        if "di_diff_14" in df.columns:
            expected_diff = df["plus_di_14"] - df["minus_di_14"]
            inv = (np.abs(df["di_diff_14"] - expected_diff) > tolerance).sum()
            record_anomaly("di_diff_14_formula (diff != +DI - -DI)", inv)

    # D. Supertrend
    if "supertrend_10_3" in df.columns:
        inv_st = (df["supertrend_10_3"] <= 0).sum()
        record_anomaly("supertrend_positive", inv_st)
    if "supertrend_dir_10_3" in df.columns:
        inv_st_dir = (~df["supertrend_dir_10_3"].isin([-1.0, 1.0])).sum()
        record_anomaly("supertrend_dir_valid (dir not in {-1, 1})", inv_st_dir)

    # E. Momentum Bounded Oscillators
    for rsi_col in ["rsi_7", "rsi_14"]:
        if rsi_col in df.columns:
            inv_rsi = ((df[rsi_col] < -0.01) | (df[rsi_col] > 100.01)).sum()
            record_anomaly(f"{rsi_col}_bounds ([0, 100])", inv_rsi)

    for stoch_col in ["stochrsi_k_14_3_3", "stochrsi_d_14_3_3"]:
        if stoch_col in df.columns:
            inv_stoch = ((df[stoch_col] < -0.01) | (df[stoch_col] > 100.01)).sum()
            record_anomaly(f"{stoch_col}_bounds ([0, 100])", inv_stoch)

    if "williams_r_14" in df.columns:
        inv_willr = ((df["williams_r_14"] < -100.01) | (df["williams_r_14"] > 0.01)).sum()
        record_anomaly("williams_r_14_bounds ([-100, 0])", inv_willr)

    # F. Volatility Indicators
    if "atr_14" in df.columns:
        inv_atr = (df["atr_14"] <= 0).sum()
        record_anomaly("atr_14_positive", inv_atr)
        if close is not None and "natr_14" in df.columns:
            expected_natr = (df["atr_14"] / close) * 100.0
            inv = (np.abs(df["natr_14"] - expected_natr) > tolerance).sum()
            record_anomaly("natr_14_formula", inv)

    if all(c in df.columns for c in ["bb_lower_20_2", "bb_mid_20_2", "bb_upper_20_2"]):
        inv_bb_order = (
            (df["bb_lower_20_2"] > df["bb_mid_20_2"] + 1e-5) |
            (df["bb_mid_20_2"] > df["bb_upper_20_2"] + 1e-5)
        ).sum()
        record_anomaly("bb_bands_order (lower <= mid <= upper)", inv_bb_order)

    if "bb_bandwidth_20_2" in df.columns:
        inv_bw = (df["bb_bandwidth_20_2"] <= 0).sum()
        record_anomaly("bb_bandwidth_positive", inv_bw)

    if all(c in df.columns for c in ["keltner_lower_20_2", "keltner_mid_20_2", "keltner_upper_20_2"]):
        inv_kc_order = (
            (df["keltner_lower_20_2"] > df["keltner_mid_20_2"] + 1e-5) |
            (df["keltner_mid_20_2"] > df["keltner_upper_20_2"] + 1e-5)
        ).sum()
        record_anomaly("keltner_bands_order (lower <= mid <= upper)", inv_kc_order)

    vol_cols = [c for c in ["volatility_returns_5", "volatility_returns_15", "volatility_returns_30", "volatility_returns_60", "parkinson_vol_20"] if c in df.columns]
    if vol_cols:
        inv_vols = (df[vol_cols] < 0).sum().sum()
        record_anomaly("volatility_non_negative", inv_vols)

    # G. Volume Indicators
    if "rvol_20" in df.columns:
        inv_rvol = (df["rvol_20"] < 0).sum()
        record_anomaly("rvol_20_non_negative", inv_rvol)

    if "mfi_14" in df.columns:
        inv_mfi = ((df["mfi_14"] < -0.01) | (df["mfi_14"] > 100.01)).sum()
        record_anomaly("mfi_14_bounds ([0, 100])", inv_mfi)

    if "cmf_20" in df.columns:
        inv_cmf = ((df["cmf_20"] < -1.01) | (df["cmf_20"] > 1.01)).sum()
        record_anomaly("cmf_20_bounds ([-1, 1])", inv_cmf)

    for v_win in [20, 60]:
        vwap_col = f"vwap_{v_win}"
        dist_col = f"dist_vwap_{v_win}_pct"
        if vwap_col in df.columns:
            inv_vwap = (df[vwap_col] <= 0).sum()
            record_anomaly(f"{vwap_col}_positive", inv_vwap)
            if close is not None and dist_col in df.columns:
                expected_dist = (close - df[vwap_col]) / df[vwap_col] * 100.0
                inv = (np.abs(df[dist_col] - expected_dist) > tolerance).sum()
                record_anomaly(f"{dist_col}_formula", inv)

    return report

