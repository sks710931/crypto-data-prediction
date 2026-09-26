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

from .config import FEATURE_COLUMNS, TIMEFRAME_SECONDS


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
    alignment_errors: int = 0
    close_time_errors: int = 0
    interval_errors: int = 0
    inf_counts: dict[str, int] = field(default_factory=dict)
    synthetic_rows: int = 0
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
            f" Alignment Errors:     {self.alignment_errors:,}",
            f" Candle Duration Errors:{self.close_time_errors:,}",
            f" Short Interval Errors: {self.interval_errors:,}",
            f" Infinite Values:       {sum(self.inf_counts.values()):,}",
            f" Synthetic Rows:        {self.synthetic_rows:,}",
            f" Gaps Detected:        {len(self.gaps):,}",
            f" Unavailable Clock Slots:{self.total_missing_candles:,}",
            f" Overall Status:       {'PASSED (historical gaps documented)' if self.is_valid and self.gaps else 'PASSED [OK]' if self.is_valid else 'FAILED [CRITICAL ERRORS]'}",
            "=" * 68,
        ]
        if self.gaps:
            lines.append("\nTop 5 Unavailable Clock Intervals (cause not inferred):")
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
    required = {*price_cols, volume_col, time_col, "close_time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")

    # 1. Null / NaN Counts
    null_dict = df.isnull().sum().to_dict()
    report.null_counts = {k: int(v) for k, v in null_dict.items() if v > 0}
    if sum(report.null_counts.values()) > 0:
        report.is_valid = False
    numeric = df.select_dtypes(include=[np.number])
    report.inf_counts = {col: int(np.isinf(numeric[col].to_numpy(dtype="float64", na_value=np.nan)).sum()) for col in numeric}
    report.inf_counts = {col: count for col, count in report.inf_counts.items() if count}
    if report.inf_counts:
        report.is_valid = False
    if "is_synthetic" in df.columns:
        report.synthetic_rows = int(df["is_synthetic"].astype(bool).sum())
        if report.synthetic_rows:
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
        excessive_taker = (taker_vol < 0) | (taker_vol > (volumes + 1e-5))
        report.volume_errors += int(np.sum(excessive_taker))
    if "quote_asset_volume" in df.columns:
        quote = df.quote_asset_volume.to_numpy()
        report.volume_errors += int(np.sum(quote < 0))
        if "taker_buy_quote_asset_volume" in df.columns:
            taker_quote = df.taker_buy_quote_asset_volume.to_numpy()
            report.volume_errors += int(np.sum((taker_quote < 0) | (taker_quote > quote + 1e-5)))
    if "number_of_trades" in df.columns:
        report.volume_errors += int(np.sum(df.number_of_trades.to_numpy() < 0))

    if report.volume_errors > 0:
        report.is_valid = False

    # 4. Timestamp Continuity & Gaps
    sample_ts = int(times[0])
    is_ms = sample_ts >= 1_000_000_000_000
    if interval not in TIMEFRAME_SECONDS:
        raise ValueError(f"Unsupported interval: {interval}")
    step_sec = TIMEFRAME_SECONDS[interval]
    expected_step = step_sec * 1000 if is_ms else step_sec
    time_diffs = np.diff(times)

    duplicates = int(np.sum(time_diffs == 0))
    out_of_order = int(np.sum(time_diffs < 0))
    report.timestamp_duplicates = duplicates
    report.out_of_order_errors = out_of_order
    report.alignment_errors = int(np.sum(times % expected_step != 0))
    report.interval_errors = int(np.sum((time_diffs > 0) & (time_diffs < expected_step)))
    if "close_time" in df.columns:
        report.close_time_errors = int(np.sum(df.close_time.to_numpy() - times != expected_step - 1))

    if duplicates > 0 or out_of_order > 0 or report.alignment_errors or report.close_time_errors or report.interval_errors:
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
    critical_null_counts: dict[str, int] = field(default_factory=dict)
    acceptable_null_counts: dict[str, int] = field(default_factory=dict)
    alignment_errors: int = 0
    close_time_errors: int = 0
    interval_errors: int = 0
    unavailable_labels: int = 0
    feature_discontinuities: int = 0
    warmup_rows_excluded: int = 0
    missing_feature_columns: list[str] = field(default_factory=list)
    duplicate_columns: int = 0
    is_valid: bool = True

    def summary(self) -> str:
        lines = [
            "=" * 68,
            "        POST-FEATURE VALIDATION & INTEGRITY REPORT",
            "=" * 68,
            f" Total Candles:          {self.total_rows:,}",
            f" Total Columns:          {self.total_columns:,}",
            f" Date Range:             {self.start_time_utc} to {self.end_time_utc}",
            f" Unavailable Clock Slots: {self.total_missing_candles:,} ({len(self.gaps):,} source gap(s))",
            f" Feature Discontinuities:{self.feature_discontinuities:,}",
            f" Excluded Warmup Rows:   {self.warmup_rows_excluded:,}",
            f" Duplicate Timestamps:   {self.timestamp_duplicates:,}",
            f" Out of Order Errors:    {self.out_of_order_errors:,}",
            f" Alignment Errors:       {self.alignment_errors:,}",
            f" Candle Duration Errors: {self.close_time_errors:,}",
            f" Short Interval Errors:  {self.interval_errors:,}",
            f" Critical NaN Errors:    {sum(self.critical_null_counts.values()):,}",
            f" Defined-Value NaNs:     {sum(self.acceptable_null_counts.values()):,}",
            f" Unavailable Labels:     {self.unavailable_labels:,}",
            f" Total Inf Errors:       {self.total_infs:,}",
            f" Indicator Anomalies:    {self.total_indicator_errors:,}",
            f" Missing Feature Columns:{len(self.missing_feature_columns):,}",
            f" Duplicate Columns:     {self.duplicate_columns:,}",
            f" Overall Status:         {'PASSED (historical gaps documented)' if self.is_valid and self.gaps else 'PASSED [OK]' if self.is_valid else 'FAILED [CRITICAL ERRORS]'}",
            "=" * 68,
        ]
        if self.null_counts:
            lines.append("\nColumns with NaN Values (critical / mathematically undefined / unavailable target):")
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
            lines.append("\nTop 5 Unavailable Clock Intervals (source data and exclusions):")
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
    source_report: ValidationReport | None = None,
    warmup_rows_excluded: int = 0,
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
    report.warmup_rows_excluded = warmup_rows_excluded
    report.missing_feature_columns = sorted(set(FEATURE_COLUMNS + ["target_next_up"]) - set(df.columns))
    report.duplicate_columns = int(df.columns.duplicated().sum())
    if report.missing_feature_columns or report.duplicate_columns:
        report.is_valid = False

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
    if interval not in TIMEFRAME_SECONDS:
        raise ValueError(f"Unsupported interval: {interval}")
    step_sec = TIMEFRAME_SECONDS[interval]
    expected_step = step_sec * 1000 if is_ms else step_sec

    time_diffs = np.diff(times)
    report.timestamp_duplicates = int(np.sum(time_diffs == 0))
    report.out_of_order_errors = int(np.sum(time_diffs < 0))
    report.alignment_errors = int(np.sum(times % expected_step != 0))
    report.interval_errors = int(np.sum((time_diffs > 0) & (time_diffs < expected_step)))
    if "close_time" in df.columns:
        report.close_time_errors = int(np.sum(df.close_time.to_numpy() - times != expected_step - 1))

    if report.timestamp_duplicates > 0 or report.out_of_order_errors > 0 or report.alignment_errors or report.close_time_errors or report.interval_errors:
        report.is_valid = False

    gap_indices = np.where(time_diffs > expected_step)[0]
    report.feature_discontinuities = len(gap_indices)
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
    if source_report is not None:
        report.gaps = source_report.gaps.copy()
        report.total_missing_candles = source_report.total_missing_candles

    # -------------------------------------------------------------
    # 3. NaNs and Infinite Values
    # -------------------------------------------------------------
    null_dict = df.isnull().sum().to_dict()
    report.null_counts = {k: int(v) for k, v in null_dict.items() if v > 0}
    report.total_nans = sum(report.null_counts.values())

    if "target_next_up" in df:
        report.unavailable_labels = int(df.target_next_up.isna().sum())
    segments = (df[time_col].diff() != expected_step).cumsum()

    def rolling_denominator(column: str, window: int, op: str) -> pd.Series:
        return df.groupby(segments)[column].transform(lambda s: getattr(s.rolling(window), op)())

    for col, count in report.null_counts.items():
        if col == "target_next_up":
            continue
        if col == "williams_r_14":
            # %R is undefined when its 14-bar high-low denominator is zero.
            highest = rolling_denominator("high", 14, "max")
            lowest = rolling_denominator("low", 14, "min")
            acceptable = df[col].isna() & (highest == lowest)
        elif col == "rvol_20":
            acceptable = df[col].isna() & (rolling_denominator("volume", 20, "mean") == 0)
        elif col == "volume_zscore_20":
            acceptable = df[col].isna() & (rolling_denominator("volume", 20, "std") == 0)
        elif col in {"vwap_20", "dist_vwap_20_pct", "vwap_60", "dist_vwap_60_pct"}:
            window = 20 if "20" in col else 60
            acceptable = df[col].isna() & (rolling_denominator("volume", window, "sum") == 0)
        else:
            acceptable = pd.Series(False, index=df.index)
        if acceptable.any():
            report.acceptable_null_counts[col] = int(acceptable.sum())
        if count > int(acceptable.sum()):
            report.critical_null_counts[col] = count - int(acceptable.sum())

    num_df = df.select_dtypes(include=[np.number])
    inf_dict = {col: int(np.isinf(num_df[col].to_numpy(dtype="float64", na_value=np.nan)).sum()) for col in num_df}
    report.inf_counts = {k: int(v) for k, v in inf_dict.items() if v > 0}
    report.total_infs = sum(report.inf_counts.values())

    if report.critical_null_counts or report.total_infs > 0:
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
        if close is not None:
            highest = rolling_denominator("high", 14, "max")
            lowest = rolling_denominator("low", 14, "min")
            expected = -100 * (highest - close) / (highest - lowest)
            checked = (highest > lowest) & df["williams_r_14"].notna()
            record_anomaly("williams_r_14_formula", ((df["williams_r_14"] - expected).abs() > tolerance)[checked].sum())

    if close is not None:
        for period in (5, 10, 20):
            name = f"roc_{period}"
            if name in df.columns:
                previous = df.groupby(segments)[price_col].shift(period)
                expected = (close / previous - 1) * 100
                checked = previous.notna() & df[name].notna()
                record_anomaly(f"{name}_formula", ((df[name] - expected).abs() > tolerance)[checked].sum())

    if "target_next_up" in df.columns:
        inv_target = (~df["target_next_up"].dropna().isin([0, 1])).sum()
        record_anomaly("target_next_up_binary", inv_target)

    # F. Volatility Indicators
    if "atr_14" in df.columns:
        inv_atr = (df["atr_14"] < 0).sum()
        record_anomaly("atr_14_non_negative", inv_atr)
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
        inv_bw = (df["bb_bandwidth_20_2"] < 0).sum()
        record_anomaly("bb_bandwidth_non_negative", inv_bw)

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
            if all(c in df.columns for c in ("high", "low", "close", "volume")):
                typical_volume = (df.high + df.low + df.close) / 3 * df.volume
                pv_sum = typical_volume.groupby(segments).transform(lambda s: s.rolling(v_win).sum())
                volume_sum = df.volume.groupby(segments).transform(lambda s: s.rolling(v_win).sum())
                expected_vwap = pv_sum / volume_sum.where(volume_sum != 0)
                checked = expected_vwap.notna() & df[vwap_col].notna()
                record_anomaly(f"{vwap_col}_formula", ((df[vwap_col] - expected_vwap).abs() > tolerance)[checked].sum())

    return report

