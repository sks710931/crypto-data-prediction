import io
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.indicators import compute_all_indicators
from src.indicators import add_volume_indicators
from src.official import load_monthly_klines, separate_complete_candles
from src.resampling import resample_ohlcv
from src.validation import validate_features, validate_ohlcv
from merger import clock_grid_exclusion_reason, get_sorted_zip_files, merge_and_validate, synthetic_gap_rows


def candles(times):
    n = len(times)
    close = 100 + np.arange(n) * 0.01 + np.sin(np.arange(n) / 7) * 0.1
    return pd.DataFrame({
        "open_time": times,
        "open": close - 0.02,
        "high": close + 0.1,
        "low": close - 0.1,
        "close": close,
        "volume": 1 + np.arange(n) % 13,
        "close_time": times + 299,
    })


class PipelineTests(unittest.TestCase):
    def test_resample_rejects_partial_and_shifted_minutes(self):
        t = np.array([0, 60, 120, 180, 240, 300, 360, 480, 540, 600, 660, 740, 780, 840])
        df = candles(t)
        df.close_time = df.open_time + 59
        result, excluded = resample_ohlcv(df, "5m", return_excluded=True)
        self.assertEqual(result.open_time.tolist(), [0])
        self.assertEqual(excluded.open_time.tolist(), [300, 600])
        self.assertEqual(result.volume.iloc[0], df.volume.iloc[:5].sum())

    def test_official_reader_normalizes_milliseconds_and_microseconds(self):
        base = 1735689600
        rows = [
            [base * 1000, 1, 2, 1, 2, 3, (base + 300) * 1000 - 1, 4, 5, 1, 2, 0],
            [(base + 300) * 10**6, 2, 3, 2, 3, 4, (base + 600) * 10**6 - 1, 5, 6, 2, 3, 0],
        ]
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as z:
            z.writestr("BTCUSDT-5m-2025-01.csv", "\n".join(",".join(map(str, r)) for r in rows))
        with patch("src.official.archives_cover_range", return_value=True), patch(
            "src.official.ZipFile", side_effect=lambda _: zipfile.ZipFile(io.BytesIO(buffer.getvalue()))
        ):
            result = load_monthly_klines(Path("unused"), "BTCUSDT", "5m", base, base + 300)
        self.assertEqual(result.open_time.tolist(), [base, base + 300])
        self.assertEqual(result.close_time.tolist(), [base + 299, base + 599])

    def test_source_filter_and_validator_detect_irregular_time(self):
        df = candles(np.array([0, 300, 614, 900]))
        df.loc[1, "close_time"] = 350
        report = validate_ohlcv(df, "5m")
        self.assertFalse(report.is_valid)
        self.assertEqual(report.alignment_errors, 1)
        self.assertEqual(report.close_time_errors, 1)
        clean, excluded = separate_complete_candles(df, 300)
        self.assertEqual(clean.open_time.tolist(), [0, 900])
        self.assertEqual(excluded.reason.tolist(), ["incomplete_or_irregular_duration", "unaligned_open_time"])
        self.assertTrue(validate_ohlcv(clean, "5m").is_valid)

    def test_synthetic_rows_are_not_validated_as_observed_market_data(self):
        df = candles(np.array([0, 300]))
        df["is_synthetic"] = [0, 1]
        report = validate_ohlcv(df, "5m")
        self.assertEqual(report.synthetic_rows, 1)
        self.assertFalse(report.is_valid)

    def test_gap_resets_ema_and_never_labels_across_gap(self):
        times = np.r_[np.arange(250) * 300, 260 * 300 + np.arange(230) * 300]
        df = candles(times)
        result = compute_all_indicators(df, interval="5m")
        self.assertEqual(result.attrs["warmup_rows_dropped"], 400)
        self.assertEqual(len(result), 80)
        self.assertTrue(pd.isna(result.loc[result.open_time == times[249], "target_next_up"]).all())
        second = compute_all_indicators(df.iloc[250:].reset_index(drop=True), interval="5m")
        got = result.loc[result.open_time == times[450], "ema_21"].iloc[0]
        self.assertAlmostEqual(got, second.ema_21.iloc[0])
        self.assertTrue(pd.isna(result.target_next_up.iloc[-1]))

    def test_williams_flat_range_is_undefined_but_other_nan_is_critical(self):
        df = candles(np.arange(240) * 300)
        df.loc[200:215, ["open", "high", "low", "close"]] = 100
        result = compute_all_indicators(df, interval="5m")
        report = validate_features(result, "5m")
        self.assertGreater(report.acceptable_null_counts.get("williams_r_14", 0), 0)
        self.assertNotIn("williams_r_14", report.critical_null_counts)
        result.loc[0, "ema_21"] = np.nan
        bad = validate_features(result, "5m")
        self.assertFalse(bad.is_valid)
        self.assertEqual(bad.critical_null_counts["ema_21"], 1)

    def test_validator_checks_indicator_formula_and_complete_schema(self):
        result = compute_all_indicators(candles(np.arange(240) * 300), interval="5m")
        result.loc[25, "williams_r_14"] += 1
        report = validate_features(result, "5m")
        self.assertEqual(report.indicator_errors["williams_r_14_formula"], 1)
        missing = validate_features(result.drop(columns=["roc_5"]), "5m")
        self.assertIn("roc_5", missing.missing_feature_columns)
        self.assertFalse(missing.is_valid)

    def test_zero_volume_denominators_remain_undefined(self):
        df = candles(np.arange(70) * 300)
        df.volume = 0.0
        volume = add_volume_indicators(df)
        self.assertTrue(pd.isna(volume.rvol_20.iloc[19]))
        self.assertTrue(pd.isna(volume.volume_zscore_20.iloc[19]))
        self.assertTrue(pd.isna(volume.vwap_20.iloc[19]))
        self.assertTrue(pd.isna(volume.vwap_60.iloc[59]))

    def test_feature_values_do_not_depend_on_future_candles(self):
        df = candles(np.arange(260) * 300)
        full = compute_all_indicators(df, interval="5m")
        prefix = compute_all_indicators(df.iloc[:230].copy(), interval="5m")
        cols = [c for c in prefix.columns if c not in df.columns and c != "target_next_up"]
        pd.testing.assert_frame_equal(full.loc[:29, cols], prefix.loc[:29, cols])

    def test_merger_marks_synthetic_rows_and_rejects_unmarked_output(self):
        rows = list(synthetic_gap_rows(0, 820, 60, "123.45"))
        self.assertEqual(len(rows), 13)
        self.assertEqual([int(row[0]) for row in rows], list(range(60, 820, 60)))
        self.assertTrue(all(row[1:5] == ["123.45"] * 4 and row[5] == "0" and row[-1] == "1" for row in rows))
        with self.assertRaisesRegex(ValueError, "requires a header"):
            merge_and_validate("BTCUSDT", Path("unused"), Path("unused.csv"), fill_missing=True, include_header=False)
        with patch.object(Path, "glob", return_value=[]):
            self.assertEqual(get_sorted_zip_files(Path("unused"), "BTCUSDT", "1m"), [])
        self.assertIsNone(clock_grid_exclusion_reason(300, 599, 300))
        self.assertEqual(clock_grid_exclusion_reason(314, 613, 300), "unaligned_open_time")
        self.assertEqual(clock_grid_exclusion_reason(300, 550, 300), "irregular_duration")


if __name__ == "__main__":
    unittest.main()
