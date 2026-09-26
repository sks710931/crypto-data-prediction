import unittest
from unittest.mock import patch

import pandas as pd

from main import parse_args
from src.training_readiness import assign_training_rows, audit_next_candle_labels


class TrainingReadinessTests(unittest.TestCase):
    def test_default_split_reserves_the_last_three_months(self):
        with patch("sys.argv", ["main.py", "--pair", "BTCUSDT", "--interval", "5m"]):
            args = parse_args()
        self.assertEqual(args.train_end, "2026-03-01")
        self.assertEqual(args.validation_end, "2026-06-01")

    def test_label_audit_distinguishes_flat_gap_and_end(self):
        source = pd.DataFrame({
            "open_time": [0, 300, 600, 1200, 1500],
            "open": [100, 100, 100, 100, 100],
            "close": [100, 101, 100, 100, 99],
        })
        features = source[["open_time"]].copy()
        features["target_next_up"] = pd.array([1, pd.NA, pd.NA, 0, pd.NA], dtype="Int8")
        audit, reasons = audit_next_candle_labels(features, source, 300)
        self.assertEqual(audit["mismatches"], 0)
        self.assertEqual(audit["flat_next_candle"], 1)
        self.assertEqual(audit["next_candle_absent"], 2)
        self.assertEqual(reasons.tolist(), ["", "flat_next_candle", "gap_or_end", "", "gap_or_end"])
        features.loc[0, "target_next_up"] = 0
        self.assertEqual(audit_next_candle_labels(features, source, 300)[0]["mismatches"], 1)

    def test_split_ledger_purges_outcome_crossing_each_boundary(self):
        times = [0, 300, 600, 900, 1200, 1500]
        source = pd.DataFrame({"open_time": times, "open": [100] * 6, "close": [101] * 6})
        features = source[["open_time"]].copy()
        features["target_next_up"] = pd.array([1, 1, 1, 1, 1, pd.NA], dtype="Int8")
        ledger, audit = assign_training_rows(features, source, 300, 900, 1500)
        self.assertEqual(audit["mismatches"], 0)
        self.assertEqual(ledger.split.tolist(), ["train"] * 3 + ["validation"] * 2 + ["test"])
        self.assertEqual(ledger.exclude_reason.tolist(),
                         ["", "", "target_crosses_split", "", "target_crosses_split", "gap_or_end"])


if __name__ == "__main__":
    unittest.main()
