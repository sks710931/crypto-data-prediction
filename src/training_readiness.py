"""Point-in-time checks and reproducible split metadata before model training."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, TIMEFRAME_SECONDS
from .indicators import compute_all_indicators


BASE_INPUT_COLUMNS = [
    "open", "high", "low", "close", "volume", "quote_asset_volume",
    "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume",
]


def _sha256(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _freeze_file(path: Path, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copyfile(path, temporary)
        if _sha256(temporary)["sha256"] != expected_sha256:
            temporary.unlink()
            raise ValueError(f"Snapshot copy differs from source: {path}")
        temporary.replace(destination)
    elif _sha256(destination)["sha256"] != expected_sha256:
        raise ValueError(f"Existing frozen snapshot has changed: {destination}")


def _boundary(value: str, step: int) -> int:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid UTC split boundary: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    timestamp = int(parsed.timestamp())
    if timestamp % step:
        raise ValueError(f"Split boundary must align to the candle interval: {value}")
    return timestamp


def audit_next_candle_labels(features: pd.DataFrame, source: pd.DataFrame, step: int) -> tuple[dict, np.ndarray]:
    """Recreate every label by a timestamp lookup, independent of row shifts."""
    if source.open_time.duplicated().any():
        raise ValueError("Source timestamps must be unique for label auditing")
    times = features.open_time.to_numpy(dtype=np.int64)
    following = source.set_index("open_time")[["open", "close"]].reindex(times + step)
    next_open = following.open.to_numpy(dtype=float)
    next_close = following.close.to_numpy(dtype=float)
    present = np.isfinite(next_open) & np.isfinite(next_close)
    flat = present & (next_open == next_close)
    expected = np.full(len(features), np.nan)
    directional = present & ~flat
    expected[directional] = (next_close[directional] > next_open[directional]).astype(float)
    actual = features.target_next_up.to_numpy(dtype=float, na_value=np.nan)
    mismatch = ~(np.isnan(expected) & np.isnan(actual)) & (expected != actual)
    report = {
        "rows_checked": len(features),
        "mismatches": int(mismatch.sum()),
        "mismatch_open_times": times[mismatch][:10].tolist(),
        "flat_next_candle": int(flat.sum()),
        "next_candle_absent": int((~present).sum()),
    }
    return report, np.select([flat, ~present], ["flat_next_candle", "gap_or_end"], default="")


def assign_training_rows(features: pd.DataFrame, source: pd.DataFrame, step: int,
                         train_end: int, validation_end: int) -> tuple[pd.DataFrame, dict]:
    """Assign every feature row; keep explicit reasons for exclusions."""
    if train_end >= validation_end:
        raise ValueError("train_end must precede validation_end")
    label_audit, null_reasons = audit_next_candle_labels(features, source, step)
    times = features.open_time.to_numpy(dtype=np.int64)
    partitions = np.select([times < train_end, times < validation_end],
                           ["train", "validation"], default="test")
    reasons = null_reasons.astype(object)
    crossing = ((times < train_end) & (times + step >= train_end)) | (
        (times >= train_end) & (times < validation_end) & (times + step >= validation_end)
    )
    reasons[(reasons == "") & crossing] = "target_crosses_split"
    ledger = pd.DataFrame({
        "open_time": times,
        "target_next_up": features.target_next_up.reset_index(drop=True),
        "split": partitions,
        "exclude_reason": reasons,
    })
    return ledger, label_audit


def _prefix_causality_audit(features: pd.DataFrame, source: pd.DataFrame,
                            interval: str) -> dict:
    """Compare saved indicators with calculations that cannot see later candles."""
    step = TIMEFRAME_SECONDS[interval]
    source_times = source.open_time.to_numpy(dtype=np.int64)
    starts = np.r_[0, np.flatnonzero(np.diff(source_times) != step) + 1]
    ends = np.r_[starts[1:], len(source)]
    eligible = [(int(start), int(end)) for start, end in zip(starts, ends) if end - start >= 230]
    if not eligible:
        return {"prefixes_checked": 0, "mismatches": [], "reason": "No segment has 230 candles"}
    selected = [eligible[i] for i in sorted({0, len(eligible) // 2, len(eligible) - 1})]
    indexed = features.set_index("open_time")
    mismatches = []
    checked = 0
    for start, _ in selected:
        prefix = source.iloc[start:start + 230].reset_index(drop=True)
        calculated = compute_all_indicators(prefix, interval=interval)
        timestamp = int(prefix.open_time.iloc[-1])
        if timestamp not in indexed.index:
            mismatches.append({"open_time": timestamp, "feature": "missing_prefix_row"})
            continue
        stored = indexed.loc[timestamp, FEATURE_COLUMNS].to_numpy(dtype=float)
        causal = calculated.iloc[-1][FEATURE_COLUMNS].to_numpy(dtype=float)
        unequal = ~np.isclose(stored, causal, rtol=1e-8, atol=1e-8, equal_nan=True)
        mismatches.extend({"open_time": timestamp, "feature": name}
                          for name in np.array(FEATURE_COLUMNS)[unequal][:10])
        checked += 1
    return {"prefixes_checked": checked, "mismatches": mismatches[:30]}


def _group_profile(features: pd.DataFrame, ledger: pd.DataFrame, mask: np.ndarray,
                   columns: list[str], near_gap: np.ndarray) -> dict:
    part = features.loc[mask]
    rows = ledger.loc[mask]
    included = rows.exclude_reason.eq("")
    labels = rows.loc[included, "target_next_up"]
    profile = {
        "feature_rows": len(part),
        "included_rows": int(included.sum()),
        "up": int((labels == 1).sum()),
        "down": int((labels == 0).sum()),
        "excluded_by_reason": {str(k): int(v) for k, v in rows.loc[~included, "exclude_reason"].value_counts().items()},
        "feature_nulls": {k: int(v) for k, v in part[columns].isna().sum().items() if v},
        "constant_features": [col for col in columns if part[col].nunique(dropna=True) <= 1],
        "rows_adjacent_to_source_gap": int(near_gap[mask].sum()),
    }
    for col in ("close", "volume"):
        values = part[col].quantile([0.01, 0.5, 0.99])
        profile[f"{col}_p01_p50_p99"] = [float(value) if pd.notna(value) else None for value in values]
    return profile


def _baseline(labels: pd.Series, probability: float) -> dict:
    y = labels.to_numpy(dtype=float)
    if len(y) == 0:
        return {"rows": 0}
    p = float(np.clip(probability, 1e-15, 1 - 1e-15))
    return {
        "rows": len(y),
        "log_loss": float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean()),
        "brier_score": float(np.square(y - p).mean()),
    }


def build_training_readiness(features: pd.DataFrame, source: pd.DataFrame, interval: str,
                             train_end: str, validation_end: str, feature_path: Path,
                             source_paths: list[Path], code_paths: list[Path],
                             output_dir: Path, pair: str) -> dict:
    step = TIMEFRAME_SECONDS[interval]
    train_boundary = _boundary(train_end, step)
    validation_boundary = _boundary(validation_end, step)
    columns = BASE_INPUT_COLUMNS + FEATURE_COLUMNS
    required_columns = set(columns) | {"open_time", "close_time", "target_next_up"}
    allowed_columns = required_columns | {"ignore"}
    missing_columns = sorted(required_columns - set(features.columns))
    unexpected_columns = sorted(set(features.columns) - allowed_columns)
    if missing_columns:
        raise ValueError(f"Missing training feature columns: {missing_columns}")
    if "is_synthetic" in source.columns and source.is_synthetic.astype(bool).any():
        raise ValueError("Synthetic candles cannot enter training readiness")

    ledger, label_audit = assign_training_rows(features, source, step, train_boundary, validation_boundary)
    causality = _prefix_causality_audit(features, source, interval)
    source_times = source.open_time.to_numpy(dtype=np.int64)
    gaps = np.flatnonzero(np.diff(source_times) != step)
    adjacent = np.isin(features.open_time.to_numpy(),
                       np.r_[source_times[gaps], source_times[gaps + 1]])
    timestamps = pd.to_datetime(features.open_time, unit="s", utc=True)
    split_masks = {name: ledger.split.eq(name).to_numpy() for name in ("train", "validation", "test")}
    profiles = {name: _group_profile(features, ledger, mask, columns, adjacent)
                for name, mask in split_masks.items()}
    years = timestamps.dt.year.to_numpy()
    yearly = {str(year): _group_profile(features, ledger, years == year, columns, adjacent)
              for year in np.unique(years)}
    train_mask = split_masks["train"] & ledger.exclude_reason.eq("").to_numpy()
    validation_mask = split_masks["validation"] & ledger.exclude_reason.eq("").to_numpy()
    train_labels = ledger.loc[train_mask, "target_next_up"]
    train_prevalence = float(train_labels.mean()) if len(train_labels) else None
    baseline = {
        "probability_from_train_only": train_prevalence,
        "train": _baseline(train_labels, train_prevalence) if train_prevalence is not None else {"rows": 0},
        "validation": _baseline(ledger.loc[validation_mask, "target_next_up"], train_prevalence)
        if train_prevalence is not None else {"rows": 0},
        "test": "reserved for model evaluation; no baseline metric calculated during readiness",
    }
    drift = {}
    for name in ("validation", "test"):
        ranked = []
        for col in columns:
            lower, upper = features.loc[train_mask, col].quantile([0.01, 0.99])
            values = features.loc[split_masks[name], col]
            valid = values.notna()
            if pd.notna(lower) and pd.notna(upper) and valid.any():
                share = float(((values[valid] < lower) | (values[valid] > upper)).mean())
                ranked.append({"feature": col, "outside_train_p01_p99_fraction": share})
        drift[name] = sorted(ranked, key=lambda item: item["outside_train_p01_p99_fraction"], reverse=True)[:10]
    warnings = []
    if drift["validation"] and drift["validation"][0]["outside_train_p01_p99_fraction"] > 0.5:
        worst = drift["validation"][0]
        warnings.append(
            f"Validation drift: {worst['outside_train_p01_p99_fraction']:.1%} of rows are outside "
            f"the train 1st-99th percentile range for {worst['feature']}; "
            "review level-based features with forward evaluation"
        )

    rolling_folds = []
    first_year = int(years.min())
    for year in range(first_year + 2, pd.Timestamp(validation_boundary, unit="s", tz="UTC").year + 1):
        fold_start = int(pd.Timestamp(f"{year}-01-01", tz="UTC").timestamp())
        fold_end = min(int(pd.Timestamp(f"{year + 1}-01-01", tz="UTC").timestamp()), validation_boundary)
        if fold_start >= fold_end:
            continue
        fold_train = (ledger.open_time.to_numpy() + step < fold_start) & ledger.exclude_reason.eq("").to_numpy()
        fold_eval = (ledger.open_time.to_numpy() >= fold_start) & (ledger.open_time.to_numpy() + step < fold_end) & ledger.exclude_reason.eq("").to_numpy()
        if fold_train.any() and fold_eval.any():
            rolling_folds.append({"evaluation_year": year, "train_rows": int(fold_train.sum()),
                                  "evaluation_rows": int(fold_eval.sum())})

    issues = []
    if unexpected_columns:
        issues.append(f"Unexpected feature columns require review: {unexpected_columns}")
    if label_audit["mismatches"]:
        issues.append(f"Label audit found {label_audit['mismatches']} mismatches")
    if causality["mismatches"] or not causality["prefixes_checked"]:
        issues.append("Prefix causality audit did not pass")
    if any(not profiles[name]["included_rows"] for name in ("train", "validation", "test")):
        issues.append("Each chronological split needs labeled rows")
    if any(profiles[name]["constant_features"] for name in ("train", "validation", "test")):
        issues.append("At least one feature is constant in a split; review before training")

    report = {
        "ready": not issues,
        "issues": issues,
        "warnings": warnings,
        "purpose": "next-candle probability forecast",
        "prediction_contract": {
            "decision_time": "after the current observed candle closes",
            "target": "next contiguous observed candle close > its own open",
            "down": "next contiguous observed candle close < its own open",
            "flat_and_missing_next_candle": "target null; excluded only after indicators are computed",
            "probability_interpretation": "P(UP | next observed contiguous candle is non-flat)",
            "synthetic_candles": "excluded from source and training",
        },
        "split_boundaries_utc": {"train_end_exclusive": str(pd.Timestamp(train_boundary, unit="s", tz="UTC")),
                                 "validation_end_exclusive": str(pd.Timestamp(validation_boundary, unit="s", tz="UTC"))},
        "label_audit": label_audit,
        "prefix_causality_audit": causality,
        "feature_whitelist": columns,
        "excluded_input_columns": [col for col in ("open_time", "close_time", "ignore", "target_next_up")
                                   if col in features.columns],
        "preprocessing": "No global scaling or imputation; XGBoost tree models may route undefined feature NaNs",
        "profiles_by_split": profiles,
        "profiles_by_year": yearly,
        "distribution_shift_top_features": drift,
        "rolling_evaluation_folds_before_test": rolling_folds,
        "constant_probability_baseline": baseline,
        "model_selection_metric": "validation log loss; test period must not be used for model selection",
        "predeclared_test_criteria": ["test log loss below test constant-probability baseline",
                                      "test Brier score below test constant-probability baseline",
                                      "report calibration and metrics by year"],
        "limitations": ["Prefix causality is sampled at three segment prefixes, not formally proven for every row",
                        "A passed readiness check does not establish predictive skill or calibrated probabilities",
                        "The target excludes flat next candles, so forecasts are conditional on a non-flat outcome"],
        "trading_evaluation": "not applicable to the requested probability forecast",
        "source_rows": len(source),
        "feature_rows": len(features),
        "source_rows_excluded_during_feature_warmup": len(source) - len(features),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{pair}_{interval}"
    ledger_path = output_dir / f"{stem}_training_rows.parquet"
    temp_ledger = ledger_path.with_suffix(".parquet.tmp")
    ledger.to_parquet(temp_ledger, index=False)
    temp_ledger.replace(ledger_path)
    feature_fingerprint = _sha256(feature_path)
    ledger_fingerprint = _sha256(ledger_path)
    frozen_dir = output_dir / "snapshots" / feature_fingerprint["sha256"]
    frozen_feature = frozen_dir / feature_path.name
    frozen_ledger = frozen_dir / f"{ledger_fingerprint['sha256']}_training_rows.parquet"
    _freeze_file(feature_path, frozen_feature, feature_fingerprint["sha256"])
    _freeze_file(ledger_path, frozen_ledger, ledger_fingerprint["sha256"])
    snapshot = {
        "source_files": [_sha256(path) for path in source_paths],
        "code_files": [_sha256(path) for path in code_paths],
        "feature_file": feature_fingerprint,
        "training_rows_file": ledger_fingerprint,
        "frozen_feature_file": str(frozen_feature.resolve()),
        "frozen_training_rows_file": str(frozen_ledger.resolve()),
        "feature_columns": list(features.columns),
        "source": "observed candles only",
    }
    snapshot_path = output_dir / f"{stem}_snapshot.json"
    _write_json(snapshot_path, snapshot)
    report["artifacts"] = {"snapshot": str(snapshot_path.resolve()),
                           "training_rows": str(frozen_ledger.resolve()),
                           "features": str(frozen_feature.resolve())}
    report_path = output_dir / f"{stem}_training_readiness.json"

    lines = [f"# {pair} {interval} training readiness", "",
             f"Status: **{'READY WITH WARNINGS' if report['ready'] and warnings else 'READY' if report['ready'] else 'REVIEW REQUIRED'}**", "",
             f"Prediction: {report['prediction_contract']['target']}.",
             f"Train ends before {train_end} UTC; validation ends before {validation_end} UTC.", "",
             "| Split | Feature rows | Included labels | UP | DOWN | Exclusions |",
             "|---|---:|---:|---:|---:|---|" ]
    for name, profile in profiles.items():
        lines.append(f"| {name} | {profile['feature_rows']:,} | {profile['included_rows']:,} | "
                     f"{profile['up']:,} | {profile['down']:,} | {profile['excluded_by_reason']} |")
    lines.extend(["", f"Label audit mismatches: {label_audit['mismatches']:,}.",
                  f"Causal prefixes checked: {causality['prefixes_checked']:,}; mismatches: {len(causality['mismatches']):,}.",
                  f"Validation constant-probability log loss: {baseline['validation'].get('log_loss', 'unavailable')}.",
                  "No scaling or imputation is applied. Test label counts are shown for quality control; test metrics are reserved for final evaluation.", ""])
    lines.extend(["## Yearly data profile", "",
                  "| Year | Feature rows | Included | UP | DOWN | Median close | Median volume |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for year, profile in yearly.items():
        lines.append(f"| {year} | {profile['feature_rows']:,} | {profile['included_rows']:,} | "
                     f"{profile['up']:,} | {profile['down']:,} | "
                     f"{profile['close_p01_p50_p99'][1]:,.2f} | "
                     f"{profile['volume_p01_p50_p99'][1]:,.2f} |")
    lines.extend(["", "## Forward feature drift", "",
                  "Fraction outside the training period's 1st-99th percentile range. This uses features only; it is not a test performance result.", "",
                  "| Split | Feature | Outside range |", "|---|---|---:|"])
    for name in ("validation", "test"):
        for item in drift[name][:5]:
            lines.append(f"| {name} | {item['feature']} | {item['outside_train_p01_p99_fraction']:.1%} |")
    lines.extend(["", "## Planned forward folds", "",
                  "| Evaluation year | Earlier training labels | Evaluation labels |",
                  "|---|---:|---:|"])
    for fold in rolling_folds:
        lines.append(f"| {fold['evaluation_year']} | {fold['train_rows']:,} | {fold['evaluation_rows']:,} |")
    lines.extend(["", "## Frozen artifacts", "",
                  f"- Feature SHA-256: `{feature_fingerprint['sha256']}`",
                  f"- Frozen feature file: `{frozen_feature}`",
                  f"- Frozen row ledger: `{frozen_ledger}`",
                  f"- Source file hashes: {len(snapshot['source_files'])}; code file hashes: {len(snapshot['code_files'])}.",
                  "- Model acceptance: improve test log loss and Brier score over a constant probability learned from training; report calibration and yearly results.", ""])
    if warnings:
        lines.extend(["## Modeling warnings", ""] + [f"- {warning}" for warning in warnings] + [""])
    if issues:
        lines.extend(["## Issues", ""] + [f"- {issue}" for issue in issues] + [""])
    markdown_path = output_dir / f"{stem}_training_readiness.md"
    temporary = markdown_path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(markdown_path)
    report["artifacts"]["readiness_json"] = str(report_path.resolve())
    report["artifacts"]["readiness_markdown"] = str(markdown_path.resolve())
    _write_json(report_path, report)
    return report
