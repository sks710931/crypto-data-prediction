#!/usr/bin/env python3
"""Evaluate TypeSafe Jev on historical BTCUSDT 5-minute feature rows via OpenRouter.

Endpoint: https://openrouter.ai/api/alpha/decisions (NOT chat/completions)
One prediction per completed 5-minute candle; no future data is sent to Jev.
Prediction: direction of next completed 5-minute close vs current close.

Usage:
    pip install pandas pyarrow requests
    export OPENROUTER_API_KEY='...'
    python jev_openrouter_compare.py --dry-run
    python jev_openrouter_compare.py --limit 5
    python jev_openrouter_compare.py  # resume and process remaining rows
    python jev_openrouter_compare.py --return-buckets --out jev_results_buckets

Output:
    OUT/requests_responses.jsonl : append-only request/response audit journal
    OUT/predictions.csv         : flattened, resumable prediction results
    OUT/manifest.json           : dataset and experiment configuration

Note: This is a *retrospective* model query, not a live historical prediction.
Jev probabilities must be tested for calibration in this particular domain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import requests

LOG = logging.getLogger("jev_compare")
API_URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"  # Record the resolved snapshot from each API response.
SCRIPT_VERSION = "1.1"


def resolve_input_path(file_input: str | Path | None = None) -> Path:
    """Resolve input dataset path across current directory, script dir, and processed/parquet/."""
    candidates = [
        Path("processed/parquet/BTCUSDT_5m_features_test_1000.parquet"),
        Path(__file__).resolve().parent / "processed" / "parquet" / "BTCUSDT_5m_features_test_1000.parquet",
        Path("BTCUSDT_5m_features_test_1000.parquet"),
        Path(__file__).resolve().parent / "BTCUSDT_5m_features_test_1000.parquet",
        Path("processed/parquet/BTCUSDT_5m_features.parquet"),
        Path(__file__).resolve().parent / "processed" / "parquet" / "BTCUSDT_5m_features.parquet",
    ]
    if file_input is None:
        for c in candidates:
            if c.is_file():
                return c
        return candidates[0]

    p = Path(file_input)
    if p.is_file():
        return p

    script_dir = Path(__file__).resolve().parent
    if (script_dir / p).is_file():
        return script_dir / p

    alt_p = Path("processed/parquet") / p.name
    if alt_p.is_file():
        return alt_p
    if (script_dir / "processed" / "parquet" / p.name).is_file():
        return script_dir / "processed" / "parquet" / p.name

    if not p.suffix:
        for candidate in [
            p.with_suffix(".parquet"),
            alt_p.with_suffix(".parquet"),
            (script_dir / "processed" / "parquet" / f"{p.name}.parquet"),
        ]:
            if candidate.is_file():
                return candidate

    return p


def load_dotenv(env_path: Path | None = None) -> None:
    """Lightweight .env file parser so API keys can be loaded without third-party dependencies."""
    if env_path is None:
        for candidate in [Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"]:
            if candidate.is_file():
                env_path = candidate
                break

    if env_path and env_path.is_file():
        try:
            with env_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
        except Exception as exc:
            LOG.warning("Could not read .env file at %s: %s", env_path, exc)


def get_api_key() -> str | None:
    load_dotenv()
    for var in ("OPENROUTER_API_KEY", "OPEN_ROUTER_KEY", "OPENROUTER_KEY"):
        val = os.environ.get(var)
        if val and val.strip():
            return val.strip()
    return None


# Curated, backward-looking inputs. No open_time/close_time, next candle, labels,
# price targets, or future-derived statistics are transmitted to Jev.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "trend": (
        "price_dist_ema21_pct", "ema9_ema21_spread_pct", "ema21_slope_pct",
        "macd_hist", "macd_hist_change", "adx_14", "plus_di_14",
        "minus_di_14", "di_diff_14", "supertrend_dir_10_3",
    ),
    "momentum": (
        "rsi_7", "rsi_14", "rsi_14_change", "roc_5", "roc_10",
        "roc_20", "stochrsi_k_14_3_3", "stochrsi_d_14_3_3", "mfi_14",
    ),
    "volatility": (
        "natr_14", "bb_bandwidth_20_2", "bb_pct_b_20_2",
        "volatility_returns_5", "volatility_returns_15",
        "volatility_returns_30", "volatility_returns_60", "parkinson_vol_20",
    ),
    "activity": (
        "rvol_20", "volume_zscore_20", "cmf_20", "number_of_trades",
        "dist_vwap_20_pct", "dist_vwap_60_pct",
    ),
}

# Exploratory return-range predictions. Jev does NOT produce a numerical
# expected-return estimate, and these buckets are NOT calibrated.
RETURN_BUCKETS: dict[str, str] = {
    "lt_m030": "Next 5-minute closing return is less than -0.30%.",
    "m030_m015": "Next 5-minute closing return is from -0.30% (inclusive) to -0.15% (exclusive).",
    "m015_m005": "Next 5-minute closing return is from -0.15% (inclusive) to -0.05% (exclusive).",
    "neutral": "Next 5-minute closing return is from -0.05% to +0.05%, inclusive.",
    "p005_p015": "Next 5-minute closing return is above +0.05% and up to +0.15%, inclusive.",
    "p015_p030": "Next 5-minute closing return is above +0.15% and up to +0.30%, inclusive.",
    "gt_p030": "Next 5-minute closing return is greater than +0.30%.",
}


@dataclass(frozen=True)
class Experiment:
    input_path: Path
    out_dir: Path
    limit: int | None
    include_return_buckets: bool
    delay_seconds: float
    timeout_seconds: float
    max_retries: int


def finite_number(value: Any, *, digits: int = 7) -> int | float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, digits)


def utc_iso(epoch_ts: int | float) -> str:
    ts = float(epoch_ts)
    if ts >= 1_000_000_000_000:
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(timespec="milliseconds")
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def make_questions(include_return_buckets: bool) -> dict[str, Any]:
    questions: dict[str, Any] = {
        "direction": {
            "type": "choice",
            "instructions": (
                "Using only the supplied BTCUSDT market-state indicators observed at "
                "the CLOSE of a completed 5-minute candle, estimate which outcome "
                "describes the CLOSE of the IMMEDIATELY NEXT completed 5-minute candle "
                "relative to the current completed candle's CLOSE. This is a forecast, "
                "not a rule-based classification of the supplied indicators."
            ),
            "criteria": {
                "UP": "BTCUSDT next 5-minute candle closes STRICTLY ABOVE current completed candle's close.",
                "DOWN": "BTCUSDT next 5-minute candle closes BELOW OR EQUAL TO current completed candle's close.",
            },
        }
    }
    if include_return_buckets:
        questions["return_bucket"] = {
            "type": "choice",
            "instructions": (
                "Estimate which percentage-return INTERVAL will contain "
                "100 * (next completed BTCUSDT 5-minute CLOSE / current completed "
                "5-minute CLOSE - 1). The supplied state has no information from "
                "the future candle. Choose the most plausible interval."
            ),
            "criteria": RETURN_BUCKETS,
        }
    return questions


def build_state(df: pd.DataFrame, position: int, lookback: int = 12) -> dict[str, Any]:
    row = df.iloc[position]
    left = max(0, position - lookback)
    history = df.iloc[left : position + 1]
    closes = history["close"]
    candle_returns = closes.pct_change(fill_method=None).iloc[1:] * 100.0

    open_price, close_price = float(row["open"]), float(row["close"])
    high, low = float(row["high"]), float(row["low"])
    candle_range = high - low
    volume = float(row["volume"])

    # Historical timestamps are logged OUTSIDE the Jev state so the experiment
    # does not directly tell the model which historical date it is forecasting.
    state: dict[str, Any] = {
        "instrument": "BTCUSDT spot",
        "observed_candle": "completed 5-minute candle",
        "forecast_horizon": "immediately next completed 5-minute candle",
        "price_action": {
            "current_candle_return_pct": finite_number((close_price / open_price - 1) * 100) if open_price else None,
            "current_candle_range_pct": finite_number((candle_range / close_price) * 100) if close_price else None,
            "current_candle_close_position_in_range": finite_number((close_price - low) / candle_range) if candle_range > 0 else None,
            "last_completed_candle_returns_pct": [finite_number(v) for v in candle_returns.tolist()],
            "last_3_candle_close_return_pct": finite_number((closes.iloc[-1] / closes.iloc[-4] - 1) * 100)
            if len(closes) >= 4 and closes.iloc[-4] > 0 else None,
            "last_6_candle_close_return_pct": finite_number((closes.iloc[-1] / closes.iloc[-7] - 1) * 100)
            if len(closes) >= 7 and closes.iloc[-7] > 0 else None,
        },
        "order_flow_proxies": {
            "taker_buy_base_volume_fraction": finite_number(
                float(row["taker_buy_base_asset_volume"]) / volume
            ) if volume > 0 else None,
            "number_of_trades": int(row["number_of_trades"]) if pd.notna(row["number_of_trades"]) else None,
        },
    }
    for group, names in FEATURE_GROUPS.items():
        state[group] = {
            name: finite_number(row[name]) for name in names
            if name in df.columns
        }
    return state


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def prepare_frame(path: Path) -> pd.DataFrame:
    # 'pyarrow' is a local dependency; no server uploads of the Parquet file.
    df = pd.read_parquet(path, engine="pyarrow")
    required = {
        "open_time", "close_time", "open", "high", "low", "close", "volume",
        "number_of_trades", "taker_buy_base_asset_volume",
    }
    required.update(feature for group in FEATURE_GROUPS.values() for feature in group)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input missing expected columns: {missing}")

    df = df.sort_values("open_time").reset_index(drop=True)
    if df["open_time"].duplicated().any():
        raise ValueError("Repeated open_time values detected; deduplicate input first")
    if df.empty:
        raise ValueError("Input Parquet is empty")
    if df[list(required)].isna().any(axis=None):
        # Nulls are allowed as JSON null in non-essential technical indicators,
        # but open/close/time/volume MUST be valid for labels and feature ratios.
        core = ["open_time", "close_time", "open", "high", "low", "close", "volume"]
        if df[core].isna().any(axis=None):
            raise ValueError("Missing OHLCV/time values; fix input before evaluating")
        LOG.warning("Some indicators contain nulls; those values will be omitted/null in Jev state")

    # Uploaded Binance schema represents timestamps as epoch milliseconds.
    # Fail loudly rather than silently confusing integer seconds with milliseconds.
    for column in ("open_time", "close_time"):
        if not pd.api.types.is_integer_dtype(df[column]):
            raise ValueError(f"{column} must be an integer epoch column")

    sample_ts = int(df["open_time"].iloc[0])
    if sample_ts >= 1_000_000_000_000:
        is_ms = True
        timeframe_step = 5 * 60 * 1000
        now_ts = int(time.time() * 1000)
        expected_range = (1_000_000_000_000, 5_000_000_000_000)
    elif sample_ts >= 1_000_000_000:
        is_ms = False
        timeframe_step = 5 * 60
        now_ts = int(time.time())
        expected_range = (1_000_000_000, 5_000_000_000)
    else:
        raise ValueError(f"Unexpected open_time sample ({sample_ts}): expected epoch seconds or milliseconds")

    for column in ("open_time", "close_time"):
        if not df[column].between(expected_range[0], expected_range[1]).all():
            raise ValueError(f"Unexpected {column}: expected range {expected_range}")
    if (df["close"] <= 0).any() or (df["open"] <= 0).any():
        raise ValueError("Nonpositive BTC prices detected")

    # In a historical file, all rows remain. If the last row is currently
    # incomplete, remove it instead of presenting it as a completed candle.
    while not df.empty and int(df.iloc[-1]["close_time"]) > now_ts:
        LOG.warning("Skipping trailing unclosed candle at %s", utc_iso(int(df.iloc[-1]["open_time"])))
        df = df.iloc[:-1]
    if df.empty:
        raise ValueError("No completed candle rows available")

    gaps = df["open_time"].diff().dropna()
    if (gaps != timeframe_step).any():
        LOG.warning("Detected %d timestamp gaps; labels across gaps will be null", int((gaps != timeframe_step).sum()))

    df.attrs["timeframe_step"] = timeframe_step
    df.attrs["is_ms"] = is_ms
    return df.reset_index(drop=True)


def actual_outcome(df: pd.DataFrame, position: int, timeframe_step: int | None = None) -> dict[str, Any]:
    close = float(df.iloc[position]["close"])
    if position + 1 >= len(df):
        return {"actual_direction": None, "actual_close_return_pct": None,
                "next_open": None, "next_close": None, "label_status": "no_next_bar_in_file"}

    current = df.iloc[position]
    nxt = df.iloc[position + 1]
    step = timeframe_step if timeframe_step is not None else df.attrs.get("timeframe_step")
    if step is None:
        sample_ts = int(current["open_time"])
        step = 5 * 60 * 1000 if sample_ts >= 1_000_000_000_000 else 5 * 60

    if int(nxt["open_time"]) - int(current["open_time"]) != step:
        return {"actual_direction": None, "actual_close_return_pct": None,
                "next_open": None, "next_close": None, "label_status": "missing_next_5m_bar"}
    nxt_close = float(nxt["close"])
    next_return = ((nxt_close / close) - 1) * 100 if close > 0 else None
    return {
        "actual_direction": "UP" if nxt_close > close else "DOWN",
        "actual_close_return_pct": next_return,
        "next_open": float(nxt["open"]),
        "next_close": nxt_close,
        "label_status": "labeled",
    }


def load_journal(journal: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    if journal.exists():
        with journal.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Corrupt journal line {line_no}; do not overwrite it") from e
                entries[record["request_fingerprint"]] = record
    return entries


def write_journal(journal: Path, record: dict[str, Any]) -> None:
    with journal.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def is_transient(status: int) -> bool:
    return status in (408, 409, 425, 429, 500, 502, 503, 504, 524, 529)


def call_jev(
    session: requests.Session, api_key: str, request_body: dict[str, Any],
    timeout: float, max_retries: int,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/sks710931/crypto-data-prediction",
        "X-Title": "Crypto Data Prediction",
    }

    for attempt in range(max_retries + 1):
        try:
            response = session.post(API_URL, headers=headers, json=request_body, timeout=timeout)
            if response.ok:
                data = response.json()
                if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
                    raise ValueError("Unexpected Jev response: missing answers object")
                return data
            if not is_transient(response.status_code):
                raise RuntimeError(
                    f"OpenRouter returned HTTP {response.status_code}: {response.text[:500]}"
                )
            err = f"OpenRouter transient HTTP {response.status_code}: {response.text[:160]}"
            retry_after = response.headers.get("Retry-After")
        except (requests.Timeout, requests.ConnectionError) as exc:
            err, retry_after = str(exc), None
        if attempt >= max_retries:
            raise RuntimeError(f"Jev call failed after {attempt + 1} attempts: {err}")
        try:
            wait = min(float(retry_after), 120) if retry_after else min(2 ** attempt, 30) + random.random()
        except ValueError:
            wait = min(2 ** attempt, 30) + random.random()
        LOG.warning("%s; retrying after %.1fs", err, wait)
        time.sleep(wait)
    raise AssertionError("Unreachable")


def validate_and_flatten(record: Mapping[str, Any], has_buckets: bool) -> dict[str, Any]:
    response = record["response"]
    answers = response["answers"]
    direction = answers.get("direction")
    if not isinstance(direction, dict) or direction.get("type") != "choice":
        raise ValueError("Unexpected Jev direction answer")
    probabilities = direction.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != {"UP", "DOWN"}:
        raise ValueError(f"Unexpected Jev direction probabilities: {probabilities}")
    p_up = float(probabilities["UP"])
    p_down = float(probabilities["DOWN"])
    if not (math.isfinite(p_up) and math.isfinite(p_down)
            and 0 <= p_up <= 1 and 0 <= p_down <= 1
            and abs(p_up + p_down - 1) <= 0.025):
        raise ValueError("Invalid Jev probability distribution")
    if direction.get("choice") not in ("UP", "DOWN"):
        raise ValueError("Missing/invalid Jev direction choice")

    usage = response.get("usage") or {}
    result: dict[str, Any] = {
        "open_time_utc": record["open_time_utc"],
        "close_time_utc": record["close_time_utc"],
        "current_close": record["current_close"],
        "jev_prediction": direction["choice"],
        "jev_p_up": p_up,
        "jev_p_down": p_down,
        "jev_confidence": direction.get("confidence"),
        "actual_direction": record["outcome"]["actual_direction"],
        "actual_close_return_pct": record["outcome"]["actual_close_return_pct"],
        "next_open": record["outcome"]["next_open"],
        "next_close": record["outcome"]["next_close"],
        "label_status": record["outcome"]["label_status"],
        "model_requested": record["request"]["model"],
        "model_resolved": response.get("model"),
        "response_id": response.get("id"),
        "jev_input_tokens": usage.get("input_tokens", usage.get("inputTokens")),
        "jev_output_tokens": usage.get("output_tokens", usage.get("outputTokens")),
        "jev_cost_usd": usage.get("cost"),
        "request_fingerprint": record["request_fingerprint"],
    }
    if has_buckets:
        buckets = answers.get("return_bucket")
        if not isinstance(buckets, dict) or buckets.get("type") != "choice":
            raise ValueError("Missing/invalid Jev return_bucket response")
        bp = buckets.get("probabilities")
        if not isinstance(bp, dict) or set(bp) != set(RETURN_BUCKETS):
            raise ValueError("Invalid Jev return_bucket probability keys")
        values = [float(bp[k]) for k in RETURN_BUCKETS]
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in values) or abs(sum(values) - 1) > 0.03:
            raise ValueError("Invalid Jev return_bucket probability distribution")
        result["jev_return_bucket"] = buckets.get("choice")
        for key in RETURN_BUCKETS:
            result[f"jev_return_prob_{key}"] = bp[key]
    return result


def write_results(
    df: pd.DataFrame, out_file: Path, journal_entries: Mapping[str, dict[str, Any]],
    requests_by_time: Mapping[int, str], with_buckets: bool,
) -> int:
    output: list[dict[str, Any]] = []
    for open_time in df["open_time"]:
        request_hash = requests_by_time.get(int(open_time))
        if request_hash and request_hash in journal_entries:
            output.append(validate_and_flatten(journal_entries[request_hash], with_buckets))
    if output:
        destination_tmp = out_file.with_suffix(out_file.suffix + ".tmp")
        pd.DataFrame(output).to_csv(destination_tmp, index=False, float_format="%.10g")
        destination_tmp.replace(out_file)
    return len(output)


def run(experiment: Experiment, dry_run: bool) -> None:
    df = prepare_frame(experiment.input_path)
    questions = make_questions(experiment.include_return_buckets)
    count = len(df) if experiment.limit is None else min(experiment.limit, len(df))
    target_indices = range(count)  # Start at the oldest of these 1000 rows; --limit 5 then resume all.
    if not count:
        LOG.info("No rows requested")
        return

    first = build_state(df, 0)
    LOG.info("Loaded %d completed 5-minute records; selected %d", len(df), count)
    if dry_run:
        print(json.dumps({"preview_request": {"model": MODEL, "state": first, "questions": questions}}, indent=2))
        print("Dry run: no network requests, no API key required.")
        return

    key = get_api_key()
    if not key:
        raise EnvironmentError(
            "API key not found. Please set OPENROUTER_API_KEY (or OPEN_ROUTER_KEY) "
            "in your environment or in a .env file."
        )

    output_dir = experiment.out_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "script_version": SCRIPT_VERSION,
        "dataset_sha256": sha256_file(experiment.input_path),
        "model": MODEL,
        "timeframe": "5m",
        "target": "close(next 5m bar) > close(current completed 5m bar)",
        "feature_groups": FEATURE_GROUPS,
        "questions": questions,
        "lookback": 12,
    }
    # Avoid silently mixing incomparable or altered experiments.
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if fingerprint(existing) != fingerprint(manifest):
            raise ValueError("Output directory belongs to a different dataset/configuration. Use --out NEW_DIR.")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    journal = output_dir / "requests_responses.jsonl"
    entries = load_journal(journal)
    req_by_time: dict[int, str] = {}
    session = requests.Session()
    successful_new = 0
    for i in target_indices:
        row = df.iloc[i]
        time_ms = int(row["open_time"])
        request_body = {"model": MODEL, "state": build_state(df, i), "questions": questions}
        req_hash = fingerprint(request_body)
        req_by_time[time_ms] = req_hash
        if req_hash in entries:
            LOG.info("%d/%d already saved: %s", i + 1, count, utc_iso(time_ms))
            continue
        LOG.info("%d/%d querying Jev: %s", i + 1, count, utc_iso(time_ms))
        response = call_jev(
            session, key, request_body, experiment.timeout_seconds, experiment.max_retries
        )
        record = {
            "request_fingerprint": req_hash,
            "requested_at_utc": datetime.now(timezone.utc).isoformat(),
            "open_time_utc": utc_iso(time_ms),
            "close_time_utc": utc_iso(int(row["close_time"])),
            "current_close": float(row["close"]),
            "outcome": actual_outcome(df, i, df.attrs.get("timeframe_step")),
            "request": request_body,  # Audit trail; never contains the API key.
            "response": response,
        }
        # Validate first so malformed API results do not pollute the successful journal.
        validate_and_flatten(record, experiment.include_return_buckets)
        write_journal(journal, record)
        entries[req_hash] = record
        successful_new += 1
        # Write CSV incrementally so interruption preserves usable flat output too.
        write_results(df, output_dir / "predictions.csv", entries, req_by_time, experiment.include_return_buckets)
        if i + 1 < count and experiment.delay_seconds:
            time.sleep(experiment.delay_seconds)

    total_saved = write_results(df, output_dir / "predictions.csv", entries, req_by_time, experiment.include_return_buckets)
    costs = [float(r["response"].get("usage", {}).get("cost") or 0)
             for r in entries.values() if r["request_fingerprint"] in set(req_by_time.values())]
    LOG.info("Done. New API calls: %d; saved results: %d; reported API cost: $%.8f",
             successful_new, total_saved, sum(costs))
    LOG.info("Prediction CSV: %s", output_dir / "predictions.csv")
    LOG.info("Raw request/response journal: %s", journal)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=None, help="Input 5m feature Parquet file (default: processed/parquet/BTCUSDT_5m_features_test_1000.parquet)")
    parser.add_argument("--out", type=Path, default=Path("jev_results"), help="Result directory")
    parser.add_argument("--limit", type=int, default=None, help="Process first N rows (e.g. 5 for a smoke test)")
    parser.add_argument("--return-buckets", action="store_true", help="Also request exploratory 5m return ranges")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls, seconds")
    parser.add_argument("--timeout", type=float, default=60.0, help="API request timeout, seconds")
    parser.add_argument("--max-retries", type=int, default=4, help="Transient error retry count")
    parser.add_argument("--dry-run", action="store_true", help="Inspect first API request, send nothing")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.delay < 0 or args.timeout <= 0 or args.max_retries < 0:
        parser.error("Invalid delay, timeout, or max-retries")

    input_path = resolve_input_path(args.input)
    if not input_path.is_file():
        parser.error(f"Input file not found: {args.input or input_path}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run(Experiment(input_path, args.out, args.limit, args.return_buckets,
                       args.delay, args.timeout, args.max_retries), args.dry_run)
    except (OSError, ValueError, RuntimeError, requests.RequestException) as exc:
        LOG.error("Experiment stopped: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
