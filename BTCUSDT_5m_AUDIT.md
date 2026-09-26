# BTCUSDT 5m pipeline audit

Run: `python main.py --pair BTCUSDT --interval 5m`, followed by validation of the saved Parquet files. Source range: 2017-08-17 04:00 through 2026-08-31 23:55 UTC. Nine focused regression tests passed. A fresh merge of the 109 raw 1m archives produced a CSV byte-for-byte identical to the existing `processed/BTCUSDT.csv` (SHA-256 `7277404759153EFCAC11FE6F6D4BDE1175B657E8E8BD8C6801BFABA994DA79D0`).

## Confirmed causes and corrections

1. **Unavailable candles.** The old 5m resample had 1,703 empty clock slots in 33 gaps. I inspected the local 1m monthly archives and downloaded all 109 official 5m monthly archives for an independent source check. The direct 5m source has the same 33 gap periods; it supplies no replacement candles for the original exchange-source gaps. Calling them exchange maintenance would be an unsupported inference. The corrected dataset records these periods and never forward fills them.
2. **Invalid clock aggregation.** The 1m source contains 21,602 off-minute timestamps, principally a `:20` offset from 2017-12-04 and a `:14` offset into 2018-02-10. It also contains truncated bars. The original resampler kept 15 bins with fewer than five 1m rows and did not check source alignment or duration. Strict resampling now requires five distinct, complete, aligned 1m bars. This leaves 944,885 valid 5m bars from the 1m source. The complete direct 5m archive supplies 4,084 additional valid aligned bars, so the 5m run prefers that official source. On the direct source, 241 off-grid bars and 15 irregular-duration bars are explicitly excluded and listed in the quality JSON.
3. **Williams %R nulls.** All eight original nulls occur when the rolling 14-candle highest high equals the lowest low, making the formula's denominator zero. Raw archive rows at the affected times confirm flat or zero-trade candles. No numeric value is inserted. The corrected source and warmup policy leave six such nulls, classified as mathematically undefined. Volume ratio, z-score and VWAP denominators are also left undefined when zero, rather than being altered with a small constant.
4. **State crossing gaps.** The old indicator code calculated rolling statistics, EMA, MACD, Supertrend and OBV through missing time, as if adjacent rows were adjacent candles. The pipeline now splits at every interval discontinuity and restarts every indicator family within each continuous segment. It explicitly excludes the first 200 rows of each segment from the feature store (6,627 rows across 34 segments; short segments are excluded in full).
5. **Next-candle labels.** The old dataset had no target, and a simple row shift would have labeled across gaps. `target_next_up` is now 1 if the next contiguous candle closes above its open, 0 if below, and null for a gap, the end, or a flat next candle. The label is computed before warmup exclusion. It is a target column, never an input feature.
6. **Validation and upstream safeguards.** Historical gaps and mathematically undefined values are now reported separately from critical errors. Validation checks clock alignment, candle duration, required feature columns, finite values, selected indicator formulas, and target domain. Merger no longer allows synthetic gap filling or silently accepts malformed, out-of-order, or conflicting duplicate rows. Archive matching is interval-specific.

## Actual-data results

| Measure | Before | After |
|---|---:|---:|
| 5m source rows accepted | 949,225 | 948,969 |
| Unavailable aligned 5m clock slots | 1,703 | 1,959 |
| Gap intervals | 33 | 33 |
| Feature rows | 949,025 | 942,342 |
| Warmup rows excluded | 200 globally | 6,627 by segment |
| Williams %R nulls | 8, reported as errors | 6, verified zero-denominator cases |
| Critical feature nulls / infinities | 8 / 0 | 0 / 0 |
| Duplicate timestamps / indicator rule violations | 0 / 0 | 0 / 0 |
| Non-null next-candle labels | absent | 937,312 |

The extra 256 unavailable clock slots are the **excluded invalid source bars**, not newly discovered missing market data. Of 5,030 null targets, 33 are at a gap or the dataset end and 4,997 precede a flat next candle. The saved feature file contains 65 columns: 11 OHLCV, 53 indicators, and the target. Both the saved 5m Parquet and saved feature Parquet passed a fresh validation read after the final run.

An optional, explicitly synthetic clock-grid view has since been written separately; see [BTCUSDT_FILLING_REPORT.md](BTCUSDT_FILLING_REPORT.md). It does not replace this observed-only feature dataset.

## Remaining limitations

- Direct official 5m candles and a strict aggregation of the local 1m source disagree on at least one OHLCV field in 3,113 of their 944,885 overlapping valid clock slots, mostly in 2017. The pipeline uses direct official 5m candles for the 5m model; cross-timeframe joins involving 1m data need a separate reconciliation policy.
- The 1,959 unavailable clock slots cannot be represented as observed, aligned, complete 5m candles from the available official archives. Their cause is not established. The quality JSON gives exact timestamps and excluded-source reasons.
- EMA and OBV restart after each gap. The 200-bar warmup limits transient state effects but does not recreate an unknowable pre-gap market state.
- XGBoost can handle the six undefined Williams %R values as missing features. Supervised training must select rows with non-null targets and remove `target_next_up` from the feature matrix. Use a chronological holdout and make predictions only after the current candle has closed.
- The monthly ZIP files are read with ZIP integrity checks, but the downloader does not verify a separate cryptographic checksum. The cached 1m Parquet is reused when present; refresh it explicitly if the underlying CSV changes.
