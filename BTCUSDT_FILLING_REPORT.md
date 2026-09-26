# BTCUSDT synthetic gap filling report

The `merger.py --fill-missing --clock-grid` runs used the downloaded Binance Vision monthly archives through August 2026. They made **separate calendar views**; the observed-only `processed/BTCUSDT.csv` and the leakage-safe 5m feature dataset were not replaced.

| Clock grid | Valid observed bars | Exchange bars saved as exceptions | Synthetic bars | Output rows | Gap intervals | Cadence verified |
|---|---:|---:|---:|---:|---:|---|
| 1m | 4,724,462 | 21,617 | 30,178 | 4,754,640 | 34 | Every timestamp exactly 60 seconds apart |
| 5m | 948,969 | 256 | 1,959 | 950,928 | 33 | Every timestamp exactly 300 seconds apart |

Each synthetic bar is marked `is_synthetic=1`; its open, high, low and close equal the most recent **valid observed** close, and its volume, quote volume and trade count are zero. This uses no future price. Original bars are marked `0`. Bars whose timestamps or duration do not fit the clock grid are saved verbatim in the exceptions CSV rather than silently shifted or discarded.

## Deliverables

| Interval | Continuous CSV | Continuous Parquet | Per-gap report | Preserved source exceptions |
|---|---|---|---|---|
| 1m | `processed/BTCUSDT_1m_grid.csv` | `processed/parquet/BTCUSDT_1m_grid.parquet` | `processed/BTCUSDT_1m_grid.fill_report.json` | `processed/BTCUSDT_1m_grid.excluded_source.csv` |
| 5m | `processed/BTCUSDT_5m_grid.csv` | `processed/parquet/BTCUSDT_5m_grid.parquet` | `processed/BTCUSDT_5m_grid.fill_report.json` | `processed/BTCUSDT_5m_grid.excluded_source.csv` |

The two Parquet files were read back and checked: every timestamp step matches its interval; every synthetic OHLC equals the previous valid close; every synthetic activity field is zero; and the flag counts match both reports. `validate_ohlcv` reports zero gaps, alignment errors, short intervals and duration errors for each grid. It correctly does **not** classify either grid as valid observed-only market data because they contain synthetic rows.

## Interpretation for modeling

The 1m source includes a long timestamp-shifted period. Creating a strict minute grid therefore requires 20,414 consecutive synthetic minutes from 2017-12-04 06:00 through 2017-12-18 10:13 UTC, even though off-grid exchange records exist and are retained in the exceptions file. The largest 5m synthetic stretch is 646 slots from 2018-02-08 00:25 through 2018-02-10 06:10 UTC.

**A continuous calendar is not continuous observed market history.** The existing 5m training feature dataset continues to exclude synthetic candles and to reset indicator state at gaps. Training on the grid files as if the synthetic flat bars were observed would create artificial returns, volatility and volume signals. Use `is_synthetic` to keep those rows out of training and targets. A model trained only on real candles will still have documented historical discontinuities.
