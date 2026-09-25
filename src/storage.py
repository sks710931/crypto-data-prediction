"""
High-Performance Storage Module.
Converts large OHLCV CSVs to compressed Parquet format and provides fast I/O utilities.
"""

import time
from pathlib import Path
from typing import Optional
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv_csv
import pyarrow.parquet as pq

from .config import COLUMN_DTYPES, PARQUET_DIR


def get_pyarrow_schema() -> pa.Schema:
    """Returns strict pyarrow schema for Binance klines."""
    type_map = {
        "open_time": pa.int64(),
        "open": pa.float64(),
        "high": pa.float64(),
        "low": pa.float64(),
        "close": pa.float64(),
        "volume": pa.float64(),
        "close_time": pa.int64(),
        "quote_asset_volume": pa.float64(),
        "number_of_trades": pa.int32(),
        "taker_buy_base_asset_volume": pa.float64(),
        "taker_buy_quote_asset_volume": pa.float64(),
        "ignore": pa.float32(),
    }
    return pa.schema([pa.field(col, dtype) for col, dtype in type_map.items()])


def csv_to_parquet(
    csv_path: Path | str,
    parquet_path: Optional[Path | str] = None,
    compression: str = "snappy",
) -> Path:
    """
    Converts a raw/processed CSV to an optimized Parquet file using pyarrow.
    Achieves 5x-10x compression and sub-second query/load times.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found at: {csv_path.resolve()}")

    if parquet_path is None:
        parquet_path = PARQUET_DIR / f"{csv_path.stem}.parquet"
    else:
        parquet_path = Path(parquet_path)

    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading CSV: {csv_path.name} ({csv_path.stat().st_size / (1024 * 1024):.1f} MB)...")
    start_time = time.time()

    # Read CSV with PyArrow multi-threaded reader
    convert_options = pv_csv.ConvertOptions(column_types=get_pyarrow_schema())
    read_options = pv_csv.ReadOptions(use_threads=True, block_size=64 * 1024 * 1024)
    table = pv_csv.read_csv(csv_path, read_options=read_options, convert_options=convert_options)

    read_elapsed = time.time() - start_time
    print(f"CSV Loaded: {len(table):,} rows in {read_elapsed:.2f}s.")

    # Write Parquet with compression
    write_start = time.time()
    pq.write_table(
        table,
        parquet_path,
        compression=compression,
        use_dictionary=True,
    )
    write_elapsed = time.time() - write_start

    csv_mb = csv_path.stat().st_size / (1024 * 1024)
    parquet_mb = parquet_path.stat().st_size / (1024 * 1024)
    compression_ratio = csv_mb / parquet_mb if parquet_mb > 0 else 0

    print(f"Parquet Written: {parquet_path.name} in {write_elapsed:.2f}s.")
    print(f"Size Reduction:  {csv_mb:.1f} MB -> {parquet_mb:.1f} MB ({compression_ratio:.1f}x compression)")

    return parquet_path


def load_parquet(
    parquet_path: Path | str,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Loads Parquet directly into a pandas DataFrame using PyArrow engine.
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found at: {parquet_path.resolve()}")

    return pd.read_parquet(parquet_path, columns=columns, engine="pyarrow")
