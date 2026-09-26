"""Read exchange-provided monthly klines without synthesizing missing candles."""

from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from .config import RAW_COLUMNS


def archives_cover_range(raw_dir: Path, pair: str, interval: str, first: int, last: int) -> bool:
    """Require every calendar month touched by the requested timestamp range."""
    months = pd.period_range(
        pd.Timestamp(first, unit="s", tz="UTC").tz_localize(None).to_period("M"),
        pd.Timestamp(last, unit="s", tz="UTC").tz_localize(None).to_period("M"),
        freq="M",
    )
    return all((raw_dir / f"{pair}-{interval}-{month}.zip").is_file() for month in months)


def load_monthly_klines(raw_dir: Path, pair: str, interval: str, first: int, last: int) -> pd.DataFrame:
    """Load genuine Binance Vision candles, retaining malformed rows for an explicit audit."""
    if not archives_cover_range(raw_dir, pair, interval, first, last):
        raise FileNotFoundError("A complete set of monthly exchange archives is required")
    months = pd.period_range(
        pd.Timestamp(first, unit="s", tz="UTC").tz_localize(None).to_period("M"),
        pd.Timestamp(last, unit="s", tz="UTC").tz_localize(None).to_period("M"),
        freq="M",
    )
    frames = []
    for month in months:
        path = raw_dir / f"{pair}-{interval}-{month}.zip"
        with ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.endswith(".csv")]
            if len(names) != 1:
                raise ValueError(f"Expected one CSV in {path}, found {len(names)}")
            with archive.open(names[0]) as source:
                frame = pd.read_csv(source, header=None, names=RAW_COLUMNS, dtype={"open_time": "int64", "close_time": "int64"})
        for col in ("open_time", "close_time"):
            values = frame[col]
            # Binance switched from milliseconds to microseconds in later archives.
            frame[col] = values.where(values < 10**14, values // 1000) // 1000
        frames.append(frame.drop(columns=["ignore"]))
    result = pd.concat(frames, ignore_index=True)
    return result.loc[result.open_time.between(first, last)].reset_index(drop=True)


def separate_complete_candles(df: pd.DataFrame, interval_seconds: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate exact clock-aligned, complete candles from source anomalies."""
    reason = pd.Series("", index=df.index, dtype="string")
    reason.loc[df.open_time % interval_seconds != 0] = "unaligned_open_time"
    duration_bad = df.close_time - df.open_time != interval_seconds - 1
    reason.loc[(reason == "") & duration_bad] = "incomplete_or_irregular_duration"
    invalid = reason != ""
    excluded = df.loc[invalid, ["open_time", "close_time"]].copy()
    excluded["reason"] = reason.loc[invalid].to_numpy()
    return df.loc[~invalid].reset_index(drop=True), excluded.reset_index(drop=True)
