"""Parse downloaded price frames and retry flaky market-data calls."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd


def close_from_download(frame: pd.DataFrame | pd.Series) -> pd.Series:
    data = frame
    if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
        level0 = {str(value) for value in data.columns.get_level_values(0)}
        level1 = {str(value) for value in data.columns.get_level_values(1)}
        if "Close" in level0:
            data = data["Close"]
        elif "Adj Close" in level0:
            data = data["Adj Close"]
        elif "Close" in level1:
            data = data.xs("Close", axis=1, level=1)
        elif "close" in level0:
            data = data["close"]
        else:
            raise ValueError(f"no close column in {list(data.columns)}")
    if isinstance(data, pd.DataFrame):
        for name in ("Close", "Adj Close", "close"):
            if name in data.columns:
                series = data[name]
                break
        else:
            if data.shape[1] != 1:
                raise ValueError(f"no close column in {list(data.columns)}")
            series = data.iloc[:, 0]
    else:
        series = data
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    series = pd.to_numeric(series, errors="coerce").dropna()
    series = series[series > 0.0].astype(float)
    if series.empty:
        raise ValueError("price history is empty")
    index = pd.to_datetime(series.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    series.index = index
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series.name = "close"
    return series


def retry(func, attempts: int = 4, pause: float = 1.5):
    last = None
    for attempt in range(attempts):
        try:
            return func()
        except Exception as exc:
            last = exc
            if attempt + 1 == attempts:
                break
            time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"request failed after {attempts} attempts") from last
