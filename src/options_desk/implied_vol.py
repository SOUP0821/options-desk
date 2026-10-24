"""Invert Black-Scholes-Merton for annualized volatility."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq

from options_desk.black_scholes import price

Number = float | np.ndarray


def implied_volatility(
    premium: ArrayLike,
    spot: ArrayLike,
    strike: ArrayLike,
    rate: ArrayLike,
    years: ArrayLike,
    dividend: ArrayLike = 0.0,
    option: str = "call",
) -> Number:
    """Vol that reprices premium. NaN when the premium is outside no-arb bounds."""
    option = str(option).lower()
    if option not in {"call", "put"}:
        raise ValueError("option must be 'call' or 'put'")
    arrays = np.broadcast_arrays(
        *[
            np.asarray(value, dtype=np.float64)
            for value in (premium, spot, strike, rate, years, dividend)
        ]
    )
    shape = arrays[0].shape
    flat = [array.ravel() for array in arrays]
    solved = np.empty(flat[0].shape, dtype=np.float64)
    for index in range(solved.size):
        solved[index] = _one_volatility(*(values[index] for values in flat), option)
    if shape == ():
        return float(solved[0])
    return solved.reshape(shape)


def _one_volatility(premium, spot, strike, rate, years, dividend, option) -> float:
    if not all(map(math.isfinite, (premium, spot, strike, rate, years, dividend))):
        return math.nan
    if spot <= 0.0 or strike <= 0.0 or years < 0.0:
        return math.nan
    if years == 0.0:
        return math.nan
    lower = float(price(spot, strike, rate, 0.0, years, dividend, option))
    if premium < lower - 1e-8:
        return math.nan
    if premium <= lower + 1e-12:
        return 0.0
    if option == "call":
        cap = spot * math.exp(-dividend * years)
    else:
        cap = strike * math.exp(-rate * years)
    if premium > cap + 1e-8:
        return math.nan

    def gap(vol: float) -> float:
        return float(price(spot, strike, rate, vol, years, dividend, option)) - premium

    low = 1e-8
    high = 5.0
    gap_low = gap(low)
    if gap_low > 0.0:
        return 0.0 if gap_low < 1e-6 else math.nan
    gap_high = gap(high)
    if gap_high < 0.0:
        high = 10.0
        gap_high = gap(high)
        if gap_high < 0.0:
            return math.nan
    try:
        return float(brentq(gap, low, high, xtol=1e-14, rtol=1e-12, maxiter=200))
    except (ValueError, RuntimeError):
        return math.nan
