"""European call/put prices and Greeks.

years: time to expiry. vol: annualized. rate and dividend: continuous.
vega is dPrice/dVol, not "per vol point". theta is per year of calendar time.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

Number = float | np.ndarray


def price(
    spot: ArrayLike,
    strike: ArrayLike,
    rate: ArrayLike,
    vol: ArrayLike,
    years: ArrayLike,
    dividend: ArrayLike = 0.0,
    option: str = "call",
) -> Number:
    option = _option(option)
    spot, strike, rate, vol, years, dividend = _broadcast(
        spot, strike, rate, vol, years, dividend
    )
    _validate(spot, strike, rate, vol, years, dividend)
    value = _value(spot, strike, rate, vol, years, dividend, option)
    return _scalar(value)


def greeks(
    spot: ArrayLike,
    strike: ArrayLike,
    rate: ArrayLike,
    vol: ArrayLike,
    years: ArrayLike,
    dividend: ArrayLike = 0.0,
    option: str = "call",
) -> dict[str, Number]:
    option = _option(option)
    spot, strike, rate, vol, years, dividend = _broadcast(
        spot, strike, rate, vol, years, dividend
    )
    _validate(spot, strike, rate, vol, years, dividend)

    expired = years <= 0.0
    zero_vol = (vol <= 0.0) & ~expired
    live = ~expired & ~zero_vol
    safe_vol = np.where(live, vol, 1.0)
    safe_years = np.where(years > 0.0, years, 1.0)

    sqrt_t = np.sqrt(safe_years)
    sigma_sqrt = safe_vol * sqrt_t
    d1 = (
        np.log(spot / strike) + (rate - dividend + 0.5 * safe_vol**2) * safe_years
    ) / sigma_sqrt
    d2 = d1 - sigma_sqrt
    disc_spot = np.exp(-dividend * safe_years)
    disc_strike = np.exp(-rate * safe_years)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    pdf_d1 = norm.pdf(d1)

    gamma = disc_spot * pdf_d1 / (spot * sigma_sqrt)
    vega = spot * disc_spot * pdf_d1 * sqrt_t
    if option == "call":
        value = disc_spot * spot * cdf_d1 - disc_strike * strike * cdf_d2
        delta = disc_spot * cdf_d1
        theta = (
            -spot * disc_spot * pdf_d1 * safe_vol / (2.0 * sqrt_t)
            - rate * strike * disc_strike * cdf_d2
            + dividend * spot * disc_spot * cdf_d1
        )
        rho = strike * safe_years * disc_strike * cdf_d2
    else:
        cdf_neg_d1 = norm.cdf(-d1)
        cdf_neg_d2 = norm.cdf(-d2)
        value = disc_strike * strike * cdf_neg_d2 - disc_spot * spot * cdf_neg_d1
        delta = disc_spot * (cdf_d1 - 1.0)
        theta = (
            -spot * disc_spot * pdf_d1 * safe_vol / (2.0 * sqrt_t)
            + rate * strike * disc_strike * cdf_neg_d2
            - dividend * spot * disc_spot * cdf_neg_d1
        )
        rho = -strike * safe_years * disc_strike * cdf_neg_d2

    forward_price = _forward_value(spot, strike, rate, years, dividend, option)
    expired_price = _payoff(spot, strike, option)
    zero_delta, zero_theta, zero_rho = _zero_vol_greeks(
        spot, strike, rate, years, dividend, option
    )
    expired_delta = _expired_delta(spot, strike, option)

    value = np.where(expired, expired_price, np.where(zero_vol, forward_price, value))
    delta = np.where(expired, expired_delta, np.where(zero_vol, zero_delta, delta))
    gamma = np.where(live, gamma, 0.0)
    vega = np.where(live, vega, 0.0)
    theta = np.where(expired, 0.0, np.where(zero_vol, zero_theta, theta))
    rho = np.where(expired, 0.0, np.where(zero_vol, zero_rho, rho))

    return {
        "price": _scalar(value),
        "delta": _scalar(delta),
        "gamma": _scalar(gamma),
        "vega": _scalar(vega),
        "theta": _scalar(theta),
        "rho": _scalar(rho),
    }


def _value(spot, strike, rate, vol, years, dividend, option) -> np.ndarray:
    expired = years <= 0.0
    zero_vol = (vol <= 0.0) & ~expired
    live = ~expired & ~zero_vol
    safe_vol = np.where(live, vol, 1.0)
    safe_years = np.where(years > 0.0, years, 1.0)
    sqrt_t = np.sqrt(safe_years)
    sigma_sqrt = safe_vol * sqrt_t
    d1 = (
        np.log(spot / strike) + (rate - dividend + 0.5 * safe_vol**2) * safe_years
    ) / sigma_sqrt
    d2 = d1 - sigma_sqrt
    disc_spot = np.exp(-dividend * safe_years)
    disc_strike = np.exp(-rate * safe_years)
    if option == "call":
        live_value = disc_spot * spot * norm.cdf(d1) - disc_strike * strike * norm.cdf(d2)
    else:
        live_value = disc_strike * strike * norm.cdf(-d2) - disc_spot * spot * norm.cdf(-d1)
    forward_price = _forward_value(spot, strike, rate, years, dividend, option)
    expired_price = _payoff(spot, strike, option)
    return np.where(expired, expired_price, np.where(zero_vol, forward_price, live_value))


def _forward_value(spot, strike, rate, years, dividend, option) -> np.ndarray:
    forward = spot * np.exp((rate - dividend) * years)
    discount = np.exp(-rate * years)
    if option == "call":
        return discount * np.maximum(forward - strike, 0.0)
    return discount * np.maximum(strike - forward, 0.0)


def _payoff(spot, strike, option) -> np.ndarray:
    if option == "call":
        return np.maximum(spot - strike, 0.0)
    return np.maximum(strike - spot, 0.0)


def _zero_vol_greeks(spot, strike, rate, years, dividend, option):
    """Limits of the BS Greeks as volatility goes to zero, away from the kink."""
    disc_spot = np.exp(-dividend * years)
    disc_strike = np.exp(-rate * years)
    forward = spot * np.exp((rate - dividend) * years)
    above = forward > strike
    below = forward < strike
    if option == "call":
        delta = np.where(above, disc_spot, np.where(below, 0.0, 0.5 * disc_spot))
        theta = np.where(above, dividend * spot * disc_spot - rate * strike * disc_strike, 0.0)
        rho = np.where(above, strike * years * disc_strike, 0.0)
    else:
        delta = np.where(below, -disc_spot, np.where(above, 0.0, -0.5 * disc_spot))
        theta = np.where(below, rate * strike * disc_strike - dividend * spot * disc_spot, 0.0)
        rho = np.where(below, -strike * years * disc_strike, 0.0)
    return delta, theta, rho


def _expired_delta(spot, strike, option) -> np.ndarray:
    if option == "call":
        return np.where(spot > strike, 1.0, np.where(spot < strike, 0.0, 0.5))
    return np.where(spot < strike, -1.0, np.where(spot > strike, 0.0, -0.5))


def _option(option: str) -> str:
    name = str(option).lower()
    if name not in {"call", "put"}:
        raise ValueError("option must be 'call' or 'put'")
    return name


def _broadcast(*values):
    arrays = [np.asarray(value, dtype=np.float64) for value in values]
    return np.broadcast_arrays(*arrays)


def _validate(spot, strike, rate, vol, years, dividend) -> None:
    _finite_positive("spot", spot)
    _finite_positive("strike", strike)
    _finite("rate", rate)
    _finite("dividend", dividend)
    if np.any(~np.isfinite(vol) | (vol < 0.0)):
        raise ValueError("vol must be finite and non-negative")
    if np.any(~np.isfinite(years) | (years < 0.0)):
        raise ValueError("years must be finite and non-negative")


def _finite_positive(name: str, values: np.ndarray) -> None:
    if np.any(~np.isfinite(values) | (values <= 0.0)):
        raise ValueError(f"{name} must be finite and positive")


def _finite(name: str, values: np.ndarray) -> None:
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{name} must be finite")


def _scalar(values: np.ndarray) -> Number:
    if values.shape == ():
        return float(values)
    return values
