"""Compare analytical Greeks with central finite differences of the price."""

from __future__ import annotations

from options_desk.black_scholes import greeks, price
from options_desk.implied_vol import implied_volatility


def finite_difference_errors(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    years: float,
    dividend: float,
    option: str,
    eps: float = 1e-4,
) -> dict[str, float]:
    analytical = greeks(spot, strike, rate, vol, years, dividend, option)

    def value(**overrides) -> float:
        args = {
            "spot": spot,
            "strike": strike,
            "rate": rate,
            "vol": vol,
            "years": years,
            "dividend": dividend,
            "option": option,
        }
        args.update(overrides)
        return float(price(**args))

    gamma_step = 1e-3
    bumped = {
        "delta": (value(spot=spot + eps) - value(spot=spot - eps)) / (2.0 * eps),
        "gamma": (
            value(spot=spot + gamma_step)
            - 2.0 * value()
            + value(spot=spot - gamma_step)
        )
        / gamma_step**2,
        "vega": (value(vol=vol + eps) - value(vol=vol - eps)) / (2.0 * eps),
        "theta": (value(years=years - eps) - value(years=years + eps)) / (2.0 * eps),
        "rho": (value(rate=rate + eps) - value(rate=rate - eps)) / (2.0 * eps),
    }
    return {name: float(analytical[name]) - bumped[name] for name in bumped}


def max_finite_difference_errors() -> dict[str, float]:
    worst = {name: 0.0 for name in ("delta", "gamma", "vega", "theta", "rho")}
    for option in ("call", "put"):
        for strike in (80.0, 100.0, 120.0):
            for dividend in (0.0, 0.03):
                errors = finite_difference_errors(
                    100.0, strike, 0.05, 0.2, 1.0, dividend, option
                )
                for name, error in errors.items():
                    worst[name] = max(worst[name], abs(error))
    return worst


def max_iv_roundtrip_error() -> float:
    """Worst volatility error on a grid where the option still has some vega."""
    worst = 0.0
    checked = 0
    for option in ("call", "put"):
        for strike in (70.0, 85.0, 100.0, 115.0, 140.0):
            for vol in (0.10, 0.20, 0.40):
                for years in (0.10, 1.0):
                    for rate in (0.01, 0.05):
                        for dividend in (0.0, 0.03):
                            premium = float(
                                price(100.0, strike, rate, vol, years, dividend, option)
                            )
                            vega = float(
                                greeks(100.0, strike, rate, vol, years, dividend, option)["vega"]
                            )
                            if premium < 1e-4 or vega < 1e-3:
                                continue
                            solved = float(
                                implied_volatility(
                                    premium, 100.0, strike, rate, years, dividend, option
                                )
                            )
                            worst = max(worst, abs(solved - vol))
                            checked += 1
    if checked == 0:
        raise RuntimeError("implied-vol grid was empty")
    return worst
