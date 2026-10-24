"""Same BS formula in torch, so autograd can check the closed-form Greeks."""

from __future__ import annotations

import torch

from options_desk.black_scholes import _option


def autograd_greeks(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    years: float,
    dividend: float = 0.0,
    option: str = "call",
) -> dict[str, float]:
    option = _option(option)
    if vol <= 0.0 or years <= 0.0:
        raise ValueError("autograd Greeks are for positive vol and time")
    spot_t = torch.tensor(float(spot), dtype=torch.float64, requires_grad=True)
    rate_t = torch.tensor(float(rate), dtype=torch.float64, requires_grad=True)
    vol_t = torch.tensor(float(vol), dtype=torch.float64, requires_grad=True)
    years_t = torch.tensor(float(years), dtype=torch.float64, requires_grad=True)
    strike_t = torch.tensor(float(strike), dtype=torch.float64)
    dividend_t = torch.tensor(float(dividend), dtype=torch.float64)
    value = _black_scholes_torch(
        spot_t, strike_t, rate_t, vol_t, years_t, dividend_t, option
    )
    delta, vega, dvalue_dyears, rho = torch.autograd.grad(
        value, [spot_t, vol_t, years_t, rate_t], create_graph=True
    )
    gamma = torch.autograd.grad(delta, spot_t)[0]
    return {
        "price": float(value.detach()),
        "delta": float(delta.detach()),
        "gamma": float(gamma.detach()),
        "vega": float(vega.detach()),
        "theta": float((-dvalue_dyears).detach()),
        "rho": float(rho.detach()),
    }


def _black_scholes_torch(spot, strike, rate, vol, years, dividend, option):
    sqrt_t = torch.sqrt(years)
    sigma_sqrt = vol * sqrt_t
    d1 = (torch.log(spot / strike) + (rate - dividend + 0.5 * vol * vol) * years) / sigma_sqrt
    d2 = d1 - sigma_sqrt
    disc_spot = torch.exp(-dividend * years)
    disc_strike = torch.exp(-rate * years)
    if option == "call":
        return disc_spot * spot * _cdf(d1) - disc_strike * strike * _cdf(d2)
    return disc_strike * strike * _cdf(-d2) - disc_spot * spot * _cdf(-d1)


def _cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x * (0.5**0.5)))
