import pytest

from options_desk.black_scholes import greeks
from options_desk.reference import PRICE_CASES
from options_desk.torch_greeks import autograd_greeks


def test_autograd_matches_published_prices_and_analytical_greeks():
    points = list(PRICE_CASES) + [
        {
            "spot": 49.0,
            "strike": 50.0,
            "rate": 0.05,
            "dividend": 0.0,
            "years": 20 / 52,
            "vol": 0.20,
            "call": None,
            "put": None,
        }
    ]
    for case in points:
        for option in ("call", "put"):
            auto = autograd_greeks(
                case["spot"], case["strike"], case["rate"], case["vol"], case["years"], case["dividend"], option
            )
            analytical = greeks(
                case["spot"], case["strike"], case["rate"], case["vol"], case["years"], case["dividend"], option
            )
            published = case[option]
            if published is not None:
                assert auto["price"] == pytest.approx(published, abs=5e-6)
            for name in ("price", "delta", "gamma", "vega", "theta", "rho"):
                assert auto[name] == pytest.approx(analytical[name], abs=1e-6)
