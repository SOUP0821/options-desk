import math

import numpy as np
import pytest

from options_desk.black_scholes import greeks, price
from options_desk.checks import max_finite_difference_errors
from options_desk.reference import HULL_GREEKS, PRICE_ABSOLUTE_TOLERANCE, PRICE_CASES


def test_published_prices():
    for case in PRICE_CASES:
        call = price(
            case["spot"], case["strike"], case["rate"], case["vol"], case["years"], case["dividend"], "call"
        )
        put = price(
            case["spot"], case["strike"], case["rate"], case["vol"], case["years"], case["dividend"], "put"
        )
        assert call == pytest.approx(case["call"], abs=PRICE_ABSOLUTE_TOLERANCE)
        assert put == pytest.approx(case["put"], abs=PRICE_ABSOLUTE_TOLERANCE)


def test_published_prices_satisfy_put_call_parity():
    for case in PRICE_CASES:
        forward = case["spot"] * math.exp(-case["dividend"] * case["years"]) - case["strike"] * math.exp(
            -case["rate"] * case["years"]
        )
        assert case["call"] - case["put"] == pytest.approx(forward, abs=5e-5)


def test_put_call_parity_and_shared_greeks():
    spot, strike, rate, vol, years, dividend = 100.0, 110.0, 0.03, 0.25, 0.75, 0.01
    call = greeks(spot, strike, rate, vol, years, dividend, "call")
    put = greeks(spot, strike, rate, vol, years, dividend, "put")
    cash_forward = spot * math.exp(-dividend * years) - strike * math.exp(-rate * years)
    assert call["price"] - put["price"] == pytest.approx(cash_forward, abs=1e-10)
    assert call["delta"] - put["delta"] == pytest.approx(math.exp(-dividend * years), abs=1e-10)
    assert call["gamma"] == pytest.approx(put["gamma"], abs=1e-12)
    assert call["vega"] == pytest.approx(put["vega"], abs=1e-10)


def test_hull_twenty_week_greeks():
    case = HULL_GREEKS
    values = greeks(case["spot"], case["strike"], case["rate"], case["vol"], case["years"], case["dividend"])
    tol = case["tolerance"]
    assert values["price"] == pytest.approx(case["price"], abs=tol["price"])
    assert values["delta"] == pytest.approx(case["delta"], abs=tol["delta"])
    assert values["gamma"] == pytest.approx(case["gamma"], abs=tol["gamma"])
    assert values["vega"] / 100.0 == pytest.approx(case["vega_per_point"], abs=tol["vega_per_point"])
    assert values["theta"] == pytest.approx(case["theta_per_year"], abs=tol["theta_per_year"])
    assert values["rho"] / 100.0 == pytest.approx(case["rho_per_point"], abs=tol["rho_per_point"])


def test_greeks_match_finite_differences():
    errors = max_finite_difference_errors()
    assert errors["delta"] < 1e-5
    assert errors["gamma"] < 1e-4
    assert errors["vega"] < 1e-4
    assert errors["theta"] < 1e-3
    assert errors["rho"] < 1e-3


def test_zero_vol_and_expiry_limits():
    assert price(100, 100, 0.05, 0.0, 1.0) == pytest.approx(100 * (1 - math.exp(-0.05)))
    assert price(100, 100, 0.05, 0.0, 1.0, option="put") == pytest.approx(0.0)
    assert price(80, 100, 0.05, 0.0, 1.0, option="put") == pytest.approx(100 * math.exp(-0.05) - 80)
    assert price(110, 100, 0.01, 0.2, 0.0) == 10
    assert price(90, 100, 0.01, 0.2, 0.0, option="put") == 10
    assert greeks(110, 100, 0.01, 0.2, 0.0)["delta"] == 1
    assert greeks(90, 100, 0.01, 0.2, 0.0, option="put")["delta"] == -1


def test_vector_prices_match_scalars():
    spots = np.array([90.0, 100.0, 110.0])
    got = price(spots, 100, 0.05, 0.2, 1.0, 0.01)
    for spot, value in zip(spots, got, strict=True):
        assert value == pytest.approx(price(float(spot), 100, 0.05, 0.2, 1.0, 0.01))


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        price(-1, 100, 0.05, 0.2, 1.0)
    with pytest.raises(ValueError):
        price(100, 100, 0.05, -0.1, 1.0)
    with pytest.raises(ValueError):
        price(100, 100, 0.05, 0.2, 1.0, option="straddle")
