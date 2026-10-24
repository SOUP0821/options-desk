import math

import numpy as np
import pytest

from options_desk.black_scholes import price
from options_desk.checks import max_iv_roundtrip_error
from options_desk.implied_vol import implied_volatility


def test_roundtrip_recovers_the_input_volatility():
    assert max_iv_roundtrip_error() < 1e-8


def test_hull_price_inverts_near_twenty_vol():
    exact = float(price(42, 40, 0.10, 0.20, 0.5))
    assert implied_volatility(exact, 42, 40, 0.10, 0.5) == pytest.approx(0.20, abs=1e-10)
    # A premium rounded to the cent, the way a textbook quote is often written.
    assert implied_volatility(4.76, 42, 40, 0.10, 0.5) == pytest.approx(0.20, abs=1e-3)


def test_vector_inversion():
    strikes = [90.0, 100.0, 110.0]
    premiums = [float(price(100, strike, 0.05, 0.2, 1.0, 0.02)) for strike in strikes]
    solved = implied_volatility(premiums, 100, strikes, 0.05, 1.0, 0.02)
    assert solved == pytest.approx([0.2, 0.2, 0.2], abs=1e-10)


def test_impossible_premiums_are_nan():
    assert math.isnan(implied_volatility(0.0, 100, 100, 0.05, 1.0))
    assert math.isnan(implied_volatility(1000.0, 100, 100, 0.05, 1.0))
    assert math.isnan(implied_volatility(1.0, 100, 100, 0.05, 0.0))
    assert implied_volatility(np.array([0.0, 1000.0]), 100, 100, 0.05, 1.0).shape == (2,)
