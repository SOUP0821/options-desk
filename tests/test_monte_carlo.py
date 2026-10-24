import pytest

from options_desk.black_scholes import greeks
from options_desk.monte_carlo import monte_carlo_european
from options_desk.reference import PRICE_CASES


def test_hull_call_is_inside_sampling_error():
    case = PRICE_CASES[0]
    result = monte_carlo_european(
        case["spot"], case["strike"], case["rate"], case["vol"], case["years"], case["dividend"],
        "call", n_paths=120_000, seed=1,
    )
    assert abs(result.price - case["call"]) < 4 * result.stderr
    analytical = greeks(case["spot"], case["strike"], case["rate"], case["vol"], case["years"], case["dividend"])
    assert abs(result.delta - analytical["delta"]) < 4 * result.delta_stderr


def test_dividend_put_is_inside_sampling_error():
    case = PRICE_CASES[2]
    result = monte_carlo_european(
        case["spot"], case["strike"], case["rate"], case["vol"], case["years"], case["dividend"],
        "put", n_paths=160_000, seed=2,
    )
    assert abs(result.price - case["put"]) < 4 * result.stderr


def test_same_seed_is_reproducible():
    kwargs = dict(spot=100, strike=100, rate=0.05, vol=0.2, years=1.0, n_paths=20_000, seed=5)
    assert monte_carlo_european(**kwargs).price == monte_carlo_european(**kwargs).price
