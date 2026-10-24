import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from options_desk.black_scholes import price
from options_desk.market import (
    bill_discount_to_continuous,
    choose_expiry,
    price_chain,
)


def test_bill_discount_conversion():
    bill_price = 100 * (1 - 0.05 * 91 / 360)
    expected = math.log(100 / bill_price) * (365 / 91)
    assert bill_discount_to_continuous(5.0) == pytest.approx(expected)
    assert bill_discount_to_continuous(5.0) == pytest.approx(0.051021, abs=1e-5)


def test_expiry_prefers_the_listed_date_closest_to_thirty_days():
    asof = date(2026, 10, 23)
    expiry, dte = choose_expiry(
        ["2026-10-30", "2026-11-20", "2026-12-18", "2027-01-15"], asof
    )
    assert expiry == "2026-11-20"
    assert dte == 28


def test_expiry_falls_back_when_nothing_is_inside_the_window():
    asof = date(2026, 10, 23)
    expiry, dte = choose_expiry(["2026-10-30", "2027-01-21"], asof)
    assert expiry == "2027-01-21"
    assert dte >= 14


def test_chain_recovers_volatility_and_dividend_from_mids():
    calls, puts = _quoted_chain(use_last=False, zero_call_strike=100.0)
    table, summary = price_chain(calls, puts, spot=100.0, rate=0.04, years=1.0)
    assert summary["quote_source"] == "mid"
    assert summary["n_positive_bids"] > 0
    assert summary["dividend_yield"] == pytest.approx(0.015, abs=1e-8)
    assert summary["dividend_pairs"] == 2
    assert table["iv"].dropna().tolist() == pytest.approx([0.22] * summary["n_priced"], abs=1e-8)
    assert summary["median_yahoo_gap"] == pytest.approx(0.0, abs=1e-6)
    assert summary["max_abs_inversion_error"] < 1e-8
    assert table["strike"].min() >= 85
    assert table["strike"].max() <= 115
    assert 100.0 not in set(table.loc[table["option"] == "call", "strike"])


def test_chain_uses_last_trades_when_bids_are_missing():
    calls, puts = _quoted_chain(use_last=True, zero_call_strike=None)
    _table, summary = price_chain(calls, puts, spot=100.0, rate=0.04, years=1.0)
    assert summary["quote_source"] == "last"
    assert summary["n_positive_bids"] == 0
    assert summary["dividend_yield"] == pytest.approx(0.015, abs=1e-8)
    assert summary["median_iv"] == pytest.approx(0.22, abs=1e-8)


def _quoted_chain(use_last: bool, zero_call_strike: float | None):
    spot, rate, years, vol, dividend = 100.0, 0.04, 1.0, 0.22, 0.015
    strikes = [50.0, *list(np.linspace(85, 115, 9)), 150.0]
    frames = {}
    for option in ("call", "put"):
        rows = []
        for strike in strikes:
            premium = float(price(spot, strike, rate, vol, years, dividend, option))
            wiped = use_last or (option == "call" and strike == zero_call_strike)
            rows.append(
                {
                    "strike": strike,
                    "bid": 0.0 if wiped else premium,
                    "ask": 0.0 if wiped else premium,
                    "lastPrice": premium,
                    "impliedVolatility": vol,
                }
            )
        frames[option] = pd.DataFrame(rows)
    return frames["call"], frames["put"]
