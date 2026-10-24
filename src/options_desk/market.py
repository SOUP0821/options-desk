"""Implied vols from an SPX (or SPY) chain.

Mids if at least 8 calls inside 80-120% of spot have a real quote
(bid > 0, spread not wider than half the mid). Otherwise last trades.
q is the median parity yield within 5% of spot. T is calendar days / 365.
Rate is the 13-week bill, taken off the discount quote.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from options_desk.black_scholes import greeks, price
from options_desk.implied_vol import implied_volatility
from options_desk.monte_carlo import monte_carlo_european
from options_desk.prices import close_from_download, retry

MONEYNESS_MIN = 0.80
MONEYNESS_MAX = 1.20
PARITY_BAND = 0.05
MIN_MID = 0.05
MAX_REL_SPREAD = 0.50
MIN_QUOTES = 8
TARGET_DTE = 30
MIN_DTE = 20
MAX_DTE = 60
FALLBACK_MIN_DTE = 14
NEW_YORK = ZoneInfo("America/New_York")


def bill_discount_to_continuous(discount_percent: float, days: int = 91) -> float:
    """Convert a bank-discount bill quote in percent to a continuous yield.

    Treasury bills are quoted on a 360-day discount basis. The continuous
    rate is ln(face / price) scaled by 365/days.
    """
    discount = float(discount_percent) / 100.0
    bill_price = 100.0 * (1.0 - discount * days / 360.0)
    if not 0.0 < bill_price < 100.0:
        raise ValueError(f"bill price {bill_price} is outside (0, 100)")
    return math.log(100.0 / bill_price) * (365.0 / days)


def choose_expiry(expiries: list[str] | tuple[str, ...], asof: date) -> tuple[str, int]:
    """Pick the listed expiry closest to 30 days, preferring the 20–60 day window."""
    parsed: list[tuple[int, str]] = []
    for text in expiries:
        day = datetime.strptime(text, "%Y-%m-%d").date()
        dte = (day - asof).days
        if dte > 0:
            parsed.append((dte, text))
    window = [item for item in parsed if MIN_DTE <= item[0] <= MAX_DTE]
    pool = window or [item for item in parsed if item[0] >= FALLBACK_MIN_DTE]
    if not pool:
        raise RuntimeError("no listed expiry with enough time left")
    pool.sort(key=lambda item: (abs(item[0] - TARGET_DTE), item[0]))
    return pool[0][1], pool[0][0]


def price_chain(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot: float,
    rate: float,
    years: float,
) -> tuple[pd.DataFrame, dict]:
    """Price a call/put chain and return the quote table plus summary stats."""
    if years <= 0.0:
        raise ValueError("years must be positive")
    if spot <= 0.0:
        raise ValueError("spot must be positive")
    listed = _long_chain(calls, puts)
    if listed.empty or not (listed["strike"].min() <= spot <= listed["strike"].max()):
        low = None if listed.empty else float(listed["strike"].min())
        high = None if listed.empty else float(listed["strike"].max())
        raise RuntimeError(f"spot {spot} is outside the strike range {low}–{high}")
    table, source = _select_quotes(listed, spot)
    dividend, pairs = implied_dividend_yield(table, spot, rate, years)
    table = _attach_model(table, spot, rate, years, dividend)
    return table, _summary(table, dividend, pairs, source)


def implied_dividend_yield(
    table: pd.DataFrame, spot: float, rate: float, years: float
) -> tuple[float, int]:
    """Median parity dividend yield from near-the-money call/put quotes."""
    calls = table.loc[table["option"] == "call", ["strike", "quote"]]
    puts = table.loc[table["option"] == "put", ["strike", "quote"]]
    merged = calls.merge(puts, on="strike", suffixes=("_call", "_put"))
    near = (merged["strike"] / spot - 1.0).abs() <= PARITY_BAND
    yields = []
    for row in merged.loc[near].itertuples(index=False):
        prepaid = row.quote_call - row.quote_put + row.strike * math.exp(-rate * years)
        if prepaid > 0.0:
            yields.append(-math.log(prepaid / spot) / years)
    if not yields:
        raise RuntimeError("no near-the-money pair produced a dividend yield")
    return float(np.median(yields)), len(yields)


def atm_monte_carlo(
    table: pd.DataFrame,
    spot: float,
    rate: float,
    years: float,
    dividend: float,
    n_paths: int = 300_000,
    seed: int = 7,
) -> dict:
    """Reprice the call nearest the spot at its implied volatility."""
    calls = table[(table["option"] == "call") & np.isfinite(table["iv"])]
    if calls.empty:
        raise RuntimeError("no priced calls")
    position = int((calls["strike"] - spot).abs().to_numpy().argmin())
    row = calls.iloc[position]
    strike = float(row["strike"])
    vol = float(row["iv"])
    simulated = monte_carlo_european(
        spot, strike, rate, vol, years, dividend, "call", n_paths=n_paths, seed=seed
    )
    model = float(price(spot, strike, rate, vol, years, dividend, "call"))
    return {
        "strike": strike,
        "quote": float(row["quote"]),
        "iv": vol,
        "model": model,
        "monte_carlo": simulated.price,
        "monte_carlo_stderr": simulated.stderr,
        "monte_carlo_paths": simulated.n_paths,
        "inversion_error": float(row["inversion_error"]),
    }


def load_market(n_paths: int = 300_000) -> dict:
    """Fetch SPX (or SPY if SPX is unavailable), invert vols, and simulate the ATM call."""
    symbol, exercise, ticker, expiries = _find_ticker()
    asof_guess = datetime.now(NEW_YORK).date()
    expiry, _dte = choose_expiry(expiries, asof_guess)
    chain = retry(lambda: ticker.option_chain(expiry))
    if chain.calls is None or chain.puts is None or len(chain.calls) == 0:
        raise RuntimeError(f"{symbol} returned an empty option chain for {expiry}")
    spot, asof = _spot_and_date(chain.underlying, ticker)
    resolved, dte = choose_expiry(expiries, asof)
    if resolved != expiry:
        expiry = resolved
        chain = retry(lambda: ticker.option_chain(expiry))
        spot, asof = _spot_and_date(chain.underlying, ticker)
        expiry, dte = choose_expiry(expiries, asof)
    years = dte / 365.0
    rate, bill_percent = _bill_rate()
    table, summary = price_chain(chain.calls, chain.puts, spot, rate, years)
    atm = atm_monte_carlo(
        table, spot, rate, years, summary["dividend_yield"], n_paths=n_paths
    )
    summary.update(
        {
            "symbol": symbol,
            "exercise": exercise,
            "spot": spot,
            "asof": asof.isoformat(),
            "expiry": expiry,
            "dte": dte,
            "years": years,
            "rate": rate,
            "bill_discount_percent": bill_percent,
            "atm": atm,
        }
    )
    return {"summary": summary, "quotes": table}


def _find_ticker():
    import yfinance as yf

    errors = []
    for symbol in ("^SPX", "SPY"):
        try:
            ticker = yf.Ticker(symbol)
            expiries = list(retry(lambda ticker=ticker: ticker.options) or [])
            if expiries:
                exercise = "european" if symbol == "^SPX" else "american"
                return symbol, exercise, ticker, expiries
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")
    raise RuntimeError("no option chain found; " + "; ".join(errors))


def _spot_and_date(underlying, ticker) -> tuple[float, date]:
    spot = None
    asof = None
    if isinstance(underlying, dict):
        raw = underlying.get("regularMarketPrice")
        if raw is not None and float(raw) > 0.0:
            spot = float(raw)
        stamp = underlying.get("regularMarketTime")
        if stamp:
            asof = datetime.fromtimestamp(int(stamp), NEW_YORK).date()
    if spot is None or asof is None:
        history = retry(lambda: ticker.history(period="5d", auto_adjust=False))
        series = close_from_download(history)
        if spot is None:
            spot = float(series.iloc[-1])
        if asof is None:
            asof = series.index[-1].date()
    return spot, asof


def _bill_rate() -> tuple[float, float]:
    import yfinance as yf

    frame = retry(
        lambda: yf.download(
            "^IRX", period="1mo", auto_adjust=True, progress=False, threads=False
        )
    )
    series = close_from_download(frame)
    raw = float(series.iloc[-1])
    if not 0.0 < raw < 25.0:
        raise RuntimeError(f"^IRX quote {raw} is outside 0–25 percent")
    return bill_discount_to_continuous(raw), raw


def _long_chain(calls: pd.DataFrame, puts: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [_one_side(calls, "call"), _one_side(puts, "put")], ignore_index=True
    )


def _one_side(frame: pd.DataFrame, option: str) -> pd.DataFrame:
    df = frame.copy()
    df.columns = [str(column).lower() for column in df.columns]
    missing = {"strike", "bid", "ask"} - set(df.columns)
    if missing:
        raise RuntimeError(f"{option} chain is missing {sorted(missing)}")
    out = pd.DataFrame(
        {
            "option": option,
            "strike": pd.to_numeric(df["strike"], errors="coerce"),
            "bid": pd.to_numeric(df["bid"], errors="coerce"),
            "ask": pd.to_numeric(df["ask"], errors="coerce"),
            "last": pd.to_numeric(df["lastprice"], errors="coerce")
            if "lastprice" in df.columns
            else np.nan,
            "yahoo_iv": pd.to_numeric(df["impliedvolatility"], errors="coerce")
            if "impliedvolatility" in df.columns
            else np.nan,
        }
    )
    out.loc[out["yahoo_iv"] <= 0.0, "yahoo_iv"] = np.nan
    return out.dropna(subset=["strike"])


def _select_quotes(listed: pd.DataFrame, spot: float) -> tuple[pd.DataFrame, str]:
    band = listed[
        (listed["strike"] >= MONEYNESS_MIN * spot)
        & (listed["strike"] <= MONEYNESS_MAX * spot)
    ].copy()
    mid = (band["bid"] + band["ask"]) / 2.0
    relative_spread = (band["ask"] - band["bid"]) / mid.where(mid > 0.0)
    mid_ok = (
        (band["bid"] > 0.0)
        & (band["ask"] >= band["bid"])
        & (mid >= MIN_MID)
        & (relative_spread <= MAX_REL_SPREAD)
    )
    n_call_mids = int(((band["option"] == "call") & mid_ok).sum())
    if n_call_mids >= MIN_QUOTES:
        chosen = band.loc[mid_ok].copy()
        chosen["quote"] = mid.loc[mid_ok].to_numpy()
        chosen["rel_spread"] = relative_spread.loc[mid_ok].to_numpy()
        return chosen, "mid"
    last_ok = band["last"].notna() & (band["last"] >= MIN_MID)
    n_call_last = int(((band["option"] == "call") & last_ok).sum())
    if n_call_last >= MIN_QUOTES:
        chosen = band.loc[last_ok].copy()
        chosen["quote"] = chosen["last"].to_numpy()
        chosen["rel_spread"] = np.nan
        return chosen, "last"
        raise RuntimeError(
            f"not enough quotes inside {MONEYNESS_MIN:.0%}-{MONEYNESS_MAX:.0%} of spot "
            f"(call mids={n_call_mids}, call lasts={n_call_last})"
        )


def _attach_model(
    table: pd.DataFrame, spot: float, rate: float, years: float, dividend: float
) -> pd.DataFrame:
    table = table.copy()
    table["iv"] = np.nan
    table["model"] = np.nan
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        table[name] = np.nan
    for option in ("call", "put"):
        mask = table["option"] == option
        if not bool(mask.any()):
            continue
        solved = implied_volatility(
            table.loc[mask, "quote"].to_numpy(),
            spot,
            table.loc[mask, "strike"].to_numpy(),
            rate,
            years,
            dividend,
            option,
        )
        table.loc[mask, "iv"] = np.asarray(solved, dtype=float)
        finite = mask & np.isfinite(table["iv"])
        if not bool(finite.any()):
            continue
        strike = table.loc[finite, "strike"].to_numpy()
        vol = table.loc[finite, "iv"].to_numpy()
        table.loc[finite, "model"] = np.asarray(
            price(spot, strike, rate, vol, years, dividend, option), dtype=float
        )
        sensitivities = greeks(spot, strike, rate, vol, years, dividend, option)
        for name in ("delta", "gamma", "vega", "theta", "rho"):
            table.loc[finite, name] = np.asarray(sensitivities[name], dtype=float)
    table["inversion_error"] = table["model"] - table["quote"]
    return table.sort_values(["option", "strike"]).reset_index(drop=True)


def _summary(table: pd.DataFrame, dividend: float, pairs: int, source: str) -> dict:
    priced = table["iv"].dropna()
    both = table.dropna(subset=["iv", "yahoo_iv"])
    yahoo_gap = None
    if not both.empty:
        yahoo_gap = float((both["iv"] - both["yahoo_iv"]).abs().median())
    inversion = table["inversion_error"].dropna()
    yahoo = table["yahoo_iv"].dropna()
    return {
        "quote_source": source,
        "dividend_yield": dividend,
        "dividend_pairs": pairs,
        "n_quotes": int(len(table)),
        "n_priced": int(len(priced)),
        "n_positive_bids": int((table["bid"] > 0).sum()),
        "median_iv": float(priced.median()) if len(priced) else None,
        "median_yahoo_iv": float(yahoo.median()) if len(yahoo) else None,
        "median_yahoo_gap": yahoo_gap,
        "median_call_put_iv_gap": _call_put_gap(table),
        "max_abs_inversion_error": float(inversion.abs().max()) if len(inversion) else None,
    }


def _call_put_gap(table: pd.DataFrame) -> float | None:
    calls = table.loc[table["option"] == "call", ["strike", "iv"]].dropna()
    puts = table.loc[table["option"] == "put", ["strike", "iv"]].dropna()
    merged = calls.merge(puts, on="strike", suffixes=("_call", "_put"))
    if merged.empty:
        return None
    return float((merged["iv_call"] - merged["iv_put"]).abs().median())
