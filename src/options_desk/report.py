"""Run the pricing checks, the live chain, and the backtest, then save the numbers."""

from __future__ import annotations

import json
import math
import warnings
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from options_desk.backtest import STRATEGIES, BacktestResult, load_spy, run_backtest
from options_desk.black_scholes import greeks, price
from options_desk.checks import max_finite_difference_errors, max_iv_roundtrip_error
from options_desk.market import load_market
from options_desk.monte_carlo import monte_carlo_european
from options_desk.reference import HULL_GREEKS, PRICE_ABSOLUTE_TOLERANCE, PRICE_CASES
from options_desk.torch_greeks import autograd_greeks


def main() -> dict:
    warnings.filterwarnings("ignore", module=r"yfinance(\..*)?")
    warnings.filterwarnings("ignore", message=r".*inplace keyword.*")
    root = project_root()
    verification = run_verification()
    _require_verification(verification)
    market = load_market()
    close = load_spy(root / "data" / "spy.csv")
    backtest = run_backtest(close)
    summary = {
        "verification": verification,
        "market": market["summary"],
        "backtest": _backtest_dict(backtest),
    }
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    market["quotes"].to_csv(out / "chain.csv", index=False)
    backtest.test_returns.to_csv(out / "test_returns.csv")
    _print_report(summary)
    return summary


def run_verification() -> dict:
    reference = []
    for case in PRICE_CASES:
        call = float(price(case["spot"], case["strike"], case["rate"], case["vol"], case["years"], case["dividend"], "call"))
        put = float(price(case["spot"], case["strike"], case["rate"], case["vol"], case["years"], case["dividend"], "put"))
        reference.append(
            {
                "source": case["source"],
                "call": call,
                "put": put,
                "call_error": call - case["call"],
                "put_error": put - case["put"],
            }
        )
    hull = greeks(
        HULL_GREEKS["spot"],
        HULL_GREEKS["strike"],
        HULL_GREEKS["rate"],
        HULL_GREEKS["vol"],
        HULL_GREEKS["years"],
        HULL_GREEKS["dividend"],
        "call",
    )
    hull_errors = {
        "price": float(hull["price"]) - HULL_GREEKS["price"],
        "delta": float(hull["delta"]) - HULL_GREEKS["delta"],
        "gamma": float(hull["gamma"]) - HULL_GREEKS["gamma"],
        "vega_per_point": float(hull["vega"]) / 100.0 - HULL_GREEKS["vega_per_point"],
        "theta_per_year": float(hull["theta"]) - HULL_GREEKS["theta_per_year"],
        "rho_per_point": float(hull["rho"]) / 100.0 - HULL_GREEKS["rho_per_point"],
    }
    first = PRICE_CASES[0]
    simulated = monte_carlo_european(
        first["spot"], first["strike"], first["rate"], first["vol"], first["years"],
        first["dividend"], "call", n_paths=200_000, seed=1,
    )
    torch_worst = {name: 0.0 for name in ("price", "delta", "gamma", "vega", "theta", "rho")}
    for case in PRICE_CASES:
        for option, published in (("call", case["call"]), ("put", case["put"])):
            auto = autograd_greeks(
                case["spot"], case["strike"], case["rate"], case["vol"], case["years"], case["dividend"], option
            )
            analytical = greeks(
                case["spot"], case["strike"], case["rate"], case["vol"], case["years"], case["dividend"], option
            )
            torch_worst["price"] = max(torch_worst["price"], abs(auto["price"] - published))
            for name in ("delta", "gamma", "vega", "theta", "rho"):
                torch_worst[name] = max(torch_worst[name], abs(auto[name] - float(analytical[name])))
    return {
        "reference": reference,
        "reference_tolerance": PRICE_ABSOLUTE_TOLERANCE,
        "hull_greeks": {name: float(hull[name]) for name in ("price", "delta", "gamma", "vega", "theta", "rho")},
        "hull_greek_errors": hull_errors,
        "finite_difference_max_abs_error": max_finite_difference_errors(),
        "iv_roundtrip_max_abs_error": max_iv_roundtrip_error(),
        "monte_carlo_hull_call": {
            "price": simulated.price,
            "stderr": simulated.stderr,
            "delta": simulated.delta,
            "delta_stderr": simulated.delta_stderr,
            "n_paths": simulated.n_paths,
            "published_price": first["call"],
            "price_gap": simulated.price - first["call"],
        },
        "torch_max_abs_error": torch_worst,
    }


def project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parents[2], Path.cwd()):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


def _require_verification(verification: dict) -> None:
    problems = []
    for row in verification["reference"]:
        if abs(row["call_error"]) > 1e-5 or abs(row["put_error"]) > 1e-5:
            problems.append(f"published price mismatch: {row}")
    fd = verification["finite_difference_max_abs_error"]
    if fd["delta"] > 1e-4 or fd["vega"] > 1e-3 or fd["theta"] > 1e-3:
        problems.append(f"finite-difference Greeks off: {fd}")
    if verification["iv_roundtrip_max_abs_error"] > 1e-6:
        problems.append("implied vol did not round-trip")
    gap = abs(verification["monte_carlo_hull_call"]["price_gap"])
    stderr = verification["monte_carlo_hull_call"]["stderr"]
    if gap > max(4 * stderr, 0.05):
        problems.append("Monte Carlo is far from the Hull call")
    if verification["torch_max_abs_error"]["price"] > 1e-5:
        problems.append("PyTorch price does not match the published cases")
    if problems:
        raise SystemExit("pricing checks failed: " + "; ".join(problems))


def _backtest_dict(result: BacktestResult) -> dict:
    return {
        "chosen_threshold": result.chosen_threshold,
        "validation_sharpe": result.validation_sharpe,
        "train_accuracy": result.train_accuracy,
        "train_base_rate": result.train_base_rate,
        "converged": result.converged,
        "n_iter": result.n_iter,
        "intercept": result.intercept,
        "coefficients": result.coefficients,
        "n_train": result.n_train,
        "n_validation": result.n_validation,
        "n_test": result.n_test,
        "fit_index_max": result.fit_index_max.date().isoformat(),
        "fit_forward_max": pd.Timestamp(result.fit_forward_max).date().isoformat(),
        "validation_index_min": result.validation_index_min.date().isoformat(),
        "validation_index_max": result.validation_index_max.date().isoformat(),
        "test_index_min": result.test_index_min.date().isoformat(),
        "test_index_max": result.test_index_max.date().isoformat(),
        "windows": result.windows,
    }


def _print_report(summary: dict) -> None:
    print("PRICING CHECKS")
    for row in summary["verification"]["reference"]:
        print(
            f"  {row['source']}: call error {row['call_error']:.3e}, put error {row['put_error']:.3e}"
        )
    hull = summary["verification"]["hull_greeks"]
    print(
        "  Hull 20-week call:"
        f" price {hull['price']:.4f}, delta {hull['delta']:.4f},"
        f" gamma {hull['gamma']:.4f}, vega/pt {hull['vega'] / 100:.4f},"
        f" theta/yr {hull['theta']:.4f}, rho/pt {hull['rho'] / 100:.4f}"
    )
    fd = summary["verification"]["finite_difference_max_abs_error"]
    print("  max |analytic - finite difference| " + ", ".join(f"{k} {v:.3e}" for k, v in fd.items()))
    print(f"  max |IV round-trip| {summary['verification']['iv_roundtrip_max_abs_error']:.3e}")
    mc = summary["verification"]["monte_carlo_hull_call"]
    print(
        f"  Monte Carlo Hull call {mc['price']:.4f} vs {mc['published_price']:.4f}"
        f" (stderr {mc['stderr']:.4f}, {mc['n_paths']} paths)"
    )
    torch_err = summary["verification"]["torch_max_abs_error"]
    print("  max |torch - reference| " + ", ".join(f"{k} {v:.3e}" for k, v in torch_err.items()))

    market = summary["market"]
    atm = market["atm"]
    print("\nMARKET")
    print(
        f"  {market['symbol']} ({market['exercise']}) spot {market['spot']:.2f} as of {market['asof']},"
        f" expiry {market['expiry']} ({market['dte']} calendar days)"
    )
    print(
        f"  continuous rate {pct(market['rate'])} from a {market['bill_discount_percent']:.3f}% bill discount;"
        f" dividend yield {pct(market['dividend_yield'])} from {market['dividend_pairs']} near-ATM pairs"
    )
    print(
        f"  {market['quote_source']} quotes: {market['n_priced']}/{market['n_quotes']} priced,"
        f" positive bids {market['n_positive_bids']},"
        f" median IV {pct(market['median_iv'])},"
        f" median Yahoo field {pct(market['median_yahoo_iv'])},"
        f" median |call IV - put IV| {points(market['median_call_put_iv_gap'])}"
    )
    if market["n_positive_bids"] == 0:
        print(
            "  Bids and asks were zero, so these are last trades. Yahoo's impliedVolatility"
            " column on that snapshot is not a second price of those trades."
        )
    print(
        f"  ATM call K={atm['strike']:.0f} quote {atm['quote']:.4f} IV {pct(atm['iv'])}"
        f" model {atm['model']:.4f} Monte Carlo {atm['monte_carlo']:.4f}"
        f" (stderr {atm['monte_carlo_stderr']:.4f})"
    )
    print(f"  max |model - quote| after inversion {market['max_abs_inversion_error']:.3e}")

    bt = summary["backtest"]
    print("\nBACKTEST")
    print(
        "  SPY total return. Train through 2014, threshold chosen on 2015-2018 at 1 bp,"
        " test from 2019. Cash earns zero. Same positions stressed at 5 bp."
    )
    print(
        f"  rows train {bt['n_train']}, validation {bt['n_validation']}, test {bt['n_test']}"
        f" ({bt['test_index_min']} to {bt['test_index_max']})"
    )
    print(
        f"  logistic threshold {bt['chosen_threshold']:.2f};"
        f" train accuracy {pct(bt['train_accuracy'])} vs base rate {pct(bt['train_base_rate'])};"
        f" converged {bt['converged']} in {bt['n_iter']} iterations"
    )
    print("  validation Sharpe by threshold: " + ", ".join(
        f"{tau} {score:.3f}" for tau, score in bt["validation_sharpe"].items()
    ))
    print("  standardized coefficients: " + ", ".join(
        f"{name} {coef:.3f}" for name, coef in bt["coefficients"].items()
    ))
    for cost_name, title in (("1bp", "Test window at 1 bp"), ("5bp", "Test window at 5 bp")):
        _print_table(title, bt["windows"]["test"], cost_name)


def _print_table(title: str, window: dict, cost_name: str) -> None:
    print(f"  {title}")
    print(f"  {'strategy':<16} {'CAGR':>8} {'vol':>8} {'Sharpe':>8} {'maxDD':>8} {'in mkt':>8} {'trades':>8} {'wealth':>8}")
    for strategy in STRATEGIES:
        row = window[strategy][cost_name]
        print(
            f"  {strategy:<16} {pct(row['cagr']):>8} {pct(row['annual_vol']):>8}"
            f" {row['sharpe']:8.2f} {pct(row['max_drawdown']):>8} {pct(row['avg_position']):>8}"
            f" {row['trades']:8.0f} {row['ending_wealth']:8.2f}"
        )


def pct(value) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def points(value) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{100.0 * float(value):.2f} vol pts"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    return value


if __name__ == "__main__":
    main()
