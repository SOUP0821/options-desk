"""European BS Monte Carlo. Antithetic pairs. Delta is the pathwise estimator."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from options_desk.black_scholes import _option


@dataclass(frozen=True)
class MonteCarloResult:
    price: float
    stderr: float
    delta: float
    delta_stderr: float
    n_paths: int


def monte_carlo_european(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    years: float,
    dividend: float = 0.0,
    option: str = "call",
    n_paths: int = 200_000,
    seed: int = 0,
) -> MonteCarloResult:
    option = _option(option)
    spot = _scalar("spot", spot)
    strike = _scalar("strike", strike)
    rate = _scalar("rate", rate)
    vol = _scalar("vol", vol)
    years = _scalar("years", years)
    dividend = _scalar("dividend", dividend)
    if spot <= 0.0 or strike <= 0.0:
        raise ValueError("spot and strike must be positive")
    if vol < 0.0 or years <= 0.0:
        raise ValueError("vol must be non-negative and years must be positive")
    if n_paths < 1000:
        raise ValueError("n_paths must be at least 1000")
    n_paths = int(n_paths) - int(n_paths) % 2
    n_half = n_paths // 2

    draws = np.random.default_rng(seed).standard_normal(n_half)
    shocks = np.concatenate([draws, -draws])
    terminal = spot * np.exp(
        (rate - dividend - 0.5 * vol * vol) * years + vol * math.sqrt(years) * shocks
    )
    discount = math.exp(-rate * years)
    if option == "call":
        payoff = np.maximum(terminal - strike, 0.0)
        delta_path = discount * (terminal / spot) * (terminal > strike)
    else:
        payoff = np.maximum(strike - terminal, 0.0)
        delta_path = -discount * (terminal / spot) * (terminal < strike)
    price, stderr = _mean_se(discount * payoff, n_half)
    delta, delta_stderr = _mean_se(delta_path, n_half)
    return MonteCarloResult(price, stderr, delta, delta_stderr, n_paths)


def _mean_se(samples: np.ndarray, n_half: int) -> tuple[float, float]:
    paired = 0.5 * (samples[:n_half] + samples[n_half:])
    mean = float(paired.mean())
    stderr = float(paired.std(ddof=1) / math.sqrt(n_half))
    return mean, stderr


def _scalar(name: str, value: float) -> float:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar")
    out = float(array)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out
