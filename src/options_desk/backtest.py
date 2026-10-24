"""SPY backtest.

Fit through 2014, pick the threshold on 2015-2018, score 2019 once.
Signal at the close, earn the next session. Cash earns 0.
1 bp is the cost used to pick the threshold; 5 bp just revalues those positions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from options_desk.prices import close_from_download, retry

TRAIN_END = "2014-12-31"
VAL_END = "2018-12-31"
TEST_START = "2019-01-01"
THRESHOLDS = (0.50, 0.52, 0.55)
COST_BPS = 1.0
COST_BPS_STRESS = 5.0
COST_LEVELS = (("1bp", COST_BPS), ("5bp", COST_BPS_STRESS))
MOMENTUM_LOOKBACK = 252
MOMENTUM_SKIP = 21
FEATURE_COLUMNS = ("ret_1", "ret_5", "ret_21", "ret_63", "vol_21", "dist_ma_21")
STRATEGIES = ("buy_and_hold", "momentum_12_1", "logistic")


@dataclass
class BacktestResult:
    chosen_threshold: float
    validation_sharpe: dict[str, float]
    train_accuracy: float
    train_base_rate: float
    converged: bool
    n_iter: int
    intercept: float
    coefficients: dict[str, float]
    scaler_mean: dict[str, float]
    train_feature_mean: dict[str, float]
    test_feature_mean: dict[str, float]
    n_train: int
    n_validation: int
    n_test: int
    fit_index_max: pd.Timestamp
    fit_forward_max: pd.Timestamp
    validation_index_min: pd.Timestamp
    validation_index_max: pd.Timestamp
    test_index_min: pd.Timestamp
    test_index_max: pd.Timestamp
    windows: dict
    test_returns: pd.DataFrame


def build_frame(close: pd.Series) -> pd.DataFrame:
    close = close.astype(float).sort_index()
    if getattr(close.index, "tz", None) is not None:
        close = close.copy()
        close.index = close.index.tz_localize(None)
    if close.index.has_duplicates:
        raise ValueError("close index has duplicate dates")
    returns = close.pct_change()
    frame = pd.DataFrame(index=close.index)
    frame["close"] = close
    frame["ret_1"] = returns
    frame["ret_5"] = close.pct_change(5)
    frame["ret_21"] = close.pct_change(21)
    frame["ret_63"] = close.pct_change(63)
    frame["vol_21"] = returns.rolling(21).std(ddof=1)
    frame["dist_ma_21"] = close / close.rolling(21).mean() - 1.0
    frame["mom_12_1"] = close.shift(MOMENTUM_SKIP) / close.shift(MOMENTUM_LOOKBACK) - 1.0
    frame["fwd_ret"] = close.shift(-1) / close - 1.0
    frame["fwd_date"] = pd.Series(close.index, index=close.index).shift(-1)
    return frame


def sharpe(daily: pd.Series) -> float:
    """Annualized Sharpe with a zero risk-free rate. Zero volatility is 0 if the mean is 0."""
    values = daily.dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return 0.0
    deviation = float(values.std(ddof=1))
    mean = float(values.mean())
    if deviation == 0.0:
        if mean == 0.0:
            return 0.0
        return math.copysign(math.inf, mean)
    return mean / deviation * math.sqrt(252.0)


def net_returns(position: pd.Series, forward_return: pd.Series, cost_bps: float) -> pd.Series:
    """Hold ``position`` over the next session and charge ``cost_bps`` on turnover.

    The book starts flat, so the first non-zero position pays to enter.
    """
    pos = position.astype(float)
    fwd = forward_return.astype(float)
    if not pos.index.equals(fwd.index):
        raise ValueError("position and forward return indexes differ")
    previous = pos.shift(1).fillna(0.0)
    turnover = (pos - previous).abs()
    cost = turnover * (cost_bps / 10_000.0)
    net = pos * fwd - cost
    net.name = "net"
    return net


def choose_threshold(
    probability: pd.Series,
    forward_return: pd.Series,
    cost_bps: float = COST_BPS,
) -> tuple[float, dict[str, float]]:
    """Pick the validation threshold with the highest Sharpe. Ties take the higher threshold."""
    aligned = pd.DataFrame({"p": probability, "r": forward_return}).dropna()
    validation = aligned.loc[
        (aligned.index > pd.Timestamp(TRAIN_END)) & (aligned.index <= pd.Timestamp(VAL_END))
    ]
    if len(validation) < 20:
        raise ValueError(f"validation window has {len(validation)} rows, need at least 20")
    scores: dict[str, float] = {}
    best_tau = THRESHOLDS[0]
    best_score = -math.inf
    for tau in THRESHOLDS:
        position = (validation["p"] >= tau).astype(float)
        score = sharpe(net_returns(position, validation["r"], cost_bps))
        scores[f"{tau:.2f}"] = score
        if _better(score, tau, best_score, best_tau):
            best_score = score
            best_tau = tau
    return best_tau, scores


def run_backtest(close: pd.Series) -> BacktestResult:
    frame = build_frame(close)
    train, validation, test = _split_masks(frame)
    if int(train.sum()) < 252 or int(validation.sum()) < 20 or int(test.sum()) < 20:
        raise ValueError("price history does not cover the locked windows")

    pipeline = _pipeline()
    x_train = frame.loc[train, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    y_train = (frame.loc[train, "fwd_ret"] > 0.0).to_numpy(dtype=int)
    pipeline.fit(x_train, y_train)

    usable = train | validation | test
    probability = pd.Series(np.nan, index=frame.index, dtype=float)
    probability.loc[usable] = pipeline.predict_proba(
        frame.loc[usable, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    )[:, 1]
    chosen, scores = choose_threshold(probability, frame["fwd_ret"], COST_BPS)

    positions = {
        "buy_and_hold": pd.Series(1.0, index=frame.index),
        "momentum_12_1": (frame["mom_12_1"] > 0.0).astype(float),
        "logistic": (probability >= chosen).astype(float),
    }
    masks = {"train": train, "validation": validation, "test": test}
    windows: dict = {}
    for window_name, mask in masks.items():
        dates = frame.index[mask]
        windows[window_name] = {}
        for strategy, position in positions.items():
            windows[window_name][strategy] = {}
            for cost_name, cost in COST_LEVELS:
                windows[window_name][strategy][cost_name] = window_metrics(
                    position.loc[dates], frame.loc[dates, "fwd_ret"], cost
                )

    test_dates = frame.index[test]
    test_returns = pd.DataFrame(
        {
            "spy": frame.loc[test_dates, "fwd_ret"],
            "buy_and_hold": net_returns(
                positions["buy_and_hold"].loc[test_dates],
                frame.loc[test_dates, "fwd_ret"],
                COST_BPS,
            ),
            "momentum_12_1": net_returns(
                positions["momentum_12_1"].loc[test_dates],
                frame.loc[test_dates, "fwd_ret"],
                COST_BPS,
            ),
            "logistic": net_returns(
                positions["logistic"].loc[test_dates],
                frame.loc[test_dates, "fwd_ret"],
                COST_BPS,
            ),
        }
    )

    model = pipeline.named_steps["model"]
    scaler = pipeline.named_steps["scaler"]
    n_iter = int(np.asarray(model.n_iter_).ravel()[0])
    predicted = pipeline.predict(x_train)
    return BacktestResult(
        chosen_threshold=float(chosen),
        validation_sharpe=scores,
        train_accuracy=float((predicted == y_train).mean()),
        train_base_rate=float(y_train.mean()),
        converged=n_iter < 2000,
        n_iter=n_iter,
        intercept=float(np.asarray(model.intercept_).ravel()[0]),
        coefficients={
            name: float(coef)
            for name, coef in zip(FEATURE_COLUMNS, np.asarray(model.coef_).ravel(), strict=True)
        },
        scaler_mean={
            name: float(value)
            for name, value in zip(FEATURE_COLUMNS, np.asarray(scaler.mean_), strict=True)
        },
        train_feature_mean={
            name: float(frame.loc[train, name].mean()) for name in FEATURE_COLUMNS
        },
        test_feature_mean={
            name: float(frame.loc[test, name].mean()) for name in FEATURE_COLUMNS
        },
        n_train=int(train.sum()),
        n_validation=int(validation.sum()),
        n_test=int(test.sum()),
        fit_index_max=pd.Timestamp(frame.index[train].max()),
        fit_forward_max=pd.Timestamp(frame.loc[train, "fwd_date"].max()),
        validation_index_min=pd.Timestamp(frame.index[validation].min()),
        validation_index_max=pd.Timestamp(frame.index[validation].max()),
        test_index_min=pd.Timestamp(frame.index[test].min()),
        test_index_max=pd.Timestamp(frame.index[test].max()),
        windows=windows,
        test_returns=test_returns,
    )


def window_metrics(position: pd.Series, forward_return: pd.Series, cost_bps: float) -> dict:
    net = net_returns(position, forward_return, cost_bps)
    gross = net_returns(position, forward_return, 0.0)
    wealth = (1.0 + net).cumprod()
    ending = float(wealth.iloc[-1])
    days = int(len(net))
    deviation = float(net.std(ddof=1))
    changes = position.astype(float).diff().abs()
    changes.iloc[0] = abs(float(position.iloc[0]))
    return {
        "first_signal": pd.Timestamp(net.index[0]).date().isoformat(),
        "last_signal": pd.Timestamp(net.index[-1]).date().isoformat(),
        "days": days,
        "ending_wealth": ending,
        "total_return": ending - 1.0,
        "cagr": ending ** (252.0 / days) - 1.0 if ending > 0.0 else float("nan"),
        "annual_vol": deviation * math.sqrt(252.0),
        "sharpe": sharpe(net),
        "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
        "avg_position": float(position.astype(float).mean()),
        "trades": float(changes.sum()),
        "total_cost": float((gross - net).sum()),
    }


def load_spy(cache_path: Path | None = None) -> pd.Series:
    import yfinance as yf

    try:
        frame = retry(
            lambda: yf.download(
                "SPY",
                start="1998-01-01",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        )
        series = close_from_download(frame)
    except Exception:
        if cache_path is not None and cache_path.exists():
            series = close_from_download(pd.read_csv(cache_path, index_col=0, parse_dates=True))
        else:
            raise
    else:
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            series.to_csv(cache_path, header=["close"])
    if series.index.min() > pd.Timestamp("1999-06-01"):
        raise RuntimeError("SPY history does not start early enough for the train window")
    if series.index.max() < pd.Timestamp("2020-01-01"):
        raise RuntimeError("SPY history does not cover the test window")
    return series


def _pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    l1_ratio=0.0,
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=0,
                ),
            ),
        ]
    )


def _split_masks(frame: pd.DataFrame):
    features_ok = frame[list(FEATURE_COLUMNS)].notna().all(axis=1)
    usable = (
        features_ok
        & frame["fwd_ret"].notna()
        & frame["fwd_date"].notna()
        & frame["mom_12_1"].notna()
    )
    train_end = pd.Timestamp(TRAIN_END)
    val_end = pd.Timestamp(VAL_END)
    test_start = pd.Timestamp(TEST_START)
    train = usable & (frame.index <= train_end) & (frame["fwd_date"] <= train_end)
    validation = (
        usable
        & (frame.index > train_end)
        & (frame.index <= val_end)
        & (frame["fwd_date"] <= val_end)
    )
    test = usable & (frame.index >= test_start)
    return train, validation, test


def _better(score: float, tau: float, best_score: float, best_tau: float) -> bool:
    if not math.isfinite(score) or not math.isfinite(best_score):
        if score == best_score:
            return tau > best_tau
        return score > best_score
    if score > best_score + 1e-12:
        return True
    return abs(score - best_score) <= 1e-12 and tau > best_tau
