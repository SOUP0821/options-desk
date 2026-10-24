import numpy as np
import pandas as pd
import pytest

from options_desk.backtest import (
    COST_BPS,
    COST_BPS_STRESS,
    FEATURE_COLUMNS,
    MOMENTUM_LOOKBACK,
    MOMENTUM_SKIP,
    TEST_START,
    THRESHOLDS,
    TRAIN_END,
    VAL_END,
    build_frame,
    choose_threshold,
    net_returns,
    run_backtest,
)
from options_desk.prices import close_from_download


def test_trailing_features_ignore_future_prices_and_the_label_does_not():
    index = pd.bdate_range("2018-01-01", periods=400)
    draws = np.random.default_rng(1).normal(0.0, 0.01, len(index))
    close = pd.Series(100 * np.exp(np.cumsum(draws)), index=index)
    frame = build_frame(close)
    columns = list(FEATURE_COLUMNS) + ["mom_12_1"]
    before = frame.iloc[320][columns].copy()
    bumped = close.copy()
    bumped.iloc[321:] = bumped.iloc[321:] * 1.5
    after = build_frame(bumped)
    pd.testing.assert_series_equal(before, after.iloc[320][columns])
    assert after["fwd_ret"].iloc[320] != pytest.approx(frame["fwd_ret"].iloc[320])


def test_momentum_uses_the_twelve_one_lag():
    assert MOMENTUM_LOOKBACK == 252
    assert MOMENTUM_SKIP == 21
    index = pd.bdate_range("2020-01-01", periods=400)
    close = pd.Series(np.arange(1, 401), index=index, dtype=float)
    frame = build_frame(close)
    expected = close.iloc[300 - 21] / close.iloc[300 - 252] - 1
    assert frame["mom_12_1"].iloc[300] == pytest.approx(expected)


def test_zero_cost_long_book_matches_the_price_ratio():
    index = pd.bdate_range("2020-01-01", periods=6)
    close = pd.Series([100, 102, 101, 105, 103, 110], index=index, dtype=float)
    forward = (close.shift(-1) / close - 1).iloc[:-1]
    wealth = (1 + net_returns(pd.Series(1.0, index=forward.index), forward, 0.0)).prod()
    assert wealth == pytest.approx(close.iloc[-1] / close.iloc[0])


def test_costs_reduce_a_round_trip():
    index = pd.bdate_range("2020-01-01", periods=4)
    position = pd.Series([1.0, 1.0, 0.0, 1.0], index=index)
    forward = pd.Series(0.01, index=index)
    assert net_returns(position, forward, 10.0).sum() < net_returns(position, forward, 0.0).sum()


def test_threshold_ignores_a_test_window_that_wants_the_opposite_rule():
    validation = pd.bdate_range("2018-01-02", "2018-12-31")
    test = pd.bdate_range("2019-01-02", "2019-06-28")
    index = validation.append(test)
    probability = pd.Series(0.51, index=index)
    forward = pd.Series(-0.002, index=index)
    forward.loc[test] = 0.002
    chosen, scores = choose_threshold(probability, forward, COST_BPS)
    assert chosen == 0.55
    assert scores["0.50"] < 0
    assert scores["0.55"] == 0.0


def test_positive_validation_keeps_the_lower_threshold():
    validation = pd.bdate_range("2018-01-02", "2018-12-31")
    test = pd.bdate_range("2019-01-02", "2019-06-28")
    index = validation.append(test)
    probability = pd.Series(0.51, index=index)
    forward = pd.Series(0.002, index=index)
    forward.loc[test] = -0.01
    chosen, _scores = choose_threshold(probability, forward, COST_BPS)
    assert chosen == 0.50


def test_tied_validation_scores_take_the_higher_threshold():
    validation = pd.bdate_range("2018-01-02", "2018-12-31")
    probability = pd.Series(0.99, index=validation)
    forward = pd.Series(0.001, index=validation)
    chosen, scores = choose_threshold(probability, forward, COST_BPS)
    assert chosen == 0.55
    assert len(set(scores.values())) == 1


def test_locked_protocol_constants():
    assert TRAIN_END == "2014-12-31"
    assert VAL_END == "2018-12-31"
    assert TEST_START == "2019-01-01"
    assert THRESHOLDS == (0.50, 0.52, 0.55)
    assert COST_BPS == 1.0
    assert COST_BPS_STRESS == 5.0


def test_fit_uses_only_the_train_window_and_scaler_is_not_refit():
    result = run_backtest(_history_with_a_different_test_regime())
    assert result.fit_index_max <= pd.Timestamp("2014-12-31")
    assert result.fit_forward_max <= pd.Timestamp("2014-12-31")
    assert result.validation_index_min > pd.Timestamp("2014-12-31")
    assert result.validation_index_max <= pd.Timestamp("2018-12-31")
    assert result.test_index_min >= pd.Timestamp("2019-01-01")
    assert result.chosen_threshold in THRESHOLDS
    for name in FEATURE_COLUMNS:
        assert result.scaler_mean[name] == pytest.approx(result.train_feature_mean[name])
    assert abs(result.test_feature_mean["ret_1"] - result.train_feature_mean["ret_1"]) > 0.001
    hold = result.windows["test"]["buy_and_hold"]
    assert hold["1bp"]["trades"] == pytest.approx(1.0)
    assert hold["5bp"]["ending_wealth"] <= hold["1bp"]["ending_wealth"] + 1e-12
    momentum = result.windows["test"]["momentum_12_1"]
    assert momentum["5bp"]["ending_wealth"] <= momentum["1bp"]["ending_wealth"] + 1e-12


def test_close_parser_accepts_the_shapes_yfinance_returns():
    index = pd.bdate_range("2020-01-01", periods=3)
    wide = pd.DataFrame({"Open": [1, 1, 1], "Close": [1.0, 2.0, 3.0]}, index=index)
    assert close_from_download(wide).tolist() == [1.0, 2.0, 3.0]

    columns = pd.MultiIndex.from_product([["Close", "Open"], ["SPY"]])
    stacked = pd.DataFrame([[1.0, 9.0], [2.0, 9.0], [3.0, 9.0]], index=index, columns=columns)
    assert close_from_download(stacked).tolist() == [1.0, 2.0, 3.0]

    flipped = pd.DataFrame(
        [[1.0, 9.0], [2.0, 9.0], [3.0, 9.0]],
        index=index,
        columns=pd.MultiIndex.from_product([["SPY"], ["Close", "Open"]]),
    )
    assert close_from_download(flipped).tolist() == [1.0, 2.0, 3.0]


def _history_with_a_different_test_regime() -> pd.Series:
    index = pd.bdate_range("1998-01-01", "2021-12-31")
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0001, 0.01, len(index))
    test = index >= "2019-01-01"
    returns[test] = rng.normal(0.004, 0.01, int(test.sum()))
    return pd.Series(100 * np.exp(np.cumsum(returns)), index=index)
