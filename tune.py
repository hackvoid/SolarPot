"""Step 2: score variants of the LightGBM price model to see which choices earn their place.

Each variant runs the same 12-month walk-forward as model.py, shared out over 24 processes.
Prints one score row per variant and blend, best first.
    .venv/bin/python tune.py             # see the round's notes for the run time
    .venv/bin/python tune.py --selftest
"""
import sys
from functools import partial

import pandas as pd

from build_dataset import TZ
from model import (add_generation_forecast, backtest_all, feature_columns, fit_predict_lasso, fit_predict_lgbm,
                   fit_predict_nn, load, scores)

# Round 1, 2026-11-14 (MAE in EUR/MWh; step 1 model 17.62, naive 35.66):
#   + own wind and solar forecast 16.27    without Yahoo commodities (cmd_) 16.06
#   rows weighted by age, half-life 2y 16.13, 1y 16.17    absolute-error loss 16.21
#   only the last 2y of rows 16.82, 3y 16.84 (spike hours much worse)
#   without capacity_ 16.36, wb_ 16.42, storage/hydro/rhine 16.42, raw weather 16.59, neighbour prices 16.63
# Round 2 starts from the own forecast without commodities (16.06) and checks combinations:
#   other random seeds 16.26, 16.27, so differences under about 0.2 are chance
#   half-life 2y 15.81 (kept in model.py)    half-life 2y + absolute-error loss 16.05
#   absolute-error loss 16.17 (negative hours 20.60 against 23.8, spike hours 48.30 against 46.7)
#   slow learning 16.11, big trees 16.09, small trees 16.30, without wb_ and capacity_ 16.41: all chance
# Round 3, 2026-11-15: forecasts ran low in the high-price autumn of 2026 (mean error -6.5 EUR/MWh in the
# last 60 days). Trees cannot forecast above what they saw, so these variants hand the model a known price
# level to start from. Result (MAE; mean error in the last 60 days):
#   base 15.78; -7.2    week mean as an input 16.05; -6.6    no squash 16.19; -1.6 (dearest 5% 43.1 against 46.2)
#   gap to yesterday's mean 17.60; -4.4    gap to week mean 18.01; +1.3    multiple of week mean 18.40; +8.1
# The lean goes, but accuracy drops by 2 or more: the starting level is itself a poor forecast after a price
# swing. Kept instead: model.recent_error, which takes the last 4 weeks' median error off the forecast.
# Round 4, 2026-11-18: inputs from ENTSO-E. Does each one lower the error? No (MAE; the test year now ends
# 2026-11-18, so the level is not comparable with the rounds above):
#   none of them 16.48, other seed 16.53    Swiss reservoir 16.43    Swiss, Austrian and French reservoirs 16.45
#   French nuclear outages 16.67    all 16.49, other seed 16.55
# All inside the chance range. They stay in: the Swiss one replaces a dead source, the others cost nothing.
# The same check with the test year ending one, two and three days earlier: with and without them within 0.05
# each time, but the level moved from 16.0 to 16.3. The weekly refit then falls on another weekday, and that
# alone shifts the score by up to 0.3. So differences under about 0.3 are chance.
# Round 5, 2026-11-18: from here on a refit for every test day (model.REFIT_DAYS), as forecast.py does. That alone
# took the same model from 16.49 to 15.48: the weekly refit had scored the model worse than it runs. A second
# seed now lands within 0.03, and the column `chance` says how big a gap must be to count.
# The question of the round: the error in the dearest hours. 353 of the 438 dearest test hours fall in
# September to November 2026, so it is mostly the price jump of that autumn. Does a second kind of model help?
# (MAE; dearest 5% of hours; 104 min for these 9 variants)
#   trees (base) 15.48; 43.8    two seeds of trees averaged 15.34; 43.1
#   neural network alone: 64 units 17.0, three of them averaged 15.90    128+64 units 17.7, two averaged 16.27
#   trees + two 128+64 networks 14.94; 41.7 (hours with a negative price 22.1 against 23.2)    <- kept in model.py
#   trees + three 64-unit networks 14.95; 42.1 (negative hours 23.9)    trees + one network 15.22 to 15.31
#   trees on unsquashed prices 15.88, blended with base 15.41    lasso in any blend: worse (16.48 with base)
#   trees with a line in each leaf (linear_tree): 48 on 25 trial days, dropped
# The gain of the blend, 0.55, is beyond chance (0.31) and four times what a second seed of trees gives (0.14).
LGBM = partial(fit_predict_lgbm, half_life_days=730, n_jobs=1)
SMALL_NN = partial(fit_predict_nn, hidden_layer_sizes=(64,), alpha=0.01)
# name: the fit_predict function. All get the columns of model.feature_columns.
VARIANTS = {
    "base": LGBM,
    "base_seed1": partial(LGBM, random_state=1),
    "raw_price": partial(LGBM, raw=True),
    "lasso": fit_predict_lasso,
    "nn64": SMALL_NN,
    "nn64_seed1": partial(SMALL_NN, random_state=1),
    "nn64_seed2": partial(SMALL_NN, random_state=2),
    "nn128": fit_predict_nn,
    "nn128_seed1": partial(fit_predict_nn, random_state=1),
}
# name: the columns whose forecasts are averaged; a blend can use the blends above it
BLENDS = {
    "base_x2": ["base", "base_seed1"],
    "nn64_x3": ["nn64", "nn64_seed1", "nn64_seed2"],
    "nn128_x2": ["nn128", "nn128_seed1"],
    "base+nn64": ["base", "nn64"],
    "base+nn64_x3": ["base", "nn64_x3"],
    "base+nn128": ["base", "nn128"],
    "base+nn128_x2": ["base", "nn128_x2"],
    "base+raw_price": ["base", "raw_price"],
    "base+lasso": ["base", "lasso"],
    "base+nn64_x3+lasso": ["base", "nn64_x3", "lasso"],
    "base+raw_price+nn64_x3": ["base", "raw_price", "nn64_x3"],
}


def level_report(pred):
    """Where round 3 looks: mean error with its sign (forecast - price), and the error at high prices."""
    error = pred.drop(columns="price").sub(pred["price"], axis=0)
    late, dear = pred.index >= pred.index[-1] - pd.Timedelta(days=60), pred["price"] > 200
    return pd.DataFrame({"bias": error.mean(), "bias_last_60d": error[late].mean(), "MAE_last_180d":
                         error[pred.index >= pred.index[-1] - pd.Timedelta(days=180)].abs().mean(),
                         "MAE_above_200": error[dear].abs().mean(),
                         "bias_when_forecast_above_200": pd.Series({c: error.loc[pred[c] > 200, c].mean() for c in error})}).round(2)


def against_base(pred):
    """Is a column's gap to the base more than chance? Per test day, its mean error minus the base's:
    the mean of that over the days, and twice its standard error. A gap smaller than `chance` is chance.
    shortcut: treats the days as independent, though hard days come in runs, so `chance` is a little too small."""
    daily = pred.drop(columns="price").sub(pred["price"], axis=0).abs().groupby(pred.index.tz_convert(TZ).normalize()).mean()
    gap = daily.sub(daily["base"], axis=0)
    return pd.DataFrame({"gap_to_base": gap.mean(), "chance": 2 * gap.std() / len(gap) ** 0.5}).round(2)


def main():
    df = add_generation_forecast(load())
    features = feature_columns(df)
    pred = backtest_all(df, {name: (fit_predict, features) for name, fit_predict in VARIANTS.items()})
    assert pred.notna().all().all(), pred.isna().sum()
    for name, parts in BLENDS.items():
        pred[name] = pred[parts].mean(axis=1)
    order = scores(pred).sort_values("MAE")
    print(order.join(against_base(pred)).join(level_report(pred)).to_string())


def _selftest():
    hours = pd.date_range("2026-01-01", periods=96, freq="h", tz=TZ)
    pred = pd.DataFrame({"price": 50.0, "base": 52.0, "same": 48.0, "worse": 55.0}, hours)
    pred.loc[hours[:24], "worse"] = 52.0  # as good as the base on the first day, 3 worse on the other three
    table = against_base(pred)
    assert table.loc["same"].tolist() == [0, 0] and table.loc["worse", "gap_to_base"] == 2.25, table
    assert table.loc["worse", "chance"] == 1.5, table  # 2 * std(0, 3, 3, 3) / sqrt(4)
    print("selftest ok")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
