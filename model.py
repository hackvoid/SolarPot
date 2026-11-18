"""Score day-ahead price forecasts on the last 12 months of data/training.db.

naive  yesterday's price for Tue-Fri, last week's for Sat-Mon
lasso  one linear model per hour of day (the usual benchmark in the price-forecast literature)
trees  one gradient-boosted tree model (LightGBM) for all hours; its settings were chosen with tune.py
nn     two small neural networks for all hours, averaged
lgbm   the price forecast: trees and nn, half each (see blended; the name dates from when it was the
       trees alone). lgbm_raw is what the models gave, lgbm has its recent lean taken out (see recent_error)
lgbm_p10, lgbm_p50, lgbm_p90   the tree model asked for the price that 10%, 50% and 90%
       of outcomes stay below; p10 to p90 is the band a dispatch plan can rely on 8 times in 10
p_negative   the chance that the hour's price is below zero
All models also get our own wind and solar forecast, made from the weather columns.
Walk-forward: for every test day each model is refitted on all earlier rows, then predicts that
day, as forecast.py does. Prints the scores and writes every prediction to data/backtest.db.
    .venv/bin/python model.py             # about 80 min, uses 24 processes
    .venv/bin/python model.py --selftest
"""
import os
import sqlite3
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from functools import partial
from multiprocessing import get_context

import numpy as np
import pandas as pd

from build_dataset import DATA, TZ, hourly, sql

warnings.filterwarnings("ignore", message="X does not have valid feature names")
TEST_DAYS = 365
# 1 = a refit for every day, as forecast.py does. With 7 the score moved by up to 0.3 EUR/MWh with the
# weekday the refits fell on, which hid every smaller gain (tune.py, round 4)
REFIT_DAYS = 1
PARTS = 12  # backtest_all cuts each model's refits into this many tasks
SPIKE_SHARE = 0.05  # the dearest 5% of test hours count as spikes
HALF_LIFE_DAYS = 730  # a row two years old counts half as much as today's (tune.py: MAE 16.26 -> 15.81)
LGBM = dict(n_estimators=600, learning_rate=0.05, num_leaves=63, colsample_bytree=0.5, verbose=-1)
BAND = ["lgbm_p10", "lgbm_p50", "lgbm_p90"]
# Smaller trees for the band edges and the negative-price chance. With the LGBM settings the band held
# 55% of prices instead of 80%, and hours given a 70-90% chance of a negative price had one 46% of the time.
CAUTIOUS = dict(num_leaves=15, min_child_samples=100)
NN = dict(hidden_layer_sizes=(128, 64), early_stopping=True, random_state=0)
NETS = ["nn_0", "nn_1"]  # the same network from two random starts: one alone scores 17.7, their mean 16.3


def naive(df):
    return df["price_lag_24h"].where(df["weekday"].between(1, 4), df["price_lag_168h"])


class Asinh:
    """Squash prices so that spikes do not dominate the fit: asinh of (price - median) / spread."""

    def __init__(self, prices):
        self.mid = np.nanmedian(prices)
        self.spread = np.nanmedian(np.abs(prices - self.mid)) * 1.4826

    def __call__(self, prices):
        return np.arcsinh((prices - self.mid) / self.spread)

    def back(self, values):
        return np.sinh(values) * self.spread + self.mid


def _numeric_inputs(train, test, features, squash):
    """For the models that are not trees: the columns that are mostly filled, with the price columns squashed."""
    # a column that is mostly empty (the Rhine gauge has 31 days) gets a tiny spread after filling
    # the gaps, so its few real values turn into huge inputs: predictions of -800 EUR/MWh
    features = [f for f in features if train[f].notna().mean() > 0.5]
    prices = [f for f in features if f.startswith(("price_", "intraday_"))]
    train_x, test_x = train[features].copy(), test[features].copy()
    train_x[prices], test_x[prices] = squash(train_x[prices]), squash(test_x[prices])
    return train_x, test_x


def fit_predict_lasso(train, test, features, squash):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LassoLarsIC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_x, test_x = _numeric_inputs(train, test, features, squash)
    out = pd.Series(np.nan, test.index)
    # shortcut: each hour's model sees only its own hour's row, not the whole day's 24 values
    # as in the published LEAR benchmark; widen the table if lasso is kept in the final blend
    for hour in sorted(test["hour"].unique()):
        rows, target = train["hour"] == hour, test["hour"] == hour
        model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                              LassoLarsIC(criterion="aic"))
        model.fit(train_x[rows], squash(train.loc[rows, "price"]))
        out[target] = squash.back(model.predict(test_x[target]))
    return out


def fit_predict_nn(train, test, features, squash, half_life_days=HALF_LIFE_DAYS, **params):
    """A small neural network for all hours. Unlike the trees it can go beyond the prices it was fitted on."""
    from sklearn.neural_network import MLPRegressor

    train, weights = recent(train, half_life_days=half_life_days)
    train_x, test_x = _numeric_inputs(train, test, features, squash)
    mid, spread = train_x.median(), train_x.std().replace(0, 1)

    def ready(x):  # every column on one scale, gaps at the middle; hour and weekday as one yes/no column each
        return np.hstack([((x - mid) / spread).fillna(0), np.eye(24)[x["hour"].astype(int)],
                          np.eye(7)[x["weekday"].astype(int)]])

    model = MLPRegressor(**{**NN, **params}).fit(ready(train_x), squash(train["price"]), sample_weight=weights)
    return pd.Series(squash.back(model.predict(ready(test_x))), test.index)


class _NoSquash:
    def __call__(self, prices):
        return prices

    def back(self, values):
        return values


def recent(train, window_days=None, half_life_days=None):
    """Keep only the last window_days of rows; weigh each row 0.5 ** (its age / half_life_days)."""
    age = (train.index[-1] - train.index).days
    if window_days:
        train, age = train[age < window_days], age[age < window_days]
    return train, (np.asarray(0.5 ** (age / half_life_days)) if half_life_days else None)


def fit_predict_lgbm(train, test, features, squash, window_days=None, half_life_days=None, anchor=None,
                     ratio=False, raw=False, **params):
    """anchor: a column holding a price level known before the auction. The model then forecasts the gap to
    it (or, with ratio, the price as a multiple of it), so it can reach a price level it never saw in training.
    raw: fit the price as it is, without the squash."""
    from lightgbm import LGBMRegressor

    train, weights = recent(train, window_days, half_life_days)
    target = train["price"]
    if anchor:
        target = target / train[anchor] if ratio else target - train[anchor]
        squash = Asinh(target.to_numpy())
    if raw:
        squash = _NoSquash()
    known = target.notna().to_numpy()
    model = LGBMRegressor(**{**LGBM, **params})
    model.fit(train.loc[known, features], squash(target[known]), sample_weight=None if weights is None else weights[known])
    out = pd.Series(squash.back(model.predict(test[features])), test.index)
    return out if not anchor else out * test[anchor] if ratio else out + test[anchor]


def fit_predict_negative(train, test, features, squash, **params):
    """The chance, 0 to 1, that the price is below zero."""
    from lightgbm import LGBMClassifier

    train, weights = recent(train, half_life_days=HALF_LIFE_DAYS)
    model = LGBMClassifier(**{**LGBM, **params}).fit(train[features], train["price"] < 0, sample_weight=weights)
    return pd.Series(model.predict_proba(test[features])[:, 1], test.index)


def add_generation_forecast(df):
    """Add our own wind and solar forecast (MW), made from the weather columns.

    SMARD's own forecasts come out after the auction, so they cannot be used. Each quarter of a
    year is predicted by a model fitted on earlier quarters only, so no row sees its own outcome.
    The first year has nothing earlier and stays empty.
    """
    from lightgbm import LGBMRegressor

    actual = hourly("smard.db", "SELECT ts, series AS name, value FROM smard"
                                " WHERE series IN ('solar', 'wind_onshore', 'wind_offshore')", scale=4).reindex(df.index)
    x = df[[c for c in df.columns if c.startswith("wx_")]].assign(utc_hour=df.index.hour, day=df.index.day_of_year)
    quarter = df.index.year * 4 + df.index.quarter
    total = 0
    for name, capacity in {"solar": "capacity_solar_ac", "wind_onshore": "capacity_wind_onshore",
                           "wind_offshore": "capacity_wind_offshore"}.items():
        installed = df[capacity] * 1000  # GW -> MW
        share = actual[name] / installed  # share of installed capacity running: stays comparable as capacity grows
        pred = pd.Series(np.nan, df.index)
        for q in np.unique(quarter)[4:]:
            rows = (quarter < q) & share.notna()
            model = LGBMRegressor(n_estimators=300, learning_rate=0.05, verbose=-1).fit(x[rows], share[rows])
            pred[quarter == q] = model.predict(x[quarter == q])
        df[f"{name}_own_forecast"] = pred * installed
        total = total + df[f"{name}_own_forecast"]
        year = df.index >= df.index[-1] - pd.Timedelta(days=TEST_DAYS)
        print(f"own {name} forecast, last 12 months: mean error "
              f"{(df[f'{name}_own_forecast'] - actual[name])[year].abs().mean() / 1000:.2f} GW", flush=True)
    df["residual_load_own_forecast"] = df["load_forecast"] - total
    return df


def backtest(df, models, part=slice(None)):
    """models: {name: (fit_predict function, feature list)} -> frame of predictions for the test days.
    part: which of the refits to do; the days of the others stay empty."""
    days = df.index.tz_convert(TZ).normalize()
    last = days[df["price"].notna()].max()
    start = last - pd.DateOffset(days=TEST_DAYS - 1)  # calendar days: 24-hour steps miss midnight across a clock change
    out = pd.DataFrame({"price": df["price"], "naive": naive(df)})[(days >= start) & (days <= last)]
    for block in pd.date_range(start, last, freq=f"{REFIT_DAYS}D")[part]:
        train = df[(days < block) & df["price"].notna()]
        test = df[(days >= block) & (days < block + pd.DateOffset(days=REFIT_DAYS)) & (days <= last)]
        squash = Asinh(train["price"].to_numpy())
        for name, (fit_predict, features) in models.items():
            out.loc[test.index, name] = fit_predict(train, test, features, squash)
    return out


def _start_worker(df):
    global _table
    _table = df  # handed to each worker once: sending it along with every task would copy it 80 times


def _backtest_task(task):
    model, part = task
    return backtest(_table, dict([model]), part)


def backtest_all(df, models, workers=24):
    """backtest() with the refits shared out over processes. Give the models one thread each: a LightGBM
    fit on 3 threads takes 9 s against 15 s on one, so one thread per process does more work per core."""
    os.environ["OPENBLAS_NUM_THREADS"] = "1"  # read by numpy in the workers: 24 x 32 threads would stall the lasso
    tasks = [(model, slice(i, None, PARTS)) for model in models.items() for i in range(PARTS)]
    # spawn, not fork: LightGBM's threads do not survive a fork
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=_start_worker,
                             initargs=(df,)) as pool:
        results = list(pool.map(_backtest_task, tasks))
    out = results[0][["price", "naive"]].assign(**{name: np.nan for name in models})
    for ((name, _), _), result in zip(tasks, results):
        out[name] = out[name].fillna(result[name])
    return out


def conformal(pred, low, high, window="56D", min_hours=168):
    """Move each edge of a P10-P90 band by what its misses over the last 8 weeks call for.

    If more than 10% of recent prices fell below the low edge, the edge moves down until 10% would
    have; the same for the high edge. A delivery day only uses misses up to the day before, whose
    prices are known when the forecast is made. Returns the two moved edges.
    """
    day = pred.index.tz_convert(TZ).normalize()
    edges = []
    for edge, miss, sign in ((pred[low], pred[low] - pred["price"], -1), (pred[high], pred["price"] - pred[high], 1)):
        move = miss.rolling(window, min_periods=min_hours).quantile(0.9).groupby(day).last().shift(1)
        edges.append(edge + sign * move.reindex(day).fillna(0).to_numpy())
    return edges


def recent_error(pred, column, window="28D", min_hours=168):
    """What a forecast column was off by lately: the median of forecast - price over the last 4 weeks.

    Subtracting it removes a steady lean. The price model leaned low (-3.1 EUR/MWh over the test year,
    -6.5 in the high-price autumn of 2026) because it is fitted on squashed prices and trees cannot go
    above what they saw. A delivery day only uses errors up to the day before. Tried on the 12-month
    test: mean error -3.1 -> -0.4, dearest 5% of hours 46.1 -> 43.7, all hours unchanged (15.81 -> 15.76),
    hours with a negative price slightly worse (23.1 -> 23.8). Windows of 14 and 56 days gave the same.
    """
    day = pred.index.tz_convert(TZ).normalize()
    error = (pred[column] - pred["price"]).rolling(window, min_periods=min_hours).median()
    return error.groupby(day).last().shift(1).reindex(day).fillna(0).to_numpy()


def band_scores(price, low, high):
    """Shares of prices inside, below and above a band, and its mean width. A P10-P90 band should give 0.8, 0.1, 0.1."""
    return {"inside": ((price >= low) & (price <= high)).mean(), "below": (price < low).mean(),
            "above": (price > high).mean(), "mean_width": (high - low).mean()}


def negative_report(pred):
    from sklearn.metrics import roc_auc_score

    actual, chance = pred["price"] < 0, pred["p_negative"]
    print(f"\nnegative prices: {actual.mean():.1%} of test hours."
          f" Ranking score (AUC; 0.5 = guessing, 1 = perfect): {roc_auc_score(actual, chance):.3f}")
    print(f"mean squared error of the chance: {((chance - actual) ** 2).mean():.4f},"
          f" of always saying {actual.mean():.1%}: {actual.var(ddof=0):.4f}")
    rules = {"chance >= 0.5": chance >= 0.5, "chance >= 0.3": chance >= 0.3, "lgbm forecast < 0": pred["lgbm"] < 0,
             "lgbm_p50 forecast < 0": pred["lgbm_p50"] < 0}
    for name, flagged in rules.items():
        print(f"  {name}: flags {flagged.sum()} hours, {actual[flagged].mean():.0%} of them negative,"
              f" catches {flagged[actual].mean():.0%} of the negative hours")
    bucket = pd.cut(chance, [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1], include_lowest=True)
    print("is the chance honest? (share_negative should match mean_chance)")
    print(pd.DataFrame({"hours": chance.groupby(bucket).size(), "mean_chance": chance.groupby(bucket).mean(),
                        "share_negative": actual.groupby(bucket).mean()}).round(2).to_string())


def scores(pred):
    """One row per model: mean absolute error overall, relative to naive, and on the hard hours."""
    error = pred.drop(columns="price").sub(pred["price"], axis=0).abs()
    spike = pred["price"] >= pred["price"].quantile(1 - SPIKE_SHARE)
    table = pd.DataFrame({"MAE": error.mean(), "rMAE": error.mean() / error["naive"].mean(),
                          "MAE_negative": error[pred["price"] < 0].mean(), "MAE_spike": error[spike].mean()})
    return table.round(2)


def load():
    df = sql("training.db", "SELECT * FROM training")
    df.index = pd.to_datetime(df.pop("ts"), utc=True)
    return df.astype(float)  # columns with no values at all come back from SQLite as text


def feature_columns(df):
    # tune.py: the Yahoo commodity columns (cmd_) change nothing, so they stay out
    return [c for c in df.columns if c != "price" and not c.startswith("cmd_")]


def daily_models(features, n_jobs=3):
    """The models forecast.py runs every day, as {name: (fit_predict function, feature list)}."""
    lgbm = partial(fit_predict_lgbm, half_life_days=HALF_LIFE_DAYS, n_jobs=n_jobs)
    return {"trees": (lgbm, features),
            **{name: (partial(fit_predict_nn, random_state=i), features) for i, name in enumerate(NETS)},
            **{name: (partial(lgbm, objective="quantile", alpha=int(name[-2:]) / 100,
                              **({} if name == "lgbm_p50" else CAUTIOUS)), features) for name in BAND},
            "p_negative": (partial(fit_predict_negative, n_jobs=n_jobs, n_estimators=200, **CAUTIOUS), features)}


def blended(pred):
    """Add the price forecast, lgbm: the trees and the mean of the networks, half each.

    The networks alone are worse than the trees (16.27 against 15.48 EUR/MWh on the 12-month test) but
    err on other hours, so the mix scores 14.94; in the dearest 5% of hours 41.7 against 43.8. Two seeds
    of trees averaged gave only 15.34 (tune.py, round 5).
    """
    pred = pred.assign(nn=pred[NETS].mean(axis=1)).drop(columns=NETS)
    return pred.assign(lgbm=(pred["trees"] + pred["nn"]) / 2)


def main():
    df = add_generation_forecast(load())
    features = feature_columns(df)
    pred = backtest_all(df, {"lasso": (fit_predict_lasso, features), **daily_models(features, n_jobs=1)})
    assert pred.notna().all().all(), pred.isna().sum()
    pred = blended(pred)
    crossed = (pred[BAND].diff(axis=1).iloc[:, 1:] < 0).any(axis=1).mean()
    pred[BAND] = np.sort(pred[BAND], axis=1)  # the three are fitted apart and can come out in the wrong order
    pred["lgbm_p10_adj"], pred["lgbm_p90_adj"] = conformal(pred, "lgbm_p10", "lgbm_p90")
    pred["lgbm_raw"] = pred["lgbm"]  # as the models gave it; lgbm is the forecast to use
    pred["lgbm"] = pred["lgbm_raw"] - recent_error(pred, "lgbm_raw")
    point = pred[["price", "naive", "lasso", "trees", "nn", "lgbm_raw", "lgbm", "lgbm_p50"]]
    print("\nmean of forecast - price:", point.drop(columns="price").sub(pred["price"], axis=0).mean().round(2).to_dict())

    print(f"\ntest {pred.index[0]} to {pred.index[-1]}, {len(pred)} hours, "
          f"{(pred['price'] < 0).sum()} negative, mean price {pred['price'].mean():.2f} EUR/MWh\n")
    print(scores(point).to_string())
    error = point.drop(columns="price").sub(pred["price"], axis=0).abs()
    print("\nMAE by hour of day (German time)")
    print(error.groupby(pred.index.tz_convert(TZ).hour).mean().round(1).to_string())
    print(f"\nP10-P90 band (should be 0.8 inside, 0.1 below, 0.1 above); {crossed:.1%} of hours came out in the wrong order")
    print(pd.DataFrame({"as fitted": band_scores(pred["price"], pred["lgbm_p10"], pred["lgbm_p90"]),
                        "adjusted by recent misses": band_scores(pred["price"], pred["lgbm_p10_adj"],
                                                                 pred["lgbm_p90_adj"])}).T.round(3).to_string())
    negative_report(pred)
    with closing(sqlite3.connect(DATA / "backtest.db")) as con:
        pred.set_axis(pred.index.strftime("%Y-%m-%dT%H:%M:%SZ")).to_sql("backtest", con, if_exists="replace",
                                                                         index_label="ts")


def _selftest():
    df = pd.DataFrame({"weekday": [0, 1, 4, 5], "price_lag_24h": 1.0, "price_lag_168h": 7.0})
    assert naive(df).tolist() == [7.0, 1.0, 1.0, 7.0]  # Monday and Saturday look back a week
    prices = np.array([-500.0, -5.0, 0.0, 40.0, 90.0, 3000.0])
    squash = Asinh(prices)
    assert np.allclose(squash.back(squash(prices)), prices)
    days = pd.DataFrame({"x": [1, 2, 3]}, index=pd.date_range("2026-01-01", periods=3, tz="UTC"))
    kept, weights = recent(days, window_days=2, half_life_days=1)
    assert kept["x"].tolist() == [2, 3] and weights.tolist() == [0.5, 1.0]  # oldest day dropped, newest counts most
    pred = pd.DataFrame({"price": [-10.0, 0.0, 10.0, 100.0], "naive": [0.0, 0.0, 0.0, 0.0],
                         "m": [-10.0, 0.0, 10.0, 80.0]})
    table = scores(pred)
    assert table.loc["m", "MAE"] == 5 and table.loc["m", "rMAE"] == round(5 / 30, 2), table
    assert table.loc["naive", "MAE_negative"] == 10 and table.loc["m", "MAE_spike"] == 20, table
    band = pd.DataFrame({"price": 50.0, "low": 45.0, "high": 52.0},
                        index=pd.date_range("2026-01-01", periods=240, freq="h", tz=TZ))
    low, high = conformal(band, "low", "high")
    assert low.iloc[0] == 45 and high.iloc[0] == 52      # no history yet: band unchanged
    assert low.iloc[-1] == 50 and high.iloc[-1] == 50    # a band that was always too wide closes in on the price
    assert band_scores(band["price"], band["low"], band["high"]) == {"inside": 1, "below": 0, "above": 0, "mean_width": 7}
    table = pd.DataFrame({"price": 50.0, "weekday": 1, "price_lag_24h": 40.0, "price_lag_168h": 40.0},
                         index=pd.date_range("2025-01-01", periods=24 * 400, freq="h", tz=TZ))
    constant = {"m": (lambda train, test, features, squash: pd.Series(float(len(train)), test.index), [])}
    whole, halves = backtest(table, constant), [backtest(table, constant, slice(i, None, 2)) for i in (0, 1)]
    assert len(whole) == 24 * TEST_DAYS and whole["m"].notna().all() and whole["m"].nunique() == TEST_DAYS / REFIT_DAYS
    assert halves[0]["m"].fillna(halves[1]["m"]).equals(whole["m"]) and halves[0]["m"].isna().mean() > 0.4
    mixed = blended(pd.DataFrame({"trees": [10.0], "nn_0": [20.0], "nn_1": [40.0], "p_negative": [0.5]}))
    assert mixed.to_dict("records") == [{"trees": 10, "p_negative": 0.5, "nn": 30, "lgbm": 20}], mixed
    lean = recent_error(band, "low")
    assert lean[0] == 0 and lean[-1] == -5  # no history yet: no correction; then: always 5 too low
    print("selftest ok")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
