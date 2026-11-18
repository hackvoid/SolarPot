"""Score the quarter-hour model: how far each quarter-hour's price sits from its hour's average.

model.py forecasts the hour. This model adds the shape inside the hour, from the ramps of demand,
sun and wind. Quarter-hour prices only exist since 2025-10-01, so the test is the last 180 days;
for every test day the model is refitted on all earlier rows.
Prints the scores and writes the predictions to table backtest_15min in data/backtest.db.
Run model.py first: the end-to-end score reads its hourly forecast from data/backtest.db.
    .venv/bin/python model_15min.py             # about 3 min
    .venv/bin/python model_15min.py --selftest
"""
import sqlite3
import sys
from contextlib import closing

import numpy as np
import pandas as pd

from build_dataset import DATA, TZ, sql
from model import REFIT_DAYS, add_generation_forecast, load

TEST_DAYS = 180
# absolute-error loss: a few huge gaps would otherwise steer the fit (mean error 5.22 -> 4.98 when tried);
# more or bigger trees, and a ramp taken from the hourly price forecast, changed nothing
SETTINGS = dict(objective="l1", n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=100, verbose=-1)
HOURLY = ["hour", "weekday", "month", "holiday", "load_forecast", "solar_own_forecast",
          "residual_load_own_forecast", "price_lag_24h", "price_day_mean"]  # copied from the hour's row
OWN = ["solar", "wind_onshore", "wind_offshore"]


def slope_within_day(x):
    """Change per hour of an hourly series, taken from its neighbours inside the same delivery day.

    The next day's first hour is not known when the bid is made, so the day's edges use one side only.
    """
    day = x.groupby(x.index.tz_convert(TZ).normalize())
    return ((day.shift(-1) - day.shift(1)) / 2).fillna(day.shift(-1) - x).fillna(x - day.shift(1))


def load_quarters(hours=None):
    """hours: the hourly table with our own wind and solar forecast, if the caller already has it."""
    hours = add_generation_forecast(load()) if hours is None else hours
    q = sql("training.db", "SELECT * FROM training_15min")
    q.index = pd.to_datetime(q.pop("ts"), utc=True)
    q["hour_ts"] = pd.to_datetime(q["hour_ts"], utc=True)
    q[HOURLY] = hours[HOURLY].reindex(q["hour_ts"]).to_numpy()
    position = (q["quarter"] - 1.5) / 4  # where the quarter-hour sits inside its hour: -0.375 to 0.375
    for name in OWN:  # our own hourly forecast, turned into an estimated ramp inside the hour (MW)
        q[f"{name}_ramp"] = slope_within_day(hours[f"{name}_own_forecast"]).reindex(q["hour_ts"]).to_numpy() * position
    q["residual_ramp"] = q["load_forecast_deviation"] - q[[f"{name}_ramp" for name in OWN]].sum(axis=1)
    return q


def feature_columns(q):
    return [c for c in q.columns if c not in ("hour_ts", "price", "deviation")]


def fit_predict(train, test, features, **params):
    from lightgbm import LGBMRegressor

    model = LGBMRegressor(**{**SETTINGS, **params}).fit(train[features], train["deviation"])
    return model.predict(test[features])


def backtest(q, features, **params):
    days = q.index.tz_convert(TZ).normalize()
    known = q["deviation"].notna()
    last = days[known].max()
    start = last - pd.DateOffset(days=TEST_DAYS - 1)  # calendar days: 24-hour steps miss midnight across a clock change
    pred = pd.Series(np.nan, q.index)
    for block in pd.date_range(start, last, freq=f"{REFIT_DAYS}D"):
        train = q[(days < block) & known]
        test = (days >= block) & (days < block + pd.DateOffset(days=REFIT_DAYS)) & (days <= last)
        pred[test] = fit_predict(train, q[test], features, **params)
    return pred[(days >= start) & (days <= last)]


def main():
    q = load_quarters()
    features = feature_columns(q)
    pred = backtest(q, features)
    out = pd.DataFrame({"price": q["price"], "deviation": q["deviation"], "flat": 0.0,
                        "yesterday": q["deviation_lag_24h"], "model": pred}).loc[pred.index]
    guesses = out[["flat", "yesterday", "model"]]
    print(f"\ntest {out.index[0]} to {out.index[-1]}, {len(out)} quarter-hours\n")
    print("mean error on the gap between a quarter-hour and its hour (EUR/MWh)")
    print(guesses.sub(out["deviation"], axis=0).abs().mean().round(2).to_string())

    with closing(sqlite3.connect(DATA / "backtest.db")) as con:
        hourly = pd.read_sql("SELECT ts, lgbm FROM backtest", con, index_col="ts")["lgbm"]
        hourly.index = pd.to_datetime(hourly.index, utc=True)
        out["hour_forecast"] = hourly.reindex(q.loc[out.index, "hour_ts"]).to_numpy()
        print("\nmean error of the full quarter-hour price forecast: hourly forecast + gap (EUR/MWh)")
        print(guesses.add(out["hour_forecast"], axis=0).sub(out["price"], axis=0).abs().mean().round(2).to_string())
        print("\nwhat the model leans on (share of its splits)")
        from lightgbm import LGBMRegressor
        known = q["deviation"].notna()
        final = LGBMRegressor(**SETTINGS).fit(q.loc[known, features], q.loc[known, "deviation"])
        weight = pd.Series(final.feature_importances_, features).sort_values(ascending=False)
        print((weight / weight.sum()).round(3).head(10).to_string())
        out.set_axis(out.index.strftime("%Y-%m-%dT%H:%M:%SZ")).to_sql("backtest_15min", con, if_exists="replace",
                                                                       index_label="ts")


def _selftest():
    hours = pd.date_range("2026-01-01", periods=48, freq="h", tz=TZ)
    x = pd.Series(np.arange(48.0), hours)
    x.iloc[24:] += 100  # a jump between the two days must not leak into either day's slopes
    assert (slope_within_day(x) == 1).all(), slope_within_day(x)
    print("selftest ok")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
