"""Daily job: forecast tomorrow's day-ahead prices before the 12:00 auction.

Rebuilds data/training.db, fits every model on all days with a published price and writes the
forecast to data/forecast.db:
  forecast        one row per hour. lgbm = the price forecast (EUR/MWh; trees and neural networks mixed,
                  see model.blended; lgbm_raw before its recent lean is taken out); lgbm_p10_adj to lgbm_p90_adj =
                  the band that should hold 8 prices in 10; p_negative = chance of a price below zero
  forecast_15min  one row per quarter-hour. price = the hour's forecast + gap
It first prints how the stored forecasts did against the prices published since.
Needs SMARD's demand forecast for the day (without it the day is not in the table and the job stops)
and data/backtest.db from model.py (the band is set from its recent misses until live ones pile up).
That demand forecast arrived between 10:51 and 11:06 German time on Sunday 2026-11-15, so the job
runs at 11:10 and again at 11:40; the second run replaces the first.
    .venv/bin/python forecast.py               # tomorrow, about 5 min, uses about 10 cores
    .venv/bin/python forecast.py 2026-11-15    # another day; stored only while its price is unpublished
    .venv/bin/python forecast.py --selftest
"""
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import build_dataset
from build_dataset import DATA, TZ, sql
from model import (BAND, Asinh, add_generation_forecast, band_scores, blended, conformal, daily_models,
                   feature_columns, load, naive, recent_error, scores)
from model_15min import feature_columns as quarter_columns
from model_15min import fit_predict as fit_predict_gap
from model_15min import load_quarters

# its own file: model.py replaces the tables of backtest.db, and a live forecast cannot be made again
SCHEMA = """
CREATE TABLE IF NOT EXISTS forecast(ts TEXT PRIMARY KEY, issued_at TEXT, lgbm REAL, lgbm_p10 REAL, lgbm_p50 REAL,
    lgbm_p90 REAL, p_negative REAL, lgbm_p10_adj REAL, lgbm_p90_adj REAL, lgbm_raw REAL);
CREATE TABLE IF NOT EXISTS forecast_15min(ts TEXT PRIMARY KEY, issued_at TEXT, hour_ts TEXT, gap REAL, price REAL);
"""
STAMP = "%Y-%m-%dT%H:%M:%SZ"


def hours_in(day):
    """24, or 23 and 25 on the two days the clock changes."""
    start = pd.Timestamp(day, tz=TZ)
    return int((start + pd.DateOffset(days=1) - start) / pd.Timedelta(hours=1))


def stored(con, table):
    frame = pd.read_sql(f"SELECT * FROM {table} ORDER BY ts", con, index_col="ts")
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame


def store(con, table, frame):
    """Write a frame indexed by UTC time; a second run for the same day replaces the first."""
    frame = frame.set_axis(frame.index.strftime(STAMP)).reset_index(names="ts")
    con.executemany(f"INSERT OR REPLACE INTO {table}({','.join(frame.columns)})"
                    f" VALUES ({','.join('?' * frame.shape[1])})", frame.itertuples(index=False))


def with_recent_misses(pred, earlier, price):
    """Apply what the recent misses call for: the band edges are moved (model.conformal) and the price
    forecast loses its lean (model.recent_error). The model's own output is kept as lgbm_raw.

    earlier: frames of past lgbm_raw, lgbm_p10 and lgbm_p90; where two cover the same hour the later frame wins.
    """
    pred = pred.rename(columns={"lgbm": "lgbm_raw"})
    hist = pd.concat([*earlier, pred])[["lgbm_raw", "lgbm_p10", "lgbm_p90"]]
    hist = hist[~hist.index.duplicated(keep="last")].sort_index().astype(float)  # an empty table reads as text
    hist["price"] = price.reindex(hist.index)
    low, high = conformal(hist, "lgbm_p10", "lgbm_p90")
    lean = pd.Series(recent_error(hist, "lgbm_raw"), hist.index)
    return pred.assign(lgbm=pred["lgbm_raw"] - lean[pred.index], lgbm_p10_adj=low[pred.index],
                       lgbm_p90_adj=high[pred.index])


def live_score(con, df):
    f = stored(con, "forecast")
    f["price"], f["naive"] = df["price"].reindex(f.index), naive(df).reindex(f.index)
    f = f.dropna(subset="price")
    if f.empty:
        return print("live score: no stored forecast has a published price yet")
    print(f"\nlive score, {len(f)} hours from {f.index[0]:%Y-%m-%d} to {f.index[-1]:%Y-%m-%d} UTC (EUR/MWh)")
    print(scores(f[["price", "naive", "lgbm"]]).to_string())
    print("band (should be 0.8 inside):",
          {k: round(float(v), 3) for k, v in band_scores(f["price"], f["lgbm_p10_adj"], f["lgbm_p90_adj"]).items()})
    quarters = pd.read_sql("SELECT ts, price FROM forecast_15min", con, index_col="ts")["price"]
    actual = sql("training.db", "SELECT ts, price FROM training_15min").set_index("ts")["price"]
    print(f"quarter-hour price forecast: mean error {(quarters - actual).abs().mean():.2f}")


def main(day=None):
    build_dataset.main()
    df = load()
    day = day or f"{pd.Timestamp.now(TZ) + pd.DateOffset(days=1):%Y-%m-%d}"
    local_day = df.index.tz_convert(TZ).strftime("%Y-%m-%d")
    with closing(sqlite3.connect(DATA / "forecast.db", timeout=60)) as con:
        con.executescript(SCHEMA)
        live_score(con, df)
        if (local_day == day).sum() != hours_in(day):
            sys.exit(f"{day}: {(local_day == day).sum()} of {hours_in(day)} hours in the table."
                     " SMARD's demand forecast for that day is not out yet; nothing forecast")
        df = add_generation_forecast(df)
        train, test = df[(local_day < day) & df["price"].notna()], df[local_day == day]
        features = feature_columns(df)
        usual = train[features].tail(24 * 365).notna().mean() > 0.5
        empty = [c for c in features if usual[c] and test[c].isna().all()]
        if empty:  # a feed that stopped: the models still run, on less
            print(f"WARNING no values for {day} in: {', '.join(empty)}")

        squash = Asinh(train["price"].to_numpy())
        pred = blended(pd.DataFrame({name: fit_predict(train, test, columns, squash)
                                     for name, (fit_predict, columns) in daily_models(features).items()}))
        pred = pred.drop(columns=["trees", "nn"])  # the two halves of lgbm; model.py scores them apart
        pred[BAND] = np.sort(pred[BAND], axis=1)  # fitted apart, they can come out in the wrong order
        backtest = sql("backtest.db", "SELECT ts, lgbm_raw, lgbm_p10, lgbm_p90 FROM backtest").set_index("ts")
        backtest.index = pd.to_datetime(backtest.index, utc=True)
        pred = with_recent_misses(pred, [backtest, stored(con, "forecast")], df["price"])

        q = load_quarters(df)
        mine = q["hour_ts"].isin(test.index)
        assert mine.sum() == 4 * len(test), f"{mine.sum()} quarter-hours for {len(test)} hours"
        gap = fit_predict_gap(q[(q.index < test.index[0]) & q["deviation"].notna()], q[mine], quarter_columns(q))
        quarters = pd.DataFrame({"hour_ts": q.loc[mine, "hour_ts"].dt.strftime(STAMP), "gap": gap,
                                 "price": pred["lgbm"].reindex(q.loc[mine, "hour_ts"]).to_numpy() + gap})

        print(f"\nforecast for {day}, German time (EUR/MWh; p_negative = chance of a price below zero)")
        print(pred.set_axis(test.index.tz_convert(TZ).strftime("%H:%M"))[
            ["lgbm", "lgbm_p10_adj", "lgbm_p90_adj", "p_negative"]].round(2).to_string())
        if test["price"].notna().any():
            print(f"\nprice already published, so this is a hindcast and is not stored. Mean error: hours"
                  f" {(pred['lgbm'] - test['price']).abs().mean():.2f} (naive {(naive(test) - test['price']).abs().mean():.2f}),"
                  f" quarter-hours {(quarters['price'] - q.loc[mine, 'price']).abs().mean():.2f}")
        else:
            issued = datetime.now(timezone.utc).strftime(STAMP)
            store(con, "forecast", pred.assign(issued_at=issued))
            store(con, "forecast_15min", quarters.assign(issued_at=issued))
            con.commit()
            print(f"\nstored {len(pred)} hours and {len(quarters)} quarter-hours in data/forecast.db")


def _selftest():
    assert [hours_in(d) for d in ("2026-03-29", "2026-10-25", "2026-11-16")] == [23, 25, 24]
    hours = pd.date_range("2026-01-01", periods=264, freq="h", tz="UTC")
    past, day = hours[:240], hours[240:]
    old = pd.DataFrame({"lgbm_raw": 0.0, "lgbm_p10": 0.0, "lgbm_p90": 99.0}, past)    # e.g. the backtest
    live = pd.DataFrame({"lgbm_raw": 60.0, "lgbm_p10": 45.0, "lgbm_p90": 52.0}, past)  # stored forecasts must win over it
    pred = pd.DataFrame({"lgbm": 60.0, "lgbm_p10": 45.0, "lgbm_p90": 52.0}, day)
    with closing(sqlite3.connect(":memory:")) as con:
        con.executescript(SCHEMA)
        live_score(con, pd.DataFrame({"price": 50.0, "weekday": 1, "price_lag_24h": 50.0, "price_lag_168h": 50.0}, past))
        # the day's own price is not known; the still empty forecast table must not disturb the rest
        got = with_recent_misses(pred, [old, live, stored(con, "forecast")], pd.Series(50.0, past))
        assert got.dtypes.eq(float).all(), got.dtypes
        assert (got["lgbm_p10_adj"] == 50).all() and (got["lgbm_p90_adj"] == 50).all(), got  # too wide: closes in
        assert (got["lgbm_raw"] == 60).all() and (got["lgbm"] == 50).all(), got  # always 10 too high: taken off
        row = got.assign(lgbm_p50=1.0, p_negative=0.1, issued_at="x")
        store(con, "forecast", row)
        store(con, "forecast", row.assign(lgbm=2.0))  # second run of the same morning
        back = stored(con, "forecast")
        assert len(back) == 24 and (back["lgbm"] == 2).all() and back.index.equals(day), back
    print("selftest ok")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    _selftest() if "--selftest" in sys.argv else main(args[0] if args else None)
