"""Join every collector database into the training tables in data/training.db.

`training`: one row per delivery hour (UTC) from 2018-10-01, the start of the DE-LU price zone, to
the last hour that has a demand forecast. `price` is the target: the day-ahead price in EUR/MWh.
`training_15min`: one row per quarter-hour from 2025-10-01, for a second model that predicts how
each quarter-hour's price differs from its hour (see build_quarter_hours).

Every other column holds only what was known when the forecast is made: the morning of the day
before delivery, ahead of the 12:00 auction. That rule sets the lags below.
    .venv/bin/python build_dataset.py             # rebuilds the whole table, about a minute
    .venv/bin/python build_dataset.py --selftest
"""
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import holidays
import pandas as pd

from commodities import SYMBOLS
from outages import out_of_service

DATA = Path(__file__).parent / "data"
START = pd.Timestamp("2018-09-30T22:00:00Z")  # 2018-10-01 00:00 German time
TZ = "Europe/Berlin"
STATES = ["BW", "BY", "BE", "BB", "HB", "HH", "HE", "MV", "NI", "NW", "RP", "SL", "SN", "ST", "SH", "TH"]
WORLDBANK = ["Coal, Australian", "Coal, South African", "Natural gas, Europe", "Nickel", "Lead", "Tin"]

# Days between a daily value's date and the first delivery day allowed to use it.
LAG_PRICE_STATS = 1   # yesterday's prices were fixed the day before
LAG_MARKET = 2        # gas, CO2, commodity closes, Rhine: last full day before the forecast morning
LAG_GAS_STORAGE = 3   # published the evening after the gas day
LAG_HYDRO = 7         # weekly figures, published some days after the week ends
LAG_HYDRO_ENTSOE = 14  # dated by the Monday the week starts; the week of 2026-11-09 was not out on 2026-11-18
LAG_WORLDBANK = 40    # a month's average is published early the following month


def sql(db, query):
    with closing(sqlite3.connect(DATA / db, timeout=60)) as con:
        return pd.read_sql(query, con)


def hourly(db, query, scale=1, freq="h"):
    """Run a query returning (ts, name, value); give an hourly frame with one column per name."""
    df = sql(db, query)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.pivot(index="ts", columns="name", values="value").resample(freq).mean() * scale


def daily(db, query):
    """Run a query returning (day, name, value); give a frame indexed by date, one column per name."""
    df = sql(db, query)
    df["day"] = pd.to_datetime(df["day"])
    return df.pivot(index="day", columns="name", values="value")


def by_local_day(frame):
    """Hourly UTC frame -> daily means by German calendar day."""
    return frame.groupby(frame.index.tz_convert(TZ).normalize().tz_localize(None)).mean()


def lagged(day_frame, lag_days, limit, days):
    """Row for delivery day D gets the value dated D - lag_days. Gaps (weekends, weekly or monthly
    data) are filled forward at most `limit` days, so a feed that died turns into missing values."""
    full = day_frame.reindex(pd.date_range(day_frame.index.min(), days.max()))
    if limit:  # limit 0 = never fill
        full = full.ffill(limit=limit)
    return full.shift(lag_days).reindex(days)


def build():
    # --- target, and the hourly index everything else is aligned to
    ec = hourly("energy_charts.db", """
        SELECT ts, CASE WHEN name LIKE 'Day Ahead%' THEN 'price' WHEN name LIKE '%ID1%' THEN 'id1'
                        WHEN name LIKE '%ID3%' THEN 'id3' WHEN name LIKE 'Gas%' THEN 'gas' ELSE 'co2' END AS name, value
        FROM series
        WHERE (res='60' AND (name LIKE 'Day Ahead Auction (DE%' OR name LIKE 'Intraday Continuous ID_-Price%'
                             OR name='Gas (NCG, THE)' OR name='CO2 Emission Allowances, Auction DE'))
           OR (res='15' AND name='Day Ahead Auction (DE-LU)')""")  # 15-min prices from 2025-10 average to hourly
    smard = hourly("smard.db", "SELECT ts, series AS name, value FROM smard", scale=4)  # MWh per 15 min -> MW
    idx = pd.date_range(START, smard["load_forecast"].dropna().index.max(), freq="h", name="ts")
    local = idx.tz_convert(TZ)
    days = local.normalize().tz_localize(None)  # delivery day of each hour
    out = pd.DataFrame(index=idx)
    out["price"] = ec["price"].reindex(idx)

    # --- calendar
    out["hour"], out["weekday"], out["month"] = local.hour, local.day_of_week, local.month
    years = range(idx[0].year, idx[-1].year + 2)
    national = holidays.Germany(years=years)
    by_state = [holidays.Germany(subdiv=s, years=years) for s in STATES]
    day_list = days.unique()
    out["holiday"] = pd.Series([d in national for d in day_list], day_list).reindex(days).to_numpy().astype(int)
    out["holiday_state_share"] = pd.Series([sum(d in h for h in by_state) / len(STATES) for d in day_list],
                                           day_list).reindex(days).to_numpy()

    # --- forecasts for the delivery hour itself
    # Only the demand forecast: SMARD's wind and solar forecasts for a day appear the evening
    # before (not yet there at 17:40 on 2026-11-14), after the 12:00 auction, so a bid cannot use them.
    out["load_forecast"] = smard["load_forecast"].reindex(idx)

    wx = sql("weather.db", "SELECT * FROM weather")
    wx["ts"] = pd.to_datetime(wx["ts"], utc=True)
    wx = wx.set_index(["kind", "ts", "point"]).unstack("point")
    wx.columns = [f"wx_{point}_{var}" for var, point in wx.columns]
    d1, d2, live, actual_wx = (wx.xs(k, level="kind").reindex(idx)
                               for k in ("forecast_d1", "forecast_d2", "forecast", "actual"))
    # forecast_d1 was issued 24 h before its hour and takes about 4 h to arrive. That is in time for a
    # bid only for delivery hours before 06:00; later hours take the forecast issued 48 h before.
    in_time = d1.where(pd.Series(local.hour < 6, idx), d2, axis=0)
    # then the live forecast (hours still ahead), then actual weather (before 2024)
    out = out.join(in_time.combine_first(live).combine_first(actual_wx))
    out["wx_is_forecast"] = (in_time.notna().any(axis=1) | live.notna().any(axis=1)).astype(int)

    # --- hourly history, shifted to what is known the morning before delivery
    for h in (24, 48, 168):
        out[f"price_lag_{h}h"] = out["price"].shift(h)
    out = out.join(ec[["id1", "id3"]].reindex(idx).shift(48).add_prefix("intraday_").add_suffix("_lag_48h"))
    actual = smard[[c for c in smard if not c.endswith("_forecast")]].reindex(idx)
    out = out.join(actual.shift(48).add_suffix("_actual_lag_48h"))
    zones = hourly("neighbour_prices.db", "SELECT ts, zone AS name, price AS value FROM prices").reindex(idx)
    out = out.join(zones.shift(24).add_prefix("price_").add_suffix("_lag_24h"))

    # --- daily, weekly and monthly inputs, each with its publication lag
    # French nuclear power out of service at 08:00 German time on the forecast morning. From the newest
    # revision of each outage document, so an outage reported late counts from its start, not from the report.
    nuclear = sql("outages.db", "SELECT unit, nominal_mw, from_ts, to_ts, available_mw FROM outage o WHERE psr='B14'"
                                " AND status='A05' AND revision=(SELECT max(revision) FROM outage WHERE mrid=o.mrid)")
    at_8 = (pd.DatetimeIndex(days.unique()).tz_localize(TZ) + pd.Timedelta(hours=8)).tz_convert("UTC")
    # Checked against actual nuclear generation on 2026-11-18: the two add up to about 60 GW from 2019 on
    # (correlation -0.9), but in November and December 2018 some 6 GW of outages are missing, so 2018 is left empty.
    nuclear_out = pd.Series(out_of_service(nuclear, at_8), days.unique()).where(days.unique() >= "2019-01-01")
    stats = by_local_day(out[["price"]])["price"]
    hourly_price_by_day = out["price"].groupby(days)
    day_parts = [
        lagged(pd.DataFrame({"price_day_mean": stats, "price_day_min": hourly_price_by_day.min(),
                             "price_day_max": hourly_price_by_day.max()}), LAG_PRICE_STATS, 0, day_list),
        lagged(by_local_day(ec[["gas", "co2"]]).add_suffix("_price"), LAG_MARKET, 10, day_list),
        lagged(daily("commodities.db", "SELECT ts AS day, symbol AS name, close AS value FROM prices"
                                       " WHERE interval='1d'").rename(columns=SYMBOLS).add_prefix("cmd_"),
               LAG_MARKET, 10, day_list),
        lagged(daily("rhine.db", "SELECT ts AS day, 'rhine_flow_model' AS name, value FROM rhine"
                                 " WHERE series='kaub_flow_m3s_model'"), LAG_MARKET, 10, day_list),
        lagged(by_local_day(hourly("rhine.db", "SELECT ts, 'rhine_level_cm' AS name, value FROM rhine"
                                               " WHERE series='kaub_level_cm'")), LAG_MARKET, 10, day_list),
        lagged(daily("gas_storage.db", "SELECT gas_day AS day, 'gas_storage_' || lower(area) || '_full_pct' AS name,"
                                       " full_pct AS value FROM gas_storage"), LAG_GAS_STORAGE, 10, day_list),
        lagged(daily("gas_storage.db", "SELECT gas_day AS day, 'gas_storage_de_net_withdrawal' AS name,"
                                       " withdrawal_gwh_d - injection_gwh_d AS value FROM gas_storage"
                                       " WHERE area='DE'"), LAG_GAS_STORAGE, 10, day_list),
        lagged(daily("hydro.db", "SELECT ts AS day, 'hydro_' || lower(area) || '_fill_pct' AS name,"
                                 " fill_pct AS value FROM hydro WHERE area IN ('NO', 'NO2')"),
               LAG_HYDRO, 21, day_list),
        lagged(daily("hydro.db", "SELECT ts AS day, 'hydro_' || lower(area) || '_stored_twh' AS name,"
                                 " stored_twh AS value FROM hydro WHERE area IN ('CH', 'AT', 'FR')"),
               LAG_HYDRO_ENTSOE, 21, day_list),
        lagged(nuclear_out.to_frame("fr_nuclear_out_mw"), 1, 0, day_list),
        lagged(daily("worldbank.db", "SELECT ts AS day, name, value FROM monthly WHERE name IN (%s)"
                     % ",".join(f"'{n}'" for n in WORLDBANK))
               .rename(columns=lambda n: "wb_" + n.lower().replace(", ", "_").replace(" ", "_")),
               LAG_WORLDBANK, 45, day_list),
        lagged(daily("installed_capacity.db", "SELECT month AS day, name, value FROM capacity")
               .rename(columns=lambda n: "capacity_" + n.lower().replace(" (", "_").replace(")", "").replace(" ", "_")),
               0, 45, day_list),
    ]
    by_day = pd.concat(day_parts, axis=1)
    out = out.join(by_day.reindex(days).set_axis(idx))
    return out


def within_hour_deviation(frame):
    """Each quarter-hour value minus the average of its hour."""
    return frame - frame.groupby(frame.index.floor("h")).transform("mean")


def build_quarter_hours():
    """Table for the second, small model: how far each quarter-hour's price sits from its hour's average.

    The hourly table stays the main model because it has 7 years of prices. Quarter-hour prices only
    exist since the auction changed on 2025-10-01, so this table starts there and carries only what
    varies inside an hour. Join it to `training` on hour_ts for everything else.
    """
    first = "2025-09-30T22:00:00Z"
    q = hourly("energy_charts.db", "SELECT ts, 'price' AS name, value FROM series"
                                   " WHERE name='Day Ahead Auction (DE-LU)' AND res='15'", freq="15min")
    # demand only: SMARD's wind and solar forecasts are published after the auction (see build), and
    # yesterday's ramps, which are known in time, added nothing when tried (model_15min.py)
    fc = hourly("smard.db", "SELECT ts, series AS name, value FROM smard WHERE series='load_forecast'"
                            f" AND ts >= '{first}'", scale=4, freq="15min")  # MWh per 15 min -> MW
    idx = pd.date_range(first, fc["load_forecast"].dropna().index.max(), freq="15min", name="ts")
    q, fc = q.reindex(idx), within_hour_deviation(fc.reindex(idx))  # the ramp inside the hour

    out = pd.DataFrame(index=idx)
    out["hour_ts"] = idx.floor("h").strftime("%Y-%m-%dT%H:%M:%SZ")
    out["quarter"] = idx.minute // 15
    out["price"] = q["price"]
    out["deviation"] = within_hour_deviation(q)["price"]  # the target
    out["load_forecast_deviation"] = fc["load_forecast"]
    out["deviation_lag_24h"] = out["deviation"].shift(96)    # same quarter-hour yesterday, known before the auction
    out["deviation_lag_168h"] = out["deviation"].shift(672)  # same quarter-hour a week ago
    return out


def main():
    tables = {"training": build(), "training_15min": build_quarter_hours()}
    with closing(sqlite3.connect(DATA / "training.db", timeout=60)) as con:
        for name, out in tables.items():
            assert out.index.is_unique and out.columns.is_unique
            frame = out.reset_index()
            frame["ts"] = frame["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            frame.to_sql(name, con, if_exists="replace", index=False)
            con.execute(f"CREATE UNIQUE INDEX {name}_ts ON {name}(ts)")
            known = out["price"].notna()
            print(f"{name}: {len(out)} rows x {out.shape[1]} columns, {out.index[0]} to {out.index[-1]},"
                  f" price missing inside history: {(~known & (out.index <= out.index[known][-1])).sum()}")


def _selftest():
    days = pd.date_range("2026-01-01", "2026-01-10")
    closes = pd.DataFrame({"gas": [10.0, 11.0, 12.0]}, index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05"]))
    got = lagged(closes, 2, 1, days)["gas"]
    assert pd.isna(got["2026-01-02"])     # nothing known two days before
    assert got["2026-01-03"] == 10.0      # value of the 1st reaches delivery on the 3rd
    assert got["2026-01-05"] == 11.0      # the 3rd had no close: Friday's value is carried one day
    assert pd.isna(got["2026-01-06"])     # ... but not two days (limit=1)
    assert got["2026-01-07"] == 12.0
    assert pd.isna(got["2026-01-10"])     # a feed that stopped goes missing, it is not carried forever
    hours = pd.DataFrame({"x": [1.0, 3.0]}, index=pd.to_datetime(["2026-01-01T22:00Z", "2026-01-01T23:00Z"]))
    assert by_local_day(hours)["x"].to_dict() == {pd.Timestamp("2026-01-01"): 1.0, pd.Timestamp("2026-01-02"): 3.0}
    quarters = pd.DataFrame({"p": [10.0, 20.0, 30.0, 40.0, 7.0]},
                            index=pd.date_range("2026-01-01T00:00Z", periods=5, freq="15min"))
    assert within_hour_deviation(quarters)["p"].tolist() == [-15.0, -5.0, 5.0, 15.0, 0.0]
    print("selftest ok")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
