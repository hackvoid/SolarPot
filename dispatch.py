"""Step 6: turn the quarter-hour price forecast into a battery schedule, and score what that earns.

The battery trades in the day-ahead auction as a price taker: its schedule is fixed from the forecast
before the auction and is paid the real prices afterwards.
    .venv/bin/python dispatch.py              # schedule for the newest day in data/forecast.db -> table plan
    .venv/bin/python dispatch.py --backtest   # last 180 days: earnings planned on our forecast, on
                                              # yesterday's prices and on the real prices (about 1 min)
    .venv/bin/python dispatch.py --selftest
"""
import sqlite3
import sys
from contextlib import closing

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from build_dataset import DATA, TZ, sql
from forecast import store, stored

# Placeholder battery until the real one is known. Earnings scale with size as long as
# CAPACITY_KWH / POWER_KW (hours to fill) stays the same.
CAPACITY_KWH = 1000.0
POWER_KW = 500.0    # limit for charging and for discharging
EFFICIENCY = 0.90   # energy out per energy in over a full round trip
STEP_H = 0.25


def schedule(prices):
    """Charge and discharge power (kW) per quarter-hour that earns the most at these prices (EUR/MWh).

    shortcut: each day is planned alone and starts empty, so energy left at midnight has no value and
    the plan ends empty. Plan two days and keep the first when carrying energy overnight matters.
    shortcut: no cost for battery wear, so small price gaps are traded too. Add a cost per kWh moved
    when the real battery's cycle life is known.
    """
    n, one_way = len(prices), EFFICIENCY ** 0.5
    eur_per_kw = np.asarray(prices, float) * STEP_H / 1000
    stored_after = np.tril(np.ones((n, n))) * STEP_H  # running total: energy in the battery after each step
    eye, zero = np.eye(n), np.zeros((n, n))
    # variables: charge (n), discharge (n), charging yes/no (n). Without the yes/no a battery could
    # charge and discharge at once to burn energy when prices are negative.
    rows = np.block([[stored_after * one_way, -stored_after / one_way, zero],
                     [eye, zero, -POWER_KW * eye],   # charge only while charging = 1
                     [zero, eye, POWER_KW * eye]])   # discharge only while charging = 0
    lower = np.r_[np.zeros(n), np.full(2 * n, -np.inf)]
    upper = np.r_[np.full(n, CAPACITY_KWH), np.zeros(n), np.full(n, POWER_KW)]
    result = milp(np.r_[eur_per_kw, -eur_per_kw, np.zeros(n)], constraints=LinearConstraint(rows, lower, upper),
                  integrality=np.r_[np.zeros(2 * n), np.ones(n)],
                  bounds=Bounds(0, np.r_[np.full(2 * n, POWER_KW), np.ones(n)]))
    assert result.success, result.message
    return result.x[:n], result.x[n:2 * n]


def earnings(prices, charge, discharge):
    """EUR earned by a schedule (kW) at these prices (EUR/MWh)."""
    return float((np.asarray(prices) * (discharge - charge)).sum() * STEP_H / 1000)


def backtest():
    q = sql("backtest.db", "SELECT ts, price, hour_forecast + model AS forecast FROM backtest_15min").set_index("ts")
    q.index = pd.to_datetime(q.index, utc=True)
    q["yesterday"], q["real prices"] = q["price"].shift(96), q["price"]
    q = q.dropna()
    plans = ["yesterday", "forecast", "real prices"]
    rows = []
    for _, day in q.groupby(q.index.tz_convert(TZ).normalize()):
        done = {name: schedule(day[name]) for name in plans}
        rows.append({**{name: earnings(day["price"], *done[name]) for name in plans},
                     "cycles": done["forecast"][1].sum() * STEP_H / CAPACITY_KWH})
    days = pd.DataFrame(rows)
    earned = days[plans]
    print(f"{len(days)} days, battery {CAPACITY_KWH:.0f} kWh / {POWER_KW:.0f} kW, round trip {EFFICIENCY:.0%}."
          " Schedule planned on each price series, paid at the real prices:")
    print(pd.DataFrame({"EUR per day": earned.mean(), "EUR per year": earned.mean() * 365,
                        "share of the best possible": earned.sum() / earned["real prices"].sum(),
                        "days with a loss": (earned < 0).sum()}).round(2).to_string())
    print(f"schedule planned on the forecast: {days['cycles'].mean():.2f} full discharges per day")


def main():
    with closing(sqlite3.connect(DATA / "forecast.db", timeout=60)) as con:
        con.execute("CREATE TABLE IF NOT EXISTS plan(ts TEXT PRIMARY KEY, price REAL, charge_kw REAL,"
                    " discharge_kw REAL, stored_kwh REAL)")
        q = stored(con, "forecast_15min")
        local = q.index.tz_convert(TZ)
        day = q[local.normalize() == local.normalize().max()]
        charge, discharge = schedule(day["price"])
        one_way = EFFICIENCY ** 0.5
        plan = pd.DataFrame({"price": day["price"], "charge_kw": charge.round(1), "discharge_kw": discharge.round(1),
                             "stored_kwh": ((charge * one_way - discharge / one_way).cumsum() * STEP_H).round(1) + 0})  # + 0: no "-0.0"
        store(con, "plan", plan)
        con.commit()
        print(f"schedule for {local.max():%Y-%m-%d}, battery {CAPACITY_KWH:.0f} kWh / {POWER_KW:.0f} kW,"
              f" expected to earn {earnings(day['price'], charge, discharge):.2f} EUR at the forecast prices")
        hours = plan.set_axis(day.index.tz_convert(TZ)).resample("h").agg(
            {"price": "mean", "charge_kw": "mean", "discharge_kw": "mean", "stored_kwh": "last"})
        print(hours.set_axis(hours.index.strftime("%H:%M")).round(1).to_string())


def _selftest():
    cheap_then_dear = [10.0] * 4 + [100.0] * 4
    charge, discharge = schedule(cheap_then_dear)
    # 500 kWh bought in the cheap hour; 450 kWh come back in the dear hour, in any split over its quarters
    assert np.allclose(charge, [500] * 4 + [0] * 4) and np.allclose(discharge[:4], 0), (charge, discharge)
    assert abs(discharge.sum() * STEP_H - 450) < 1e-6, discharge
    assert abs(earnings(cheap_then_dear, charge, discharge) - 40) < 1e-6  # 450 kWh sold at 0.10, 500 bought at 0.01
    charge, discharge = schedule([50.0] * 8)
    assert np.allclose(charge, 0) and np.allclose(discharge, 0)  # flat prices: every trade loses to the round trip
    charge, discharge = schedule([-50.0] * 8)
    assert np.allclose(charge, 500) and np.allclose(discharge, 0)  # paid to charge; never both at once
    print("selftest ok")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else backtest() if "--backtest" in sys.argv else main()
