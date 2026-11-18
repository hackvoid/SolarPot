"""Pull Rhine water level and flow at Kaub into data/rhine.db.

Series, all upserted:
  kaub_level_cm             gauge, every 15 min - the official feed only keeps the last 31 days
  kaub_flow_m3s             gauge, every 15 min, same limit
  kaub_flow_m3s_model       daily flow from Open-Meteo's river model, from 2018 - fills the history
    .venv/bin/python rhine.py             # run hourly
    .venv/bin/python rhine.py --selftest
"""
import sys
from datetime import date, datetime, timedelta, timezone

from common import fetch_json, open_db

GAUGE = "https://www.pegelonline.wsv.de/webservices/rest-api/v2/stations/KAUB/{}/measurements.json?start=P31D"
GAUGE_SERIES = {"W": "kaub_level_cm", "Q": "kaub_flow_m3s"}
# the model grid cell that holds the Rhine next to Kaub; the cell at the gauge itself is a side stream
MODEL = "https://flood-api.open-meteo.com/v1/flood?latitude=50.025&longitude=7.775&daily=river_discharge"


def utc(stamp):
    """'2026-10-14T00:15:00+02:00' -> '2026-10-13T22:15:00Z'"""
    return datetime.fromisoformat(stamp).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    con = open_db("rhine.db", "CREATE TABLE IF NOT EXISTS rhine(series TEXT, ts TEXT, value REAL,"
                              " PRIMARY KEY(series, ts)) WITHOUT ROWID")
    rows = []
    for code, series in GAUGE_SERIES.items():
        rows += [(series, utc(m["timestamp"]), m["value"]) for m in fetch_json(GAUGE.format(code))]

    today = date.today()
    has_model = con.execute("SELECT 1 FROM rhine WHERE series='kaub_flow_m3s_model' LIMIT 1").fetchone()
    start = today - timedelta(days=30) if has_model else date(2018, 1, 1)
    daily = fetch_json(f"{MODEL}&start_date={start}&end_date={today}")["daily"]
    rows += [("kaub_flow_m3s_model", t, v) for t, v in zip(daily["time"], daily["river_discharge"]) if v is not None]

    con.executemany("INSERT OR REPLACE INTO rhine VALUES (?,?,?)", rows)
    con.commit()
    print(f"{len(rows)} rows")


def _selftest():
    assert utc("2026-10-14T00:15:00+02:00") == "2026-10-13T22:15:00Z"
    assert utc("2026-11-13T20:15:00+01:00") == "2026-11-13T19:15:00Z"
    print("selftest ok")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
