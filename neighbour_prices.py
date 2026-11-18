"""Pull day-ahead prices of the zones wired to Germany from the Energy-Charts API into data/neighbour_prices.db.

EUR/MWh, hourly until 2025-09-30 and quarter-hourly after. Rows are upserted; a zone with no
rows yet is loaded from 2018. Germany's own price is in energy_charts.db, not here.
    .venv/bin/python neighbour_prices.py             # run a few times a day; takes about 4 min
    .venv/bin/python neighbour_prices.py --selftest
"""
import sys
import time
from datetime import date, datetime, timedelta, timezone

from common import fetch_json, open_db

API = "https://api.energy-charts.info/price"
# Austria, Belgium, Switzerland, Czechia, Denmark west/east, France, Netherlands,
# south Norway (NordLink cable), Poland, south Sweden (Baltic Cable)
ZONES = ["AT", "BE", "CH", "CZ", "DK1", "DK2", "FR", "NL", "NO2", "PL", "SE4"]


def to_rows(zone, doc):
    return [(zone, datetime.fromtimestamp(s, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), float(p))
            for s, p in zip(doc["unix_seconds"], doc["price"], strict=True) if p is not None]


def main():
    con = open_db("neighbour_prices.db", "CREATE TABLE IF NOT EXISTS prices(zone TEXT, ts TEXT, price REAL,"
                                         " PRIMARY KEY(zone, ts)) WITHOUT ROWID")
    end = date.today() + timedelta(days=2)  # tomorrow's prices are published around 13:00
    failed = []
    for i, zone in enumerate(ZONES):
        if i:
            time.sleep(20)  # the API allows only a few requests a minute
        last = con.execute("SELECT max(ts) FROM prices WHERE zone=?", (zone,)).fetchone()[0]
        start = date.fromisoformat(last[:10]) - timedelta(days=3) if last else date(2018, 1, 1)
        try:
            rows = to_rows(zone, fetch_json(f"{API}?bzn={zone}&start={start}&end={end}"))
        except Exception as e:  # one zone failing must not stop the others
            failed.append(f"{zone}: {e!r}")
            continue
        con.executemany("INSERT OR REPLACE INTO prices VALUES (?,?,?)", rows)
        con.commit()
        print(f"{zone}: {len(rows)} rows from {start}")
    if failed:
        sys.exit("ERROR " + "; ".join(failed))


def _selftest():
    doc = {"unix_seconds": [1794178800, 1794179700], "price": [181.52, None]}
    assert to_rows("AT", doc) == [("AT", "2026-11-08T23:00:00Z", 181.52)]
    print("selftest ok")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
