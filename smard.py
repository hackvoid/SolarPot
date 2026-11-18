"""Pull German wind, solar and demand - actual and day-ahead forecast - from SMARD into data/smard.db.

SMARD is the Federal Network Agency's market data site. Values are MWh per quarter-hour
(multiply by 4 for MW). Rows are upserted; a series with no rows yet is backfilled from 2018.
    .venv/bin/python smard.py             # run every 15 min; the first run takes a few minutes
    .venv/bin/python smard.py --selftest
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from common import fetch_json, open_db

BASE = "https://www.smard.de/app/chart_data/{id}/DE"
SERIES = {  # our name: SMARD series number
    "wind_onshore": 4067, "wind_offshore": 1225, "solar": 4068, "load": 410,
    "wind_onshore_forecast": 123, "wind_offshore_forecast": 3791, "solar_forecast": 125, "load_forecast": 411,
}
FIRST = datetime(2017, 12, 25, tzinfo=timezone.utc).timestamp() * 1000  # the weekly file holding 2018-01-01


def stamp(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_rows(name, doc):
    return [(name, stamp(ms), float(v)) for ms, v in doc["series"] if v is not None]


def load(task):
    """Fetch one weekly file. A file that fails is reported, not fatal."""
    name, week = task
    url = f"{BASE}/{{id}}_DE_quarterhour_{week}.json".format(id=SERIES[name])
    try:
        return to_rows(name, fetch_json(url)), None
    except Exception as e:
        return [], f"{name} {stamp(week)}: {e!r}"


def main():
    con = open_db("smard.db", "CREATE TABLE IF NOT EXISTS smard(series TEXT, ts TEXT, value REAL,"
                              " PRIMARY KEY(series, ts)) WITHOUT ROWID")
    tasks = []
    for name, number in SERIES.items():
        weeks = fetch_json(f"{BASE}/index_quarterhour.json".format(id=number))["timestamps"]
        known = con.execute("SELECT 1 FROM smard WHERE series=? LIMIT 1", (name,)).fetchone()
        # later runs re-read last week too: SMARD corrects values for days after delivery
        tasks += [(name, w) for w in (weeks[-2:] if known else weeks) if w >= FIRST]

    written, failed = 0, []
    with ThreadPoolExecutor(4) as pool:  # 4 at a time, like a browser loading the SMARD chart
        for rows, error in pool.map(load, tasks):
            con.executemany("INSERT OR REPLACE INTO smard VALUES (?,?,?)", rows)
            written += len(rows)
            if error:
                failed.append(error)
    con.commit()

    # logged every run so the log shows at what time of day tomorrow's forecast gets published
    horizon = dict(con.execute("SELECT series, max(ts) FROM smard"
                               " WHERE series IN ('solar_forecast', 'load_forecast') GROUP BY series"))
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    print(f"{now}: {written} rows from {len(tasks)} files, forecast reaches {horizon.get('solar_forecast')},"
          f" load forecast reaches {horizon.get('load_forecast')}")
    if failed:
        sys.exit(f"ERROR {len(failed)} files failed: " + "; ".join(failed[:5]))


def _selftest():
    doc = {"series": [[1794178800000, 5250.3], [1794179700000, None]]}
    assert to_rows("wind_onshore", doc) == [("wind_onshore", "2026-11-08T23:00:00Z", 5250.3)]
    print("selftest ok")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
