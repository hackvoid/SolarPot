"""Pull German power, gas and CO2 prices from energy-charts.info into data/energy_charts.db.

Reads the JSON files behind the site's "spot prices" chart (not the documented API: that one
has no intraday, gas or CO2 prices). Rows are upserted, so every run is safe to repeat.
    .venv/bin/python energy_charts.py                  # this week and last week; run every 15 min
    .venv/bin/python energy_charts.py --backfill 2018  # whole years from 2018, run once
    .venv/bin/python energy_charts.py --selftest
"""
import json
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

BASE = "https://www.energy-charts.info/charts/price_spot_market/data/de"
DB = Path(__file__).parent / "data" / "energy_charts.db"
RESOLUTIONS = (("", "60"), ("_15min", "15"))  # file name part, minutes per value
DAILY = ("CO2", "Gas", "2 ×")  # one value per day: keep the hourly copy, skip the 15-min one


def parse(doc, res):
    """Turn one chart file into (name, res, ts, value) tuples; ts is UTC."""
    ts = [datetime.fromtimestamp(x / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
          for x in doc[0]["xAxisValues"]]
    rows = []
    for series in doc:
        name = series["name"]  # a string, a {lang: text} dict, or a list holding that dict
        if isinstance(name, list):
            name = name[0]
        if isinstance(name, dict):
            name = name["en"]
        if res == "15" and name.startswith(DAILY):
            continue
        rows += [(name, res, t, float(v)) for t, v in zip(ts, series["data"], strict=True) if v is not None]
    return rows


def files(backfill_from, today):
    if backfill_from:
        return [(f"year{part}_{y}.json", res) for y in range(backfill_from, today.year + 1)
                for part, res in RESOLUTIONS]
    # last week too: intraday and balancing prices are still corrected after delivery.
    # Tomorrow too: on a Sunday the day-ahead prices for Monday are in next week's file.
    weeks = sorted({(today - timedelta(days=d)).isocalendar()[:2] for d in (7, 0, -1)})
    return [(f"week{part}_{y}_{w:02d}.json", res) for y, w in weeks for part, res in RESOLUTIONS]


def main(backfill_from=None):
    DB.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB, timeout=60)
    con.execute("CREATE TABLE IF NOT EXISTS series(name TEXT, res TEXT, ts TEXT, value REAL,"
                " PRIMARY KEY(name, res, ts)) WITHOUT ROWID")
    failed = []
    for fname, res in files(backfill_from, date.today()):
        try:
            req = Request(f"{BASE}/{fname}", headers={"User-Agent": "Mozilla/5.0 (price collector)"})
            with urlopen(req, timeout=600) as resp:
                rows = parse(json.load(resp), res)
            con.executemany("INSERT OR REPLACE INTO series VALUES (?,?,?,?)", rows)
            con.commit()
            print(f"{fname}: {len(rows)} rows")
        except Exception as e:  # one bad file must not stop the others
            failed.append(f"{fname}: {e!r}")
    if failed:
        sys.exit("ERROR " + "; ".join(failed))


def _selftest():
    doc = [
        {"name": {"en": "Load"}, "xAxisValues": [1794178800000, 1794179700000], "data": [1, None]},
        {"name": [{"en": "Day Ahead Auction (DE-LU)"}], "data": [180.25, -5]},
        {"name": "Gas (NCG, THE)", "data": [88.2, 88.2]},
    ]
    rows = parse(doc, "15")
    assert rows == [("Load", "15", "2026-11-08T23:00:00Z", 1.0),
                    ("Day Ahead Auction (DE-LU)", "15", "2026-11-08T23:00:00Z", 180.25),
                    ("Day Ahead Auction (DE-LU)", "15", "2026-11-08T23:15:00Z", -5.0)], rows
    assert len(parse(doc, "60")) == 5  # hourly keeps the daily gas series
    assert files(None, date(2026, 1, 2)) == [
        ("week_2025_52.json", "60"), ("week_15min_2025_52.json", "15"),
        ("week_2026_01.json", "60"), ("week_15min_2026_01.json", "15")]
    assert files(2025, date(2026, 1, 2))[-1] == ("year_15min_2026.json", "15")
    assert files(None, date(2026, 11, 15))[-1] == ("week_15min_2026_47.json", "15")  # Sunday: Monday's week too
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    elif "--backfill" in sys.argv:
        main(int(sys.argv[sys.argv.index("--backfill") + 1]))
    else:
        main()
