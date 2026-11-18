"""Pull Germany's monthly installed wind, solar, biomass and battery capacity into data/installed_capacity.db.

From the Energy-Charts API (its source is the national plant register). GW, except
'Battery storage (capacity)' which is GWh. Monthly from 2002; recent months are revised
upward as late registrations arrive, so every run reloads the whole series.
    .venv/bin/python installed_capacity.py             # run once a week
    .venv/bin/python installed_capacity.py --selftest
"""
import sys

from common import fetch_json, open_db

API = "https://api.energy-charts.info/installed_power?country=de&time_step=monthly&installation_decommission=false"


def to_rows(doc):
    """'MM.YYYY' labels and one list per technology -> (name, 'YYYY-MM-01', value) tuples."""
    months = [f"{t[3:]}-{t[:2]}-01" for t in doc["time"]]
    return [(p["name"], m, float(v)) for p in doc["production_types"]
            for m, v in zip(months, p["data"], strict=True) if v is not None]


def main():
    rows = to_rows(fetch_json(API))
    if not any(r[0] == "Solar AC" for r in rows):
        sys.exit("ERROR reply changed: 'Solar AC' not found")
    con = open_db("installed_capacity.db", "CREATE TABLE IF NOT EXISTS capacity(name TEXT, month TEXT,"
                                           " value REAL, PRIMARY KEY(name, month)) WITHOUT ROWID")
    con.executemany("INSERT OR REPLACE INTO capacity VALUES (?,?,?)", rows)
    con.commit()
    print(f"{len(rows)} rows, latest month {max(r[1] for r in rows)}")


def _selftest():
    doc = {"time": ["10.2026", "11.2026"], "production_types": [{"name": "Solar AC", "data": [120.5, None]}]}
    assert to_rows(doc) == [("Solar AC", "2026-10-01", 120.5)]
    print("selftest ok")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
