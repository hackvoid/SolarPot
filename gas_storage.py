"""Pull daily gas storage levels for Germany and the EU from AGSI+ into data/gas_storage.db.

Needs AGSI_API_KEY in .env. Rows are upserted; the first run loads history from 2018.
    .venv/bin/python gas_storage.py             # run twice a day
    .venv/bin/python gas_storage.py --selftest
"""
import sys
import time
from datetime import date, timedelta

from common import ROOT, fetch_json, open_db

API = "https://agsi.gie.eu/api"
AREAS = {"DE": "country=DE", "EU": "type=eu"}
# AGSI+ field -> our column: % full, TWh stored, GWh/day in, GWh/day out, TWh capacity
FIELDS = {"full": "full_pct", "gasInStorage": "stored_twh", "injection": "injection_gwh_d",
          "withdrawal": "withdrawal_gwh_d", "workingGasVolume": "capacity_twh"}


def number(text):
    """AGSI+ sends numbers as strings and '-' for unknown."""
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def to_rows(area, days):
    return [(area, d["gasDayStart"], *(number(d.get(f)) for f in FIELDS)) for d in days]


def main():
    env = dict(line.split("=", 1) for line in (ROOT / ".env").read_text().split())
    cols = ", ".join(f"{c} REAL" for c in FIELDS.values())
    con = open_db("gas_storage.db", f"CREATE TABLE IF NOT EXISTS gas_storage(area TEXT, gas_day TEXT, {cols},"
                                    " PRIMARY KEY(area, gas_day)) WITHOUT ROWID")
    today = date.today()
    for area, selector in AREAS.items():
        known = con.execute("SELECT 1 FROM gas_storage WHERE area=? LIMIT 1", (area,)).fetchone()
        start = today - timedelta(days=14) if known else date(2018, 1, 1)  # recent days get revised
        page, total = 1, 0
        while True:
            doc = fetch_json(f"{API}?{selector}&from={start}&to={today}&size=300&page={page}",
                             {"x-key": env["AGSI_API_KEY"]})
            if doc.get("error"):
                sys.exit(f"ERROR AGSI+ {doc['error']}: {doc.get('message')}")
            rows = to_rows(area, doc["data"])
            con.executemany(f"INSERT OR REPLACE INTO gas_storage VALUES (?,?,{','.join('?' * len(FIELDS))})", rows)
            con.commit()
            total += len(rows)
            if page >= doc["last_page"]:
                break
            page += 1
            time.sleep(1)  # AGSI+ allows 60 requests a minute
        print(f"{area}: {total} days from {start}")


def _selftest():
    day = {"gasDayStart": "2026-11-12", "full": "57.57", "gasInStorage": "142.4102", "injection": "102.51",
           "withdrawal": "358.8", "workingGasVolume": "-"}
    assert to_rows("DE", [day]) == [("DE", "2026-11-12", 57.57, 142.4102, 102.51, 358.8, None)]
    print("selftest ok")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
