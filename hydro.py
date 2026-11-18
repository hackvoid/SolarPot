"""Pull weekly hydro reservoir levels for Norway, Switzerland, Austria and France into data/hydro.db.

Areas: NO (all Norway), NO1..NO5 (its price zones; NO2 holds the cable to Germany) from the water
directorate NVE, from 1995, with fill_pct. CH, AT, FR from ENTSO-E (needs ENTSOE_API_KEY in .env), from
2015, stored energy only, dated by the Monday the week starts.
CH rows that carry fill_pct, up to 2026-09-28, came from the Swiss energy office file, which stopped then.
ENTSO-E's Swiss numbers equal that file from 2025 on but cover only 45 to 87% of it before, so those rows
are kept and ENTSO-E only adds the weeks after them.
    .venv/bin/python hydro.py             # run once a day
    .venv/bin/python hydro.py --selftest
"""
import re
import sys
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from common import entsoe, fetch_json, open_db

NORWAY = "https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk/HentOffentligData"
ENTSOE_ZONES = {"CH": "10YCH-SWISSGRIDZ", "AT": "10YAT-APG------L", "FR": "10YFR-RTE------C"}


def norway_rows(weeks):
    """NVE rows -> (area, ts, fill_pct, stored_twh, capacity_twh). Skips its water-region rows."""
    rows = []
    for w in weeks:
        if w["omrType"] in ("NO", "EL"):
            area = "NO" if w["omrType"] == "NO" else f"NO{w['omrnr']}"
            rows.append((area, w["dato_Id"], w["fyllingsgrad"] * 100, w["fylling_TWh"], w["kapasitet_TWh"]))
    return rows


def entsoe_rows(area, xml):
    """ENTSO-E reservoir document (MWh per week) -> (area, Monday the week starts, stored_twh)."""
    doc = ET.fromstring(re.sub(r' xmlns="[^"]+"', "", xml, count=1))
    rows = []
    for period in doc.iter("Period"):
        start = datetime.fromisoformat(period.findtext("timeInterval/start")).astimezone(ZoneInfo("Europe/Berlin")).date()
        for point in period.iter("Point"):
            week = start + timedelta(weeks=int(point.findtext("position")) - 1)
            rows.append((area, str(week), float(point.findtext("quantity")) / 1e6))
    return rows


def main():
    con = open_db("hydro.db", "CREATE TABLE IF NOT EXISTS hydro(area TEXT, ts TEXT, fill_pct REAL,"
                              " stored_twh REAL, capacity_twh REAL, PRIMARY KEY(area, ts)) WITHOUT ROWID")
    # after the first run only the newest week is needed from Norway (the full file is 4 MB)
    known = con.execute("SELECT 1 FROM hydro WHERE area='NO' LIMIT 1").fetchone()
    rows = norway_rows(fetch_json(NORWAY + ("SisteUke" if known else "")))
    con.executemany("INSERT OR REPLACE INTO hydro VALUES (?,?,?,?,?)", rows)
    weeks = []
    for area, zone in ENTSOE_ZONES.items():
        # from the year of the newest stored week (recent weeks get revised); an area with no rows yet from 2015
        newest = con.execute("SELECT max(ts) FROM hydro WHERE area=?", (area,)).fetchone()[0]
        for year in range(int(newest[:4]) if newest else 2015, date.today().year + 1):
            weeks += entsoe_rows(area, entsoe(documentType="A72", processType="A16", in_Domain=zone,
                                              periodStart=f"{year}01010000", periodEnd=f"{year + 1}01010000").decode())
    # rows with fill_pct are the old Swiss file's: never overwritten (see the top of this file)
    con.executemany("INSERT INTO hydro(area, ts, stored_twh) VALUES (?,?,?) ON CONFLICT(area, ts)"
                    " DO UPDATE SET stored_twh=excluded.stored_twh WHERE hydro.fill_pct IS NULL", weeks)
    con.commit()
    print(f"{len(rows) + len(weeks)} rows")


def _selftest():
    weeks = [{"omrType": "EL", "omrnr": 2, "dato_Id": "2026-11-08", "fyllingsgrad": 0.5,
              "fylling_TWh": 16.0, "kapasitet_TWh": 32.0},
             {"omrType": "VASS", "omrnr": 1, "dato_Id": "2026-11-08", "fyllingsgrad": 0.9,
              "fylling_TWh": 1.0, "kapasitet_TWh": 1.1}]
    assert norway_rows(weeks) == [("NO2", "2026-11-08", 50.0, 16.0, 32.0)]
    xml = """<GL_MarketDocument xmlns="urn:x"><TimeSeries><Period><timeInterval><start>2026-10-18T22:00Z</start>
        <end>2026-11-08T23:00Z</end></timeInterval><resolution>P7D</resolution>
        <Point><position>1</position><quantity>6378000</quantity></Point>
        <Point><position>2</position><quantity>6258000</quantity></Point></Period></TimeSeries></GL_MarketDocument>"""
    # the first week starts in summer time, the second in winter time
    assert entsoe_rows("CH", xml) == [("CH", "2026-10-19", 6.378), ("CH", "2026-10-26", 6.258)], entsoe_rows("CH", xml)
    print("selftest ok")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
