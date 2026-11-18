"""Pull outages of French generation units from ENTSO-E into data/outages.db.

One row per document revision and block of constant available power. ENTSO-E hands out only the
newest revision of a document, so the history shows each outage as it finally stood, not as it was
first announced, and holds no cancelled documents. Every revision seen from the first run on is kept.
Needs ENTSOE_API_KEY in .env.
    .venv/bin/python outages.py             # run every hour; the first run loads history from 2018
    .venv/bin/python outages.py --selftest
"""
import io
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from common import entsoe, open_db

FRANCE = "10YFR-RTE------C"
PAGE = 200  # ENTSO-E sends at most this many documents per request
WHEN = "%Y%m%d%H%M"  # how ENTSO-E wants times, UTC
SCHEMA = """CREATE TABLE IF NOT EXISTS outage(mrid TEXT, revision INTEGER, unit TEXT, from_ts TEXT, to_ts TEXT,
    available_mw REAL, nominal_mw REAL, unit_name TEXT, psr TEXT, business TEXT, status TEXT, created TEXT,
    PRIMARY KEY(mrid, revision, unit, from_ts)) WITHOUT ROWID"""
# psr = plant type (B14 nuclear), business = A53 planned or A54 forced, status = A05 active, A09 cancelled, A13 withdrawn


def stamp(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse(xml):
    """One outage document -> rows in table order. Inside a block the available power changes at each point."""
    doc = ET.fromstring(re.sub(r' xmlns="[^"]+"', "", xml, count=1))
    rows = []
    for series in doc.iter("TimeSeries"):
        plant = "production_RegisteredResource.pSRType."
        nominal = series.findtext(plant + "powerSystemResources.nominalP")
        for block in series.iter("Available_Period"):
            start, end = (pd.Timestamp(block.findtext(f"timeInterval/{edge}")) for edge in ("start", "end"))
            step = pd.Timedelta(block.findtext("resolution"))
            points = sorted((int(p.findtext("position")), float(p.findtext("quantity"))) for p in block.iter("Point"))
            begins = [start + (position - 1) * step for position, _ in points]
            for begin, until, (_, mw) in zip(begins, begins[1:] + [end], points):
                rows.append((doc.findtext("mRID"), int(doc.findtext("revisionNumber")),
                             series.findtext(plant + "powerSystemResources.mRID"), stamp(begin), stamp(until), mw,
                             nominal and float(nominal), series.findtext(plant + "powerSystemResources.name"),
                             series.findtext(plant + "psrType"), series.findtext("businessType"),
                             doc.findtext("docStatus/value") or "A05", doc.findtext("createdDateTime")))
    return rows


def fetch_rows(query, split=("periodStart", "periodEnd")):
    """Every document matching one query, as rows. A full answer may be cut short, and ENTSO-E's paging
    (offset) loses documents: 75 of 491 for January 2024, checked 2026-11-18. So a full answer is asked
    again as the two halves of the time range named by `split`."""
    # offset=0: without it ENTSO-E answers a query with more than 200 documents with an error
    body = entsoe(documentType="A80", biddingZone_Domain=FRANCE, offset=0, **query)
    if body[:2] != b"PK":  # an XML note instead of a zip: nothing found
        return []
    with zipfile.ZipFile(io.BytesIO(body)) as files:
        names = files.namelist()
        if len(names) < PAGE:
            return [row for name in names for row in parse(files.read(name).decode())]
    a, b = (pd.to_datetime(query[edge], format=WHEN) for edge in split)
    assert b - a > pd.Timedelta(minutes=1), f"{PAGE} or more documents in one minute: {query[split[0]]}"
    middle = f"{(a + (b - a) / 2).floor('min'):{WHEN}}"
    return fetch_rows({**query, split[1]: middle}, split) + fetch_rows({**query, split[0]: middle}, split)


def out_of_service(blocks, times):
    """MW out of service at each time. blocks: unit, nominal_mw, from_ts, to_ts, available_mw; times: sorted,
    UTC. Documents for one unit can overlap, so per unit only the largest shortfall counts."""
    lost = {}
    first = times.searchsorted(pd.to_datetime(blocks["from_ts"], utc=True))  # a block covers from_ts <= t < to_ts
    after = times.searchsorted(pd.to_datetime(blocks["to_ts"], utc=True))
    shortfall = (blocks["nominal_mw"] - blocks["available_mw"]).clip(lower=0)
    for unit, a, b, mw in zip(blocks["unit"], first, after, shortfall):
        row = lost.setdefault(unit, np.zeros(len(times)))
        row[a:b] = np.maximum(row[a:b], mw)
    return sum(lost.values(), np.zeros(len(times)))


def main():
    con = open_db("outages.db", SCHEMA)
    now = pd.Timestamp.now("UTC").floor("h")
    newest = con.execute("SELECT max(created) FROM outage").fetchone()[0]
    if newest:  # only documents changed since the last run (a day of overlap), for outages from a month back on
        split = ("periodStartUpdate", "periodEndUpdate")
        changed = {"periodStart": f"{now - pd.Timedelta(days=30):{WHEN}}", "periodEnd": f"{now + pd.Timedelta(days=335):{WHEN}}",
                   "periodStartUpdate": f"{pd.Timestamp(newest) - pd.Timedelta(days=1):{WHEN}}",
                   "periodEndUpdate": f"{now + pd.Timedelta(hours=1):{WHEN}}"}
        # active and cancelled come by default; withdrawn only on request. Both end an outage we hold as active.
        queries = [changed, {**changed, "docStatus": "A13"}]
    else:  # first run: every outage touching each week since 2018-09, and those planned for the year ahead.
        # Active documents only: cancelled ones never count, and 327 of them overlap 2018-09-01 00:00 alone.
        split = ("periodStart", "periodEnd")
        weeks = pd.date_range("2018-09-01", (now + pd.Timedelta(days=372)).tz_localize(None), freq="7D")
        queries = [{"periodStart": f"{a:{WHEN}}", "periodEnd": f"{b:{WHEN}}", "docStatus": "A05"} for a, b in zip(weeks, weeks[1:])]
    total = 0
    with ThreadPoolExecutor(6) as pool:
        for rows in pool.map(lambda query: fetch_rows(query, split), queries):
            con.executemany("INSERT OR REPLACE INTO outage VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            con.commit()
            total += len(rows)
    print(f"{stamp(now)}: {total} rows from {len(queries)} queries")


def _selftest():
    xml = """<Unavailability_MarketDocument xmlns="urn:x"><mRID>m1</mRID><revisionNumber>3</revisionNumber>
      <createdDateTime>2026-11-17T15:05:05Z</createdDateTime><docStatus><value>A09</value></docStatus>
      <TimeSeries><businessType>A54</businessType>
        <production_RegisteredResource.pSRType.psrType>B14</production_RegisteredResource.pSRType.psrType>
        <production_RegisteredResource.pSRType.powerSystemResources.mRID>U1</production_RegisteredResource.pSRType.powerSystemResources.mRID>
        <production_RegisteredResource.pSRType.powerSystemResources.name>UNIT 1</production_RegisteredResource.pSRType.powerSystemResources.name>
        <production_RegisteredResource.pSRType.powerSystemResources.nominalP unit="MAW">900.0</production_RegisteredResource.pSRType.powerSystemResources.nominalP>
        <Available_Period><timeInterval><start>2026-11-17T00:00Z</start><end>2026-11-17T03:00Z</end></timeInterval>
          <resolution>PT60M</resolution>
          <Point><position>1</position><quantity>0</quantity></Point>
          <Point><position>3</position><quantity>450</quantity></Point></Available_Period></TimeSeries>
    </Unavailability_MarketDocument>"""
    tail = (900.0, "UNIT 1", "B14", "A54", "A09", "2026-11-17T15:05:05Z")
    assert parse(xml) == [("m1", 3, "U1", "2026-11-17T00:00:00Z", "2026-11-17T02:00:00Z", 0.0, *tail),
                          ("m1", 3, "U1", "2026-11-17T02:00:00Z", "2026-11-17T03:00:00Z", 450.0, *tail)], parse(xml)
    blocks = pd.DataFrame([("U1", 900.0, "2026-11-17T00:00:00Z", "2026-11-17T02:00:00Z", 0.0),
                           ("U1", 900.0, "2026-11-17T01:00:00Z", "2026-11-17T04:00:00Z", 450.0),  # overlaps: no double count
                           ("U2", 1300.0, "2026-11-17T03:00:00Z", "2026-11-17T04:00:00Z", 300.0)],
                          columns=["unit", "nominal_mw", "from_ts", "to_ts", "available_mw"])
    times = pd.date_range("2026-11-17T00:00Z", periods=5, freq="h")
    assert out_of_service(blocks, times).tolist() == [900, 900, 450, 1450, 0]
    print("selftest ok")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
