"""Pull World Bank monthly commodity prices (the "Pink Sheet") into data/worldbank.db.

71 series from 1960, nominal USD, updated early each month: coal, European gas, oil, metals,
fertiliser, food. Fills what the other collectors lack (coal, nickel, tin, lead).
    .venv/bin/python worldbank.py             # run once a day
    .venv/bin/python worldbank.py --selftest
"""
import io
import re
import sqlite3
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

PAGE = "https://www.worldbank.org/en/research/commodity-markets"
LINK = re.compile(r"https://thedocs\.worldbank\.org/[^\"']*CMO-Historical-Data-Monthly\.xlsx")
DB = Path(__file__).parent / "data" / "worldbank.db"


def get(url):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (price collector)"})
    with urlopen(req, timeout=120) as resp:
        return resp.read()


def parse(df):
    """Turn the 'Monthly Prices' sheet (read with header=None) into (name, unit, ts, value) tuples."""
    first = df.index[df[0].astype(str).str.fullmatch(r"\d{4}M\d{2}")][0]  # first data row
    data = df.loc[first:]
    ts = data[0].str.replace("M", "-") + "-01"
    rows = []
    for col in df.columns[1:]:
        name = str(df.loc[first - 2, col]).strip(" *")
        unit = str(df.loc[first - 1, col]).strip("()")
        values = pd.to_numeric(data[col], errors="coerce")  # the sheet marks missing months with "…"
        rows += [(name, unit, t, float(v)) for t, v in zip(ts, values) if pd.notna(v)]
    return rows


def main():
    # the file's address changes every year, so read it from the landing page
    link = LINK.search(get(PAGE).decode("utf-8", "replace"))
    if not link:
        sys.exit("ERROR no monthly price file linked on " + PAGE)
    rows = parse(pd.read_excel(io.BytesIO(get(link.group())), sheet_name="Monthly Prices", header=None))
    if not any(r[0] == "Coal, Australian" for r in rows):
        sys.exit("ERROR sheet layout changed: 'Coal, Australian' not found")
    DB.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB, timeout=60)
    con.execute("CREATE TABLE IF NOT EXISTS monthly(name TEXT, unit TEXT, ts TEXT, value REAL,"
                " PRIMARY KEY(name, ts)) WITHOUT ROWID")
    con.executemany("INSERT OR REPLACE INTO monthly VALUES (?,?,?,?)", rows)
    con.commit()
    print(f"{len(rows)} rows, latest month {max(r[2] for r in rows)}")


def _selftest():
    df = pd.DataFrame([["Pink Sheet", None, None], [None, "Coal, Australian", "Beef **"],
                       [None, "($/mt)", "($/kg)"], ["2026M09", 147.1, "…"], ["2026M10", 152.3, 6.5]])
    assert parse(df) == [("Coal, Australian", "$/mt", "2026-09-01", 147.1),
                         ("Coal, Australian", "$/mt", "2026-10-01", 152.3),
                         ("Beef", "$/kg", "2026-10-01", 6.5)], parse(df)
    print("selftest ok")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
