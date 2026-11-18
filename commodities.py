"""Pull commodity prices from Yahoo Finance into data/commodities.db.

Safe to run every 15 minutes: rows are upserted on (symbol, interval, ts).
    .venv/bin/python commodities.py             # backfill on first run, then update
    .venv/bin/python commodities.py --selftest
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

SYMBOLS = {
    # fuels
    "TTF=F": "gas_ttf_eur_mwh", "NG=F": "gas_henry_hub", "BZ=F": "oil_brent", "CL=F": "oil_wti",
    "HO=F": "heating_oil", "RB=F": "gasoline",
    "MTF=F": "coal_api2",  # Yahoo stopped updating this on 2025-12-26; kept for its history
    # carbon: exchange-traded proxies, Yahoo has no EUA future
    "CO2.L": "eua_carbon_etc_eur", "CARB.L": "eua_carbon_etc_usd", "KRBN": "global_carbon_etf",
    # metals
    "GC=F": "gold", "SI=F": "silver", "HG=F": "copper", "PL=F": "platinum", "PA=F": "palladium",
    "ALI=F": "aluminium", "HRC=F": "steel_hrc", "TIO=F": "iron_ore", "ZNC=F": "zinc",
    "LIT": "lithium_etf",  # proxy: no lithium future with history on Yahoo
    # agriculture and other
    "ZC=F": "corn", "ZW=F": "wheat", "ZS=F": "soybeans", "ZO=F": "oats", "ZR=F": "rice",
    "SB=F": "sugar", "KC=F": "coffee", "CC=F": "cocoa", "CT=F": "cotton", "OJ=F": "orange_juice",
    "LBR=F": "lumber", "LE=F": "live_cattle",
    # needed to convert USD prices to EUR
    "EURUSD=X": "eurusd",
}

# interval: (period when the table has no rows for it, period on later runs; None = never again)
# 1h and 1mo are backfill only: TTF has no daily history on Yahoo, so these fill its past.
PERIODS = {"15m": ("60d", "5d"), "1d": ("max", "1mo"), "1h": ("730d", None), "1mo": ("max", None)}

DB = Path(__file__).parent / "data" / "commodities.db"
COLS = ["Open", "High", "Low", "Close", "Volume"]


def to_rows(data, interval):
    """Turn a yfinance group_by='ticker' frame into (symbol, interval, ts, o, h, l, c, v) tuples."""
    rows = []
    for sym in data.columns.get_level_values(0).unique():
        df = data[sym].dropna(subset=["Close"])
        if df.index.tz is None:  # daily and monthly bars carry dates only
            ts = df.index.strftime("%Y-%m-%d")
        else:
            ts = df.index.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        values = df[COLS].astype(float).itertuples(index=False, name=None)
        rows += [(sym, interval, t, *v) for t, v in zip(ts, values)]
    return rows


def main():
    DB.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute(
        "CREATE TABLE IF NOT EXISTS prices(symbol TEXT, interval TEXT, ts TEXT, open REAL, high REAL,"
        " low REAL, close REAL, volume REAL, PRIMARY KEY(symbol, interval, ts)) WITHOUT ROWID"
    )
    written = 0
    for interval, (full, recent) in PERIODS.items():
        has_rows = con.execute("SELECT 1 FROM prices WHERE interval=? LIMIT 1", (interval,)).fetchone()
        period = recent if has_rows else full
        if not period:
            continue
        data = yf.download(list(SYMBOLS), period=period, interval=interval, group_by="ticker",
                           auto_adjust=False, progress=False, threads=False)
        rows = to_rows(data, interval)
        con.executemany("INSERT OR REPLACE INTO prices VALUES (?,?,?,?,?,?,?,?)", rows)
        con.commit()
        written += len(rows)
        print(f"{interval}: {len(rows)} rows ({period})")

    latest = dict(con.execute("SELECT symbol, max(ts) FROM prices WHERE interval='1d' GROUP BY symbol"))
    cutoff = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    stale = [f"{s} (last {latest.get(s, 'never')})" for s in SYMBOLS if latest.get(s, "") < cutoff]
    if stale:
        print("WARNING no daily bar in 7 days:", ", ".join(stale), file=sys.stderr)
    if not written:
        sys.exit("ERROR nothing downloaded: Yahoo blocked the request or the network is down")


def _selftest():
    idx = pd.to_datetime(["2026-01-05 09:00", "2026-01-05 09:15"]).tz_localize("America/New_York")
    cols = pd.MultiIndex.from_product([["AAA", "BBB"], COLS])
    data = pd.DataFrame(1.0, index=idx, columns=cols)
    data.loc[idx[1], ("BBB", "Close")] = float("nan")  # a bar with no close must be dropped
    rows = to_rows(data, "15m")
    assert len(rows) == 3, rows
    assert rows[0] == ("AAA", "15m", "2026-01-05T14:00:00Z", 1.0, 1.0, 1.0, 1.0, 1.0), rows[0]
    daily = to_rows(data.tz_localize(None), "1d")
    assert daily[0][2] == "2026-01-05", daily[0]
    print("selftest ok")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
