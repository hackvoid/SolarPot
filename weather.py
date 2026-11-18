"""Pull hourly weather for 10 German points from Open-Meteo into data/weather.db.

Three kinds of row, all upserted:
  actual       what the weather was (reanalysis), from 2018
  forecast_d1  what was forecast 24 hours before each hour, from 2024
  forecast_d2  what was forecast 48 hours before each hour, from 2024 - build_dataset.py picks
               between d1 and d2 so that training only sees forecasts issued before the auction
  forecast     the newest forecast for yesterday through the next 3 days
    .venv/bin/python weather.py             # run hourly; the first run backfills (about 8 min)
    .venv/bin/python weather.py --selftest
"""
import sys
import time
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlencode

from common import fetch_json, open_db

POINTS = {  # name: (latitude, longitude)
    "north_sea": (54.3, 7.0), "baltic_sea": (54.6, 13.2),            # offshore wind
    "schleswig_holstein": (54.4, 9.2), "lower_saxony": (52.9, 8.2),  # onshore wind
    "saxony_anhalt": (51.9, 11.6), "brandenburg": (52.4, 13.4),      # wind, solar, Berlin demand
    "nrw": (51.2, 6.9), "hesse": (50.1, 8.7),                        # demand
    "bavaria": (48.6, 11.8), "baden_wuerttemberg": (48.7, 9.2),      # solar, demand
}
VARS = ["temperature_2m", "wind_speed_100m", "shortwave_radiation", "cloud_cover"]  # C, m/s, W/m2, %
LIVE = "https://api.open-meteo.com/v1/forecast"
ARCHIVES = {  # kind: (address, suffix Open-Meteo puts on each variable name, first date, days behind today)
    # the archive rejects today's date for the first hours of each UTC day, so stop at yesterday
    "actual": ("https://archive-api.open-meteo.com/v1/archive", "", date(2018, 1, 1), 1),
    "forecast_d1": ("https://previous-runs-api.open-meteo.com/v1/forecast", "_previous_day1", date(2024, 1, 1), 0),
    "forecast_d2": ("https://previous-runs-api.open-meteo.com/v1/forecast", "_previous_day2", date(2024, 1, 1), 0),
}


def to_rows(kind, docs, suffix):
    """Turn Open-Meteo's one-document-per-point reply into (kind, point, ts, *VARS) tuples; ts is UTC."""
    rows = []
    for point, doc in zip(POINTS, docs, strict=True):
        hourly = doc["hourly"]
        for t, *values in zip(hourly["time"], *(hourly[v + suffix] for v in VARS), strict=True):
            if any(v is not None for v in values):  # hours the archive does not cover come back empty
                rows.append((kind, point, t + ":00Z", *values))
    return rows


def main():
    cols = ", ".join(f"{v} REAL" for v in VARS)
    con = open_db("weather.db", f"CREATE TABLE IF NOT EXISTS weather(kind TEXT, point TEXT, ts TEXT, {cols},"
                                " PRIMARY KEY(kind, point, ts)) WITHOUT ROWID")
    today = datetime.now(timezone.utc).date()  # Open-Meteo counts days in UTC
    jobs = [("forecast", LIVE, "", {"past_days": 1, "forecast_days": 3})]
    for kind, (base, suffix, first, lag) in ARCHIVES.items():
        last = con.execute("SELECT max(ts) FROM weather WHERE kind=?", (kind,)).fetchone()[0]
        start = date.fromisoformat(last[:10]) - timedelta(days=7) if last else first  # recent days get revised
        final = today - timedelta(days=lag)
        while start <= final:  # one request per calendar year
            end = min(date(start.year, 12, 31), final)
            jobs.append((kind, base, suffix, {"start_date": start, "end_date": end}))
            start = end + timedelta(days=1)

    for i, (kind, base, suffix, params) in enumerate(jobs):
        if i and len(jobs) > 3:
            time.sleep(30)  # backfill: stay under the free tier's per-minute limit
        query = urlencode({
            "latitude": ",".join(str(lat) for lat, _ in POINTS.values()),
            "longitude": ",".join(str(lon) for _, lon in POINTS.values()),
            "hourly": ",".join(v + suffix for v in VARS), "wind_speed_unit": "ms", **params})
        rows = to_rows(kind, fetch_json(f"{base}?{query}"), suffix)
        con.executemany(f"INSERT OR REPLACE INTO weather VALUES (?,?,?,{','.join('?' * len(VARS))})", rows)
        con.commit()
        print(f"{kind} {params.get('start_date', 'live')}: {len(rows)} rows")


def _selftest():
    hourly = {"time": ["2024-06-01T00:00", "2024-06-01T01:00"]}
    for n, v in enumerate(VARS):
        hourly[v + "_previous_day1"] = [float(n), None]
    rows = to_rows("forecast_d1", [{"hourly": hourly}] * len(POINTS), "_previous_day1")
    assert len(rows) == len(POINTS), len(rows)  # the hour with no values is dropped
    assert rows[0] == ("forecast_d1", "north_sea", "2024-06-01T00:00:00Z", 0.0, 1.0, 2.0, 3.0), rows[0]
    print("selftest ok")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
