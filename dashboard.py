"""Local website: the price now, the forecast for every quarter-hour, and how the forecast is doing.

Serves dashboard.html and the figures behind it, read from the collector databases and data/forecast.db.
It only reads: every database is opened read-only, so it cannot hold up a collector or the daily job.
The page asks for fresh figures once a minute.
    .venv/bin/python dashboard.py              # http://127.0.0.1:8050, reachable from this machine only
    .venv/bin/python dashboard.py 9000         # another port
    .venv/bin/python dashboard.py --selftest
From another machine: ssh -L 8050:127.0.0.1:8050 user@server, then open http://127.0.0.1:8050 there.
"""
import json
import sqlite3
import sys
import tempfile
import threading
import time
from contextlib import closing
from datetime import datetime, timedelta, timezone
from datetime import time as clock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from zoneinfo import ZoneInfo

from common import ROOT

DATA = ROOT / "data"
TZ = ZoneInfo("Europe/Berlin")
STAMP = "%Y-%m-%dT%H:%M:%SZ"
QUARTER = timedelta(minutes=15)
HISTORY_DAYS = 14  # days before today on the page
KEEP_SECONDS = 15  # one set of figures serves every request in this time
PRICE = "name='Day Ahead Auction (DE-LU)' AND res='15'"
GRID = ["load", "solar", "wind_onshore", "wind_offshore"]  # series in smard.db, MWh per quarter-hour
# collector: seconds between two of its runs (the schedule is in setup.sh)
FEEDS = {"energy_charts": 900, "smard": 900, "commodities": 900, "weather": 3600, "rhine": 3600, "outages": 3600,
         "gas_storage": 12 * 3600, "neighbour_prices": 14 * 3600, "hydro": 86400, "worldbank": 86400,
         "forecast": 86400, "installed_capacity": 7 * 86400}


def rows(data, db, query, args=()):
    """Run a query on one database. A database or table that is not there yet gives no rows."""
    try:
        with closing(sqlite3.connect(f"file:{data / db}?mode=ro", uri=True, timeout=30)) as con:
            return con.execute(query, args).fetchall()
    except sqlite3.OperationalError:
        return []


def parse(stamp):
    return datetime.strptime(stamp, STAMP).replace(tzinfo=timezone.utc)


def midnight(day):
    """00:00 German time on a date, in UTC."""
    return datetime.combine(day, clock(), TZ).astimezone(timezone.utc)


def rounded(value):
    return None if value is None or value != value else round(value, 2)


def benchmark(stamp, actual):
    """model.naive for a quarter-hour: the same one yesterday from Tuesday to Friday, else a week ago."""
    t = parse(stamp)
    days_back = 1 if 1 <= t.astimezone(TZ).weekday() <= 4 else 7
    return actual.get((t - timedelta(days=days_back)).strftime(STAMP))


def summary(scored):
    """scored: (error, benchmark error or None, inside the likely range) per quarter-hour with a published price."""
    if not scored:
        return None
    bench = [b for _, b, _ in scored if b is not None]
    return {"n": len(scored), "mae": rounded(sum(e for e, _, _ in scored) / len(scored)),
            "benchmark": rounded(sum(bench) / len(bench)) if bench else None,
            "inside": rounded(sum(i for _, _, i in scored) / len(scored))}


def snapshot(now, data=DATA):
    """Everything the page shows, as of `now` (UTC)."""
    today = now.astimezone(TZ).date()
    start = midnight(today - timedelta(days=HISTORY_DAYS))
    hour = {r[0]: r[1:] for r in rows(data, "forecast.db", "SELECT ts, lgbm_p10_adj, lgbm_p90_adj, p_negative FROM forecast")}
    # the likely range is made for the hour; each quarter-hour gets it moved by its own gap to the hour
    forecast = {ts: (price, *[edge + gap if edge is not None else None for edge in hour.get(hour_ts, (None, None))[:2]],
                     hour.get(hour_ts, (None,) * 3)[2], issued)
                for ts, hour_ts, gap, price, issued in
                rows(data, "forecast.db", "SELECT ts, hour_ts, gap, price, issued_at FROM forecast_15min")}
    first = min([start.strftime(STAMP), *forecast])
    actual = dict(rows(data, "energy_charts.db", f"SELECT ts, value FROM series WHERE {PRICE} AND ts >= ?",
                       ((parse(first) - timedelta(days=7)).strftime(STAMP),)))  # a week more: the benchmark looks back

    scored = {}  # every stored forecast whose price is out, also those older than the page shows
    for ts, (price, low, high, _, _) in forecast.items():
        if ts in actual and price is not None:
            bench = benchmark(ts, actual)
            scored[ts] = (abs(price - actual[ts]), None if bench is None else abs(bench - actual[ts]),
                          low is not None and low <= actual[ts] <= high)

    last = max([now.strftime(STAMP), *actual, *forecast])
    end = midnight(parse(last).astimezone(TZ).date() + timedelta(days=1))
    grid = [(start + i * QUARTER).strftime(STAMP) for i in range(int((end - start) / QUARTER))]
    index = lambda t: int((t - start) / QUARTER)
    days = []
    for n in range((end.astimezone(TZ).date() - start.astimezone(TZ).date()).days):
        day = start.astimezone(TZ).date() + timedelta(days=n)
        i0, i1 = index(midnight(day)), index(midnight(day + timedelta(days=1)))
        days.append({"day": day.isoformat(), "i0": i0, "i1": i1,
                     "score": summary([scored[ts] for ts in grid[i0:i1] if ts in scored])})
    forecast_days = [d for d in days if any(ts in forecast for ts in grid[d["i0"]:d["i1"]])]
    week_ago = (now - timedelta(days=7)).strftime(STAMP)

    smard = {}
    for series, ts, value in rows(data, "smard.db", "SELECT series, ts, value FROM smard WHERE ts >= ?", (grid[0],)):
        smard.setdefault(series, {})[ts] = value * 4 / 1000  # MWh per quarter-hour -> GW
    gw = lambda name: [rounded(smard.get(name, {}).get(ts)) for ts in grid]
    wind = [None if a is None or b is None else round(a + b, 2)
            for a, b in zip(gw("wind_onshore_forecast"), gw("wind_offshore_forecast"))]
    together = [ts for ts in sorted(smard.get("load", {}), reverse=True) if all(ts in smard.get(s, {}) for s in GRID)]
    measured = {"ts": together[0], **{s: rounded(smard[s][together[0]]) for s in GRID}} if together else None

    def latest(name):
        found = rows(data, "energy_charts.db", "SELECT ts, value FROM series WHERE name=? AND res='60'"
                                               " ORDER BY ts DESC LIMIT 1", (name,))
        return {"ts": found[0][0], "value": rounded(found[0][1])} if found else None

    this_quarter = grid[min(index(now), len(grid) - 1)]
    ages = {name: max((p.stat().st_mtime for p in (data / f"{name}.db", data / f"{name}.log") if p.exists()), default=None)
            for name in FEEDS}
    column = lambda k: [rounded(forecast[ts][k]) if ts in forecast else None for ts in grid]
    return {
        "now": now.timestamp(), "start": start.timestamp(), "n": len(grid), "days": days,
        "actual": [rounded(actual.get(ts)) for ts in grid],
        "forecast": column(0), "low": column(1), "high": column(2), "p_negative": column(3),
        "demand": gw("load_forecast"), "wind": wind, "solar": gw("solar_forecast"),
        "forecast_day": forecast_days[-1] | {"issued_at": max(forecast[ts][4] for ts in grid if ts in forecast)}
        if forecast_days else None,
        "score": {"live": summary(list(scored.values())), "since": min(scored, default=None),
                  "week": summary([s for ts, s in scored.items() if ts >= week_ago])},
        "backtest": dict(zip(("mae", "benchmark", "hours", "from", "to"), rows(data, "backtest.db",
                         "SELECT round(avg(abs(lgbm-price)),2), round(avg(abs(naive-price)),2), count(*), min(ts), max(ts)"
                         " FROM backtest")[0])) if rows(data, "backtest.db", "SELECT 1 FROM backtest LIMIT 1") else None,
        "next_run": next(t for t in (midnight(today + timedelta(days=d)).astimezone(TZ).replace(hour=11, minute=10)
                                     for d in (0, 1)) if t > now).timestamp(),
        "market": {"gas": latest("Gas (NCG, THE)"), "co2": latest("CO2 Emission Allowances, Auction DE"),
                   "grid": measured},
        "neighbours": [{"zone": zone, "price": rounded(price)} for zone, price in
                       rows(data, "neighbour_prices.db", "SELECT zone, price FROM prices WHERE ts=?", (this_quarter,))],
        "feeds": [{"name": name, "every": every, "age": None if ages[name] is None else round(now.timestamp() - ages[name])}
                  for name, every in FEEDS.items()],
    }


_kept = (0.0, b"")
_lock = threading.Lock()


def figures():
    """The snapshot as JSON; the page of every open browser tab shares one."""
    global _kept
    with _lock:
        if time.time() - _kept[0] > KEEP_SECONDS:
            _kept = (time.time(), json.dumps(snapshot(datetime.now(timezone.utc))).encode())
        return _kept[1]


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.reply((ROOT / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api":
            self.reply(figures(), "application/json")
        else:
            self.send_error(404)

    def reply(self, body, kind):
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # every open page asks once a minute: the log would only fill up
        pass


def main(port=8050):
    # 127.0.0.1 only: the page has no login, so it must not be reachable from the network
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"dashboard on http://127.0.0.1:{port}", flush=True)
    server.serve_forever()


def _selftest():
    assert midnight(parse("2026-03-29T12:00:00Z").date()) == parse("2026-03-28T23:00:00Z")
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp)
        now = parse("2026-11-18T10:20:00Z")  # Wednesday 11:20 German time
        assert snapshot(now, data)["forecast_day"] is None  # a new server, before the first collector run
        day = parse("2026-11-17T23:00:00Z")  # 00:00 on the 18th, German time
        with closing(sqlite3.connect(data / "energy_charts.db")) as con:
            con.execute("CREATE TABLE series(name TEXT, res TEXT, ts TEXT, value REAL)")
            for back, price in ((0, 100.0), (1, 90.0), (7, 50.0)):  # the 18th, the day before, a week before
                con.executemany("INSERT INTO series VALUES ('Day Ahead Auction (DE-LU)', '15', ?, ?)",
                                [((day - timedelta(days=back) + i * QUARTER).strftime(STAMP), price) for i in range(96)])
            con.commit()
        with closing(sqlite3.connect(data / "forecast.db")) as con:
            con.execute("CREATE TABLE forecast(ts TEXT, lgbm_p10_adj REAL, lgbm_p90_adj REAL, p_negative REAL)")
            con.execute("CREATE TABLE forecast_15min(ts TEXT, issued_at TEXT, hour_ts TEXT, gap REAL, price REAL)")
            for h in range(48):  # the 18th: price out; the 19th: not yet
                hour_ts = (day + timedelta(hours=h)).strftime(STAMP)
                con.execute("INSERT INTO forecast VALUES (?, 97, 115, 0.25)", (hour_ts,))
                con.executemany("INSERT INTO forecast_15min VALUES (?, 'made', ?, ?, ?)",
                                [((day + timedelta(hours=h) + q * QUARTER).strftime(STAMP), hour_ts, gap, 110 + gap)
                                 for q, gap in enumerate((-4.0, 0.0, 0.0, 4.0))])
            con.commit()
        s = snapshot(now, data)
        i = int((s["now"] - s["start"]) // 900)
        assert s["n"] == 16 * 96 and (s["actual"][i], s["forecast"][i]) == (100, 110), (s["n"], s["actual"][i])
        assert (s["low"][i - 1], s["high"][i - 1], s["p_negative"][i]) == (93, 111, 0.25)  # 11:00: moved by its gap of -4
        today = s["days"][14]
        assert (today["day"], today["i1"] - today["i0"]) == ("2026-11-18", 96), today
        # errors 6, 10, 10, 14; Wednesday's benchmark is the day before: 90 against 100; the last quarter's range starts at 101
        assert today["score"] == {"n": 96, "mae": 10, "benchmark": 10, "inside": 0.75}, today
        assert s["score"]["live"] == today["score"] and s["score"]["since"] == day.strftime(STAMP)
        assert s["forecast_day"]["day"] == "2026-11-19" and s["forecast_day"]["score"] is None, s["forecast_day"]
        assert datetime.fromtimestamp(s["next_run"], TZ).strftime("%d %H:%M") == "19 11:10"  # today's 11:10 has passed
        assert benchmark("2026-11-16T12:00:00Z", {"2026-11-09T12:00:00Z": 1.0}) == 1  # a Monday looks a week back
        json.dumps(s, allow_nan=False)
    print("selftest ok")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    _selftest() if "--selftest" in sys.argv else main(*map(int, args))
