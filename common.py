"""Helpers shared by the weather, rhine, gas_storage, hydro and outages collectors."""
import json
import sqlite3
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).parent
# a browser-style agent: the Swiss energy office rejects anything else
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"


def fetch(url, headers=None, tries=5):
    """GET url and return the body. On 'too many requests' waits as long as the server asks, then retries."""
    for attempt in range(tries):
        try:
            with urlopen(Request(url, headers={"User-Agent": UA, **(headers or {})}), timeout=300) as resp:
                return resp.read()
        except HTTPError as e:
            if e.code != 429 or attempt == tries - 1:
                raise
            time.sleep(int(e.headers.get("Retry-After") or 60) + 1)


def fetch_json(url, headers=None):
    return json.loads(fetch(url, headers))


def entsoe(**params):
    """GET from the ENTSO-E Transparency Platform. Needs ENTSOE_API_KEY in .env."""
    env = dict(line.split("=", 1) for line in (ROOT / ".env").read_text().split())
    return fetch("https://web-api.tp.entsoe.eu/api?" + urlencode({"securityToken": env["ENTSOE_API_KEY"], **params}))


def open_db(name, schema):
    (ROOT / "data").mkdir(exist_ok=True)
    con = sqlite3.connect(ROOT / "data" / name, timeout=60)
    con.execute(schema)
    return con
