# SolarPot

Forecast of the German day-ahead power price (zone Germany and Luxembourg) for every quarter-hour of the next day,
made each morning before the 12:00 auction, with a local dashboard.

## Move it to a new server

The server needs Linux, cron, its clock on German time, and Python 3.12 or [uv](https://docs.astral.sh/uv/).

1. Copy the folder, with its data (about 810 MB) and the `.env` file that holds the two API keys:

       tar --exclude=.venv --exclude=__pycache__ -cf - SolarPot | ssh user@server tar -xf -

   Run it from the folder that holds `SolarPot`; the copy lands in the home folder on the server.

2. On the server:

       cd SolarPot && ./setup.sh

   It installs the packages, runs the checks, fetches fresh data, schedules every job and starts the dashboard.
   `./setup.sh --no-cron` only installs and checks, to try the code without scheduling anything.

3. Stop the schedule on the old machine (`crontab -e`), or keep both running: they do not disturb each other.

Copy the data on the day you set up. The collectors fill a gap of a few days by themselves, not a longer one.

## Dashboard

`http://127.0.0.1:8050` on the server. It has no login, so it answers only on the server itself.
From your own computer, open a tunnel first and then the same address:

    ssh -L 8050:127.0.0.1:8050 user@server

It shows the price now, the forecast for each quarter-hour, the forecast's error against a simple benchmark,
and the state of every data feed. The page refreshes itself once a minute.

## What runs when (German time)

| Job | When | What |
|---|---|---|
| collectors (`energy_charts.py`, `smard.py`, `weather.py`, ...) | every 15 minutes to weekly | fetch prices, demand, weather and the other inputs into `data/*.db` |
| `forecast.py` | 11:10 and 11:40 | forecasts tomorrow into `data/forecast.db`; about 5 minutes on 10 cores |
| `dashboard.py` | always | serves the dashboard |

Every job writes a log to `data/<name>.log`. Each script explains itself in its first lines and checks itself
with `--selftest`.

`model.py` scores the forecast on the last 12 months (about 80 minutes on 16 cores). The daily job does not need
it to be run again: its result, `data/backtest.db`, comes with the data.
