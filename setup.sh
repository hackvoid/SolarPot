#!/usr/bin/env bash
# Set up SolarPot in this folder: Python packages, checks, fresh data, the schedule and the dashboard.
#   ./setup.sh              everything
#   ./setup.sh --no-cron    packages and checks only: nothing is scheduled and nothing keeps running
#   PORT=9000 ./setup.sh    the dashboard on another port (default 8050)
# Safe to run again: it replaces its own schedule lines and restarts the dashboard.
set -euo pipefail
cd "$(dirname "$0")"
DIR=$(pwd)
PORT=${PORT:-8050}
PY="$DIR/.venv/bin/python"
say() { printf '\n== %s\n' "$*"; }
stop() { printf '\nSTOPPED: %s\n' "$*" >&2; exit 1; }

[ -f data/energy_charts.db ] || stop "data/energy_charts.db is missing. Copy the data folder from the old machine first (see README.md)."
if [ "${1:-}" != "--no-cron" ]; then
    command -v crontab >/dev/null || stop "cron is not installed. On Debian or Ubuntu: sudo apt install cron"
    # the schedule below is written in German time
    [ "$(date -d 2026-01-15 +%z)$(date -d 2026-07-15 +%z)" = "+0100+0200" ] ||
        stop "This machine's clock is not on German time. Run: sudo timedatectl set-timezone Europe/Berlin   then ./setup.sh again."
fi

say "Python packages"
if ! "$PY" -c 'import sys' 2>/dev/null; then
    rm -rf .venv  # a .venv copied from another machine does not run here
    if command -v uv >/dev/null; then
        uv venv --python 3.12 .venv
    elif python3 -c 'import sys; sys.exit(sys.version_info < (3, 12))' 2>/dev/null; then
        python3 -m venv .venv || stop "python3 cannot make a virtual environment. On Debian or Ubuntu: sudo apt install python3-venv"
    else
        stop "Python 3.12 or newer is needed. uv brings its own: curl -LsSf https://astral.sh/uv/install.sh | sh   then open a new shell and run ./setup.sh again."
    fi
fi
if command -v uv >/dev/null; then
    uv pip install --python "$PY" -r requirements.txt
else
    "$PY" -m pip install -r requirements.txt
fi

say "Checks"
"$PY" -c 'import lightgbm' || stop "LightGBM does not load. On Debian or Ubuntu: sudo apt install libgomp1"
for script in model forecast dashboard; do
    printf '%s: ' "$script"
    "$PY" "$script.py" --selftest
done
if [ -f .env ]; then
    chmod 600 .env
else
    echo "NOTE: no .env file, so gas storage, reservoirs and French outages will not update."
    echo "      It holds two lines: AGSI_API_KEY=... and ENTSOE_API_KEY=..."
fi

if [ "${1:-}" = "--no-cron" ]; then
    say "Done. Nothing is scheduled."
    echo "Forecast a past day:  .venv/bin/python forecast.py 2026-11-18"
    echo "Open the dashboard:   .venv/bin/python dashboard.py   then http://127.0.0.1:8050"
    exit 0
fi

say "Fresh data (each collector once; the copied data may be hours old)"
for collector in energy_charts smard commodities weather rhine neighbour_prices gas_storage hydro outages worldbank installed_capacity; do
    printf '%s: ' "$collector"
    if "$PY" "$collector.py" >> "data/$collector.log" 2>&1; then echo ok; else echo "FAILED, see data/$collector.log"; fi
done

say "Schedule"
RUN="cd $DIR && .venv/bin/python"
OTHERS=$(crontab -l 2>/dev/null | grep -vF "$RUN" || true)  # everything that is not ours stays
{
    if [ -n "$OTHERS" ]; then printf '%s\n' "$OTHERS"; fi
    cat <<EOF
*/15 * * * * $RUN energy_charts.py >> data/energy_charts.log 2>&1
3-59/15 * * * * $RUN commodities.py >> data/commodities.log 2>&1
6-59/15 * * * * $RUN smard.py >> data/smard.log 2>&1
7 * * * * $RUN weather.py >> data/weather.log 2>&1
11 * * * * $RUN rhine.py >> data/rhine.log 2>&1
35 * * * * $RUN outages.py >> data/outages.log 2>&1
40 8,20 * * * $RUN gas_storage.py >> data/gas_storage.log 2>&1
30 7 * * * $RUN hydro.py >> data/hydro.log 2>&1
20 6 * * * $RUN worldbank.py >> data/worldbank.log 2>&1
25 13,14,17,23 * * * $RUN neighbour_prices.py >> data/neighbour_prices.log 2>&1
50 5 * * 1 $RUN installed_capacity.py >> data/installed_capacity.log 2>&1
10,40 11 * * * $RUN forecast.py >> data/forecast.log 2>&1
@reboot $RUN $DIR/dashboard.py $PORT >> data/dashboard.log 2>&1
EOF
} | crontab -
echo "$(crontab -l | grep -cF "$RUN") lines in the crontab; the forecast runs daily at 11:10 and 11:40"

say "Dashboard"
pkill -f "$DIR/dashboard.py" || true
nohup "$PY" "$DIR/dashboard.py" "$PORT" >> data/dashboard.log 2>&1 &
for _ in 1 2 3 4 5 6 7 8 9 10; do
    sleep 1
    if "$PY" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/api', timeout=60)" 2>/dev/null; then
        echo "Running on http://127.0.0.1:$PORT (this machine only; it starts again after a reboot)"
        echo "From your own computer:  ssh -L $PORT:127.0.0.1:$PORT $(whoami)@$(hostname)   then open http://127.0.0.1:$PORT"
        exit 0
    fi
done
stop "The dashboard did not start. See data/dashboard.log"
