#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is required. Install Python 3.10 or newer, then run this script again."
  exit 1
fi

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r server/requirements.txt

URL="http://127.0.0.1:8787/"
echo
echo "DoubleDeep AI is starting at $URL"
echo "Keep this terminal open while using the website."
echo

if command -v xdg-open >/dev/null 2>&1; then
  (sleep 2 && xdg-open "$URL" >/dev/null 2>&1) &
elif command -v open >/dev/null 2>&1; then
  (sleep 2 && open "$URL" >/dev/null 2>&1) &
fi

python server/app.py
