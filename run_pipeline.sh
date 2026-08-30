#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "==> Creating Python 3.11 virtual environment"
if [ ! -d .venv ]; then
  python3.11 -m venv .venv
fi

source .venv/bin/activate

echo "==> Installing dependencies"
python -m pip install --upgrade pip
python -m pip install -r requirements/train.txt

echo "==> Running tests"
python -m pytest tests/test_model.py -q

echo "==> Training model"
python src/train.py

echo "==> Starting serving API on port 8000"
python -m uvicorn src.serve:app --host 0.0.0.0 --port 8000
