#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
./.venv/bin/python main.py --step check_env
./.venv/bin/python main.py --step validate_scenarios
./.venv/bin/python main.py --step readiness
./.venv/bin/python main.py --step preflight
