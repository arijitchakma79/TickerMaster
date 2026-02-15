#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:8001}"

echo "== TickerMaster pre-deploy checks =="

echo "1) Validate backend environment"
(
  cd "$BACKEND_DIR"
  .venv/bin/python scripts/validate_env.py
)

echo "2) Build frontend"
(
  cd "$FRONTEND_DIR"
  npm run build
)

echo "3) Run backend hardening tests"
(
  cd "$BACKEND_DIR"
  .venv/bin/python -m unittest discover -s tests -v
)

echo "4) Backend health"
curl -fsS "$BACKEND_URL/api/health" >/dev/null

echo "5) Unauthorized tracker access should be blocked"
status="$(curl -sS -o /dev/null -w "%{http_code}" "$BACKEND_URL/api/tracker/agents")"
if [[ "$status" != "401" ]]; then
  echo "Expected 401 from /api/tracker/agents, got $status"
  exit 1
fi

echo "All pre-deploy checks passed."
