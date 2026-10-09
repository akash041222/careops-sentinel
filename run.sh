#!/usr/bin/env bash
# One command: installs deps, builds the web app if needed, starts API + website on http://localhost:8000
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install -q -r requirements.txt
if [ -f frontend/.needs-ui-build ]; then
  command -v npm >/dev/null 2>&1 || { echo "Node.js and npm are required to build the updated alarm UI."; exit 1; }
  [ -d frontend/node_modules ] || (cd frontend && npm install)
  (cd frontend && npm run build)
  rm frontend/.needs-ui-build
fi
if [ ! -f frontend/dist/index.html ]; then (cd frontend && npm install && npm run build); fi
echo "Open http://localhost:8000   (login: EMP-1001 / CareOps@123)"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
