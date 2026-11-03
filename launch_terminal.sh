#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="fomo_terminal"
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v tmux >/dev/null 2>&1; then
    printf 'tmux is required to launch the terminal.\n' >&2
    exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    exec tmux attach-session -t "$SESSION_NAME"
fi

tmux new-session -d -s "$SESSION_NAME" -n "terminal" -c "$PROJECT_ROOT"
tmux send-keys -t "$SESSION_NAME:0.0" \
    "FOMO_EXTERNAL_CHART_DASHBOARD=true python -m src.main" C-m
tmux split-window -h -t "$SESSION_NAME:0.0" -c "$PROJECT_ROOT"
tmux send-keys -t "$SESSION_NAME:0.1" \
    "FOMO_STATE_API_URL=http://127.0.0.1:8765 streamlit run src/reporting/chart_dashboard.py --server.port 8050 --server.headless true" C-m

exec tmux attach-session -t "$SESSION_NAME"
