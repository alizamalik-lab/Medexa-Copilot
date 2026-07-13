#!/usr/bin/env bash
set -euo pipefail

RAG_DIR="${RAG_DIR:-/app/chatbot/chatbot}"
export RAG_URL="${RAG_URL:-http://127.0.0.1:8000}"
export RAG_CHAT_PATH="${RAG_CHAT_PATH:-/chat}"
export PORT="${PORT:-10000}"

cleanup() {
  if [ -n "${RAG_PID:-}" ] && kill -0 "$RAG_PID" 2>/dev/null; then
    kill "$RAG_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[start] GROQ_API_KEY set: $([ -n "${GROQ_API_KEY:-}" ] && echo yes || echo NO)"
echo "[start] Starting RAG on :8000 from ${RAG_DIR}"
cd "${RAG_DIR}"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 &
RAG_PID=$!

echo "[start] Starting Express on :${PORT} (RAG_URL=${RAG_URL})"
cd /app
node server/index.js &
NODE_PID=$!

echo "[start] Waiting for RAG /health (embeddings load on first request)..."
for i in $(seq 1 300); do
  if curl -fsS "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
    echo "[start] RAG is ready"
    break
  fi
  if ! kill -0 "$RAG_PID" 2>/dev/null; then
    echo "[start] RAG process exited early — check GROQ_API_KEY and logs" >&2
    wait "$RAG_PID" || true
    kill "$NODE_PID" 2>/dev/null || true
    exit 1
  fi
  if ! kill -0 "$NODE_PID" 2>/dev/null; then
    echo "[start] Express process exited early" >&2
    wait "$NODE_PID" || true
    exit 1
  fi
  if [ "$i" -eq 300 ]; then
    echo "[start] Timed out waiting for RAG; Express stays up but chat may fail" >&2
  fi
  sleep 2
done

wait "$NODE_PID"
EXIT_CODE=$?
cleanup
exit "$EXIT_CODE"
