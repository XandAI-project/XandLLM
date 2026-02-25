#!/bin/sh
# XandLLM — Docker integration tests
# Runs after the API server is healthy.  Exit 0 = all pass, Exit 1 = failures.

set -e

API="${API_URL:-http://xandllm:11435}"
MODEL="${TEST_MODEL_ID:-bartowski/Meta-Llama-3.1-8B-Instruct-GGUF}"
PASS=0
FAIL=0
TOTAL=0

# ── helpers ────────────────────────────────────────────────────────────────────

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

pass() { printf "${GREEN}✓ PASS${NC}: %s\n" "$1"; PASS=$((PASS+1)); TOTAL=$((TOTAL+1)); }
fail() { printf "${RED}✗ FAIL${NC}: %s\n  expected: %s\n  got:      %s\n" "$1" "$2" "$3"; FAIL=$((FAIL+1)); TOTAL=$((TOTAL+1)); }

contains() { echo "$1" | grep -qF "$2"; }

# ── GET /health ────────────────────────────────────────────────────────────────

echo "=== XandLLM Integration Tests (${API}) ==="
echo ""

result=$(curl -sf "${API}/health" 2>&1) || result="CONNECTION_FAILED"
if contains "$result" '"status"' && contains "$result" '"ok"'; then
    pass "GET /health → {\"status\":\"ok\"}"
else
    fail "GET /health" '{"status":"ok"}' "$result"
fi

# ── GET /v1/models ─────────────────────────────────────────────────────────────

result=$(curl -sf "${API}/v1/models" 2>&1) || result="CONNECTION_FAILED"
if contains "$result" '"object":"list"' || contains "$result" '"object": "list"'; then
    pass "GET /v1/models → object:list"
else
    fail "GET /v1/models" '"object":"list"' "$result"
fi

if contains "$result" '"data"'; then
    pass "GET /v1/models → data array present"
else
    fail "GET /v1/models data array" '"data":[...]' "$result"
fi

# ── POST /v1/chat/completions (non-streaming) ──────────────────────────────────

CHAT_BODY=$(cat <<'JSON'
{
  "model": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
  "messages": [{"role": "user", "content": "Say the single word HELLO and nothing else."}],
  "stream": false,
  "max_tokens": 5,
  "temperature": 0.0
}
JSON
)

result=$(curl -sf -X POST "${API}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$CHAT_BODY" 2>&1) || result="CONNECTION_FAILED"

if contains "$result" '"object":"chat.completion"' || contains "$result" '"object": "chat.completion"'; then
    pass "POST /v1/chat/completions → object:chat.completion"
else
    fail "POST /v1/chat/completions" '"object":"chat.completion"' "$result"
fi

if contains "$result" '"choices"'; then
    pass "POST /v1/chat/completions → choices present"
else
    fail "POST /v1/chat/completions choices" '"choices":[...]' "$result"
fi

if contains "$result" '"role":"assistant"' || contains "$result" '"role": "assistant"'; then
    pass "POST /v1/chat/completions → assistant role in message"
else
    fail "POST /v1/chat/completions assistant role" '"role":"assistant"' "$result"
fi

if contains "$result" '"usage"'; then
    pass "POST /v1/chat/completions → usage block present"
else
    fail "POST /v1/chat/completions usage" '"usage":{...}' "$result"
fi

# ── POST /v1/completions (non-streaming) ──────────────────────────────────────

COMP_BODY=$(cat <<'JSON'
{
  "model": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
  "prompt": "1 + 1 =",
  "stream": false,
  "max_tokens": 3
}
JSON
)

result=$(curl -sf -X POST "${API}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "$COMP_BODY" 2>&1) || result="CONNECTION_FAILED"

if contains "$result" '"object":"text_completion"' || contains "$result" '"object": "text_completion"'; then
    pass "POST /v1/completions → object:text_completion"
else
    fail "POST /v1/completions" '"object":"text_completion"' "$result"
fi

if contains "$result" '"choices"'; then
    pass "POST /v1/completions → choices present"
else
    fail "POST /v1/completions choices" '"choices":[...]' "$result"
fi

# ── POST /v1/chat/completions (streaming SSE) ─────────────────────────────────

SSE_BODY=$(cat <<'JSON'
{
  "model": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
  "messages": [{"role": "user", "content": "Say hi."}],
  "stream": true,
  "max_tokens": 3
}
JSON
)

result=$(curl -sf -X POST "${API}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Accept: text/event-stream" \
    -d "$SSE_BODY" 2>&1) || result="CONNECTION_FAILED"

if contains "$result" "data:"; then
    pass "POST /v1/chat/completions stream=true → SSE data lines received"
else
    fail "POST /v1/chat/completions streaming" "data: {...}" "$result"
fi

if contains "$result" "[DONE]"; then
    pass "POST /v1/chat/completions stream=true → [DONE] sentinel received"
else
    fail "POST /v1/chat/completions streaming [DONE]" "data: [DONE]" "$result"
fi

# ── Malformed request → 4xx ───────────────────────────────────────────────────

http_status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${API}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{invalid json}' 2>&1)

if [ "$http_status" -ge 400 ] && [ "$http_status" -lt 500 ]; then
    pass "Malformed JSON → HTTP ${http_status} (4xx)"
else
    fail "Malformed JSON should be 4xx" "4xx" "HTTP ${http_status}"
fi

# ── Summary ────────────────────────────────────────────────────────────────────

echo ""
echo "────────────────────────────────────────────"
printf "Results: %d passed, %d failed / %d total\n" "$PASS" "$FAIL" "$TOTAL"
echo "────────────────────────────────────────────"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
