#!/bin/bash
# RHAIIS GPU Quickstart — Ask the running model a question
# Looks up the served model via /v1/models, then sends your question to /v1/chat/completions.

set -e

PORT=8000
BASE_URL="http://127.0.0.1:${PORT}"

command -v curl &>/dev/null || { echo "curl is required."; exit 1; }
command -v jq &>/dev/null || { echo "jq is required."; exit 1; }

MODEL=$(curl -sf "${BASE_URL}/v1/models" | jq -r '.data[0].id' 2>/dev/null)
if [ -z "$MODEL" ] || [ "$MODEL" = "null" ]; then
    echo "Couldn't reach the inference server at ${BASE_URL}. Is it running? (./start.sh)"
    exit 1
fi

echo "Model: $MODEL"
read -rp "Ask a question: " QUESTION

if [ -z "$QUESTION" ]; then
    echo "No question entered."
    exit 1
fi

BODY=$(jq -n --arg model "$MODEL" --arg q "$QUESTION" \
    '{model: $model, messages: [{role: "user", content: $q}], max_tokens: 300}')

RESPONSE=$(curl -s "${BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$BODY")

echo
echo "$RESPONSE" | jq -r '.choices[0].message.content'
