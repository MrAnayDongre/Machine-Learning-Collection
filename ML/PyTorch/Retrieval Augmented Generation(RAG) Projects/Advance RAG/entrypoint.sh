#!/bin/bash

# Get the value of LLM_MODEL environment variable
ollama_server=$OLLAMA_SERVER
llm_model=$LLM_MODEL
embedding_model=$EMBEDDING_MODEL

# Check if OLLAMA_SERVER environment variable is set
if [ -z "$ollama_server" ]; then
  echo "OLLAMA_SERVER environment variable is not set."
  exit 1
fi

# Check if LLM_MODEL environment variable is set
if [ -z "$llm_model" ]; then
  echo "LLM_MODEL environment variable is not set."
  exit 1
fi

# Check if EMBEDDING_MODEL environment variable is set
if [ -z "$embedding_model" ]; then
  echo "EMBEDDING_MODEL environment variable is not set."
  exit 1
fi

# Pull models from Ollama server
curl -X POST $ollama_server/api/pull -d "{\"name\": \"$embedding_model\"}"
curl -X POST $ollama_server/api/pull -d "{\"name\": \"$llm_model\"}"

# Start the FastAPI server
uvicorn api:app --host "0.0.0.0" --port 80