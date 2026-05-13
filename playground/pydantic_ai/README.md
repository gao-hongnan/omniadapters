# pydantic-ai demo

A small config-driven FastAPI service that builds a registry of `pydantic_ai.Agent` instances from a YAML file (one agent per provider) and exposes them over HTTP. Provider credentials and model selection are pulled from `.env` + YAML at startup; each request is a dict lookup plus `agent.run(prompt)`.

## Install

```bash
uv sync --extra playground
```

## Configure

```bash
cp playground/pydantic_ai/.env.sample playground/pydantic_ai/.env
# Edit .env to add real API keys for openai_mini / anthropic_haiku
# Edit playground/pydantic_ai/config/config.yaml to add or remove agents
```

The `test` agent uses pydantic-ai's built-in `TestModel` and works with no real API key — useful for smoke-testing the wiring.

### Provider support

This demo includes `openai`, `anthropic`, and the built-in `test` provider.
`gemini` and `azure-openai` are intentionally omitted: omniadapters' provider
names (`gemini`, `azure-openai`) do not match the names pydantic-ai's
`infer_provider_class` expects (`google-gla`/`google-vertex`, `azure`), so they
fail to instantiate through `PydanticAIAdapter` today.

## Run

```bash
uv run -m playground.pydantic_ai.main \
  --env-file playground/pydantic_ai/.env \
  --yaml-file playground/pydantic_ai/config/config.yaml \
  --port 8000
```

## Try it

```bash
# health check
curl -s http://127.0.0.1:8000/health

# list configured agents
curl -s http://127.0.0.1:8000/agents | jq

# completion via the test agent (no API key needed)
curl -s -X POST http://127.0.0.1:8000/completions \
  -H "Content-Type: application/json" \
  -d '{"agent":"test","prompt":"hello"}'

# real provider call (requires key in .env)
curl -s -X POST http://127.0.0.1:8000/completions \
  -H "Content-Type: application/json" \
  -d '{"agent":"openai_mini","prompt":"Reply with the word OK."}'
```

Unknown agent names return HTTP 404.
