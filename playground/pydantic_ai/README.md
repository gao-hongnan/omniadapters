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

This demo supports `openai` and `anthropic` (plus the built-in `test` agent
for credential-free smoke tests). `gemini` and `azure-openai` are rejected at
config-load time by `AgentConfig._validate_provider_supported`: omniadapters'
provider names (`gemini`, `azure-openai`) do not match the names pydantic-ai's
`infer_provider_class` expects (`google-gla`/`google-vertex`, `azure`), so
they cannot currently be wired through `PydanticAIAdapter`. Adding one to
`config.yaml` raises a clear `ValueError` listing the supported providers.

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

## Generation params (`model_settings`)

Each agent can declare default generation parameters in YAML using
pydantic-ai's [`ModelSettings`](https://ai.pydantic.dev/api/settings/) TypedDict
(cross-provider keys: `temperature`, `max_tokens`, `top_p`, `tool_choice`,
`thinking`, `service_tier`, …):

```yaml
openai_mini:
  provider_config:
    <<: *openai-provider-config
  model_name: "gpt-4o-mini"
  model_settings:
    temperature: 0.0
    max_tokens: 500
```

Callers can override per request — the override is merged on top of the
agent's defaults via `pydantic_ai.settings.merge_model_settings`, and the
effective merged settings are echoed back in the response:

```bash
curl -s -X POST http://127.0.0.1:8000/completions \
  -H "Content-Type: application/json" \
  -d '{
        "agent":"openai_mini",
        "prompt":"Say OK.",
        "model_settings": {"temperature": 0.9, "max_tokens": 50}
      }'
# Response includes "model_settings": {"temperature": 0.9, "max_tokens": 50}
# (request keys win over agent defaults).
```

Unknown `model_settings` keys are silently stripped by Pydantic (TypedDict
`total=False` semantics) — pass only valid `ModelSettings` keys.
