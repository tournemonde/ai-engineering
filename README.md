# ai-engineering — Estimador CAG

Servicio FastAPI + interfaz Streamlit que recibe la transcripción de una reunión
y devuelve una estimación de software generada por un LLM con arquitectura
**CAG** (contexto estático inyectado en el system prompt).

## Requisitos

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- API key de OpenAI y/o Anthropic

## Setup

```bash
uv sync
cp .env.example .env
# Edita .env y añade OPENAI_API_KEY y/o ANTHROPIC_API_KEY
```

Variables relevantes:

| Variable | Descripción |
|---|---|
| `LLM_PROVIDER` | `openai` (default) o `anthropic` |
| `LLM_MODEL` | Por defecto `gpt-4o-mini` / `claude-haiku-4-5` |
| `OPENAI_API_KEY` | Obligatoria si `LLM_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | Obligatoria si `LLM_PROVIDER=anthropic` |

## Arranque — API FastAPI

```bash
uv run uvicorn app.main:app --reload
```

- Health: [http://localhost:8000/health](http://localhost:8000/health)
- Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)

## Arranque — Interfaz Streamlit (Sesión 3)

Interfaz de chat que reutiliza el mismo system prompt CAG y settings que el
endpoint FastAPI. Llama al LLM en streaming (no pasa por HTTP).

```bash
uv run streamlit run streamlit_app.py
```

Transcripción de ejemplo: [`data/sample_transcription.txt`](data/sample_transcription.txt).
Pégala en el chat; la estimación se escribe token a token. El panel lateral
muestra el system prompt, los ejemplos CAG y las métricas de la última llamada.

## Probar el endpoint

```bash
curl -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d "{\"transcription\": \"$(cat data/sample_transcription.txt)\"}"
```

Respuesta esperada (forma):

```json
{
  "estimation": "## Estimación: ...",
  "model": "gpt-4o-mini",
  "provider": "openai",
  "input_tokens": 1234,
  "output_tokens": 567
}
```

## Arquitectura

```
app/
├── main.py                 # FastAPI + /health
├── config.py               # pydantic-settings
├── routers/estimations.py  # POST /api/v1/estimate
├── services/llm_service.py # prompt CAG + OpenAI/Anthropic (async + stream sync)
├── schemas/estimation.py   # request/response contracts
└── context/examples.py     # few-shot estático
streamlit_app.py            # chat UI (session_state + st.write_stream)
```

Flujo API: request → router (valida) → `llm_service` (inyecta ejemplos + llama al modelo) → JSON.

Flujo Streamlit: chat input → `stream_estimation` (mismo prompt CAG) → tokens vía `st.write_stream`.

## Tests y CI

```bash
uv run ruff check app tests streamlit_app.py
uv run ruff format --check app tests streamlit_app.py
uv run pytest -v
```

GitHub Actions (`.github/workflows/ci.yml`) ejecuta lint, tests (estimate/stream mockeados)
y smoke de `/health` sin llamadas de pago al LLM.

## Entrega del curso

Rama de trabajo (Sesión 3): `session3/conversational_interface_wrapper`
