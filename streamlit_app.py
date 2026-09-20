"""Streamlit conversational UI for the CAG software estimator."""

from __future__ import annotations

from typing import Any

import streamlit as st

from app.config import Settings, get_settings
from app.context.examples import ESTIMATION_EXAMPLES
from app.services.llm_service import StreamMetrics, build_system_prompt, stream_estimation


def _load_settings() -> Settings | None:
    """Load settings from .env; show an error if credentials are missing."""
    try:
        return get_settings()
    except Exception as exc:  # noqa: BLE001 — surface config errors in the UI
        st.error(f"No se pudo cargar la configuración: {exc}")
        st.info(
            "Copia `.env.example` a `.env` y define `OPENAI_API_KEY` o "
            "`ANTHROPIC_API_KEY` según `LLM_PROVIDER`."
        )
        return None


def _init_session_state() -> None:
    """Ensure chat history and last-call metrics exist in session_state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_metrics" not in st.session_state:
        st.session_state.last_metrics = None


def _render_sidebar(system_prompt: str, settings: Settings) -> None:
    """Level 3: CAG context visibility (prompt, examples, last-call metrics)."""
    with st.sidebar:
        st.header("Contexto CAG")
        st.caption(f"Proveedor: `{settings.LLM_PROVIDER}` · Modelo: `{settings.LLM_MODEL}`")

        with st.expander("System prompt (solo lectura)", expanded=False):
            st.code(system_prompt, language="markdown")

        with st.expander("Estimaciones de referencia", expanded=False):
            for index, example in enumerate(ESTIMATION_EXAMPLES, start=1):
                st.subheader(f"Ejemplo {index}")
                st.markdown("**Resumen de la reunión**")
                st.write(example["meeting_summary"])
                st.markdown("**Estimación**")
                st.markdown(example["estimation"])

        st.subheader("Última llamada")
        metrics: dict[str, Any] | None = st.session_state.last_metrics
        if not metrics:
            st.caption("Aún no hay métricas. Envía una transcripción.")
            return
        st.metric("Modelo", metrics.get("model") or "—")
        col_in, col_out = st.columns(2)
        col_in.metric("Tokens entrada", metrics.get("input_tokens") or "—")
        col_out.metric("Tokens salida", metrics.get("output_tokens") or "—")
        latency = metrics.get("latency_ms")
        st.metric("Latencia (ms)", latency if latency is not None else "—")
        st.caption(f"Proveedor: {metrics.get('provider') or '—'}")


def main() -> None:
    """Run the Streamlit chat app for meeting-transcription estimations."""
    st.set_page_config(page_title="Estimador CAG", page_icon="💬", layout="wide")
    st.title("Estimador de software (CAG)")
    st.caption(
        "Pega la transcripción de una reunión y recibe una estimación en streaming. "
        "Cada mensaje se estima de forma independiente (mismo prompt CAG que el API)."
    )

    settings = _load_settings()
    if settings is None:
        return

    _init_session_state()
    system_prompt = build_system_prompt()
    _render_sidebar(system_prompt, settings)

    # snippet: render conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # snippet: accept transcription and stream estimation
    if prompt := st.chat_input("Pega la transcripción de la reunión..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        metrics: StreamMetrics = {}
        with st.chat_message("assistant"):
            try:
                response = st.write_stream(
                    stream_estimation(prompt, settings=settings, metrics=metrics)
                )
            except Exception as exc:  # noqa: BLE001 — show provider errors in chat
                response = f"Error al llamar al LLM: {exc}"
                st.error(response)

        st.session_state.messages.append({"role": "assistant", "content": str(response)})
        if metrics:
            st.session_state.last_metrics = dict(metrics)
            st.rerun()


if __name__ == "__main__":
    main()
