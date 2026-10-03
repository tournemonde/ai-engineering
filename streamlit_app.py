"""Streamlit conversational client for the Session 5 estimator.

Creates a session on load, posts multipart estimates to
``POST /sessions/{id}/estimate``, shows ``project_metadata`` in the sidebar,
and offers a "Nueva conversación" reset. The transactional
``POST /api/v1/estimate`` endpoint remains available for API clients.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st
from dotenv import load_dotenv

from app.schemas.estimation import DetailLevel, OutputFormat, ProjectType

load_dotenv()

API_BASE_URL = os.getenv("ESTIMATOR_API_BASE_URL", "http://localhost:8000").rstrip("/")
SESSIONS_ENDPOINT = f"{API_BASE_URL}/sessions"


def _create_session() -> str:
    response = httpx.post(SESSIONS_ENDPOINT, timeout=httpx.Timeout(30.0, connect=10.0))
    response.raise_for_status()
    return response.json()["session_id"]


def _ensure_session() -> None:
    if "session_id" not in st.session_state:
        try:
            st.session_state.session_id = _create_session()
            st.session_state.project_metadata = {}
            st.session_state.last_result = None
            st.session_state.last_error = None
            st.session_state.turns = []
        except httpx.HTTPError as exc:
            st.session_state.session_id = None
            st.session_state.last_error = f"Could not create session: {exc}"


def _reset_session() -> None:
    try:
        st.session_state.session_id = _create_session()
        st.session_state.project_metadata = {}
        st.session_state.last_result = None
        st.session_state.last_error = None
        st.session_state.turns = []
    except httpx.HTTPError as exc:
        st.session_state.last_error = f"Could not reset session: {exc}"


def _render_estimation(result: dict[str, Any]) -> None:
    """Render a structured EstimationResult defensively (no KeyError mid-page)."""
    st.subheader("Estimation")
    summary = result.get("summary") or "(no summary)"
    # Plain text avoids rare markdown/HTML edge cases freezing the frontend.
    st.text(summary)
    st.markdown(
        f"**Total:** {result.get('total_duration_weeks', '?')} weeks · "
        f"{result.get('total_cost_eur', '?')} EUR · "
        f"confidence {result.get('confidence_pct', '?')}%"
    )
    phases = result.get("phases") or []
    if not phases:
        return
    st.markdown("**Phases**")
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        name = phase.get("name", "?")
        weeks = phase.get("duration_weeks", "?")
        cost = phase.get("cost_eur", "?")
        phase_summary = phase.get("summary", "")
        st.markdown(f"- **{name}** — {weeks}w / {cost} EUR — {phase_summary}")


def _metadata_is_empty(metadata: dict[str, Any]) -> bool:
    return not metadata or (
        metadata.get("project_name") is None
        and metadata.get("assumed_team_size") is None
        and not metadata.get("mentioned_technologies")
        and metadata.get("agreed_scope") is None
    )


st.set_page_config(page_title="Software Estimator", page_icon="📊")
st.title("Software Estimator")
st.caption(
    "Conversational estimation with session memory and optional PDF/DOCX attachments. "
    "Write a follow-up in the box at the bottom after each estimate."
)

_ensure_session()

if st.session_state.get("last_error") and not st.session_state.get("session_id"):
    st.error(st.session_state.last_error)
    st.stop()

# --- Prior turns (above the composer, chat-like) ----------------------------
turns: list[dict[str, Any]] = st.session_state.get("turns") or []
if turns:
    st.header("Conversation")
    for idx, turn in enumerate(turns, start=1):
        with st.expander(f"Turn {idx}", expanded=(idx == len(turns))):
            st.caption("You")
            st.text(turn.get("transcript", ""))
            st.caption(f"Prompt `{turn.get('prompt_version', '?')}` · cached={turn.get('cached')}")
            _render_estimation(turn.get("result") or {})

if st.session_state.get("last_error"):
    st.error(st.session_state.last_error)

# --- Composer always at the bottom so the next message is obvious ----------
st.header("Next message")
with st.form("estimation_form", clear_on_submit=True):
    transcript = st.text_area(
        "Transcript / refinement",
        height=160,
        placeholder="Describe the project, or refine the previous turn…",
        help="At least 20 characters. Attachments enrich this turn's context.",
    )
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        project_type = st.selectbox(
            "Project type",
            options=[t.value for t in ProjectType],
            index=1,
        )
    with col_b:
        detail_level = st.selectbox(
            "Detail level",
            options=[d.value for d in DetailLevel],
            index=1,
        )
    with col_c:
        output_format = st.selectbox(
            "Output format",
            options=[f.value for f in OutputFormat],
            index=0,
        )
    attachments = st.file_uploader(
        "Attachments (PDF or DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )
    submitted = st.form_submit_button("Estimate / refine", type="primary")

if submitted:
    if st.session_state.session_id is None:
        st.session_state.last_error = "No active session. Click Nueva conversación."
        st.rerun()
    if len(transcript.strip()) < 20:
        st.session_state.last_error = "The transcript must be at least 20 characters long."
        st.rerun()

    files = [
        (
            "attachments",
            (upload.name, upload.getvalue(), upload.type or "application/octet-stream"),
        )
        for upload in (attachments or [])
    ]
    data = {
        "transcript": transcript.strip(),
        "project_type": project_type,
        "detail_level": detail_level,
        "output_format": output_format,
    }
    endpoint = f"{SESSIONS_ENDPOINT}/{st.session_state.session_id}/estimate"
    with st.spinner("Calling the estimator service…"):
        try:
            response = httpx.post(
                endpoint,
                data=data,
                files=files or None,
                timeout=httpx.Timeout(180.0, connect=10.0),
            )
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            st.session_state.last_error = (
                f"Service returned {exc.response.status_code}: {exc.response.text[:800]}"
            )
            st.rerun()
        except httpx.HTTPError as exc:
            st.session_state.last_error = f"Could not reach the estimator: {exc}"
            st.rerun()

    st.session_state.last_error = None
    st.session_state.last_result = body
    st.session_state.project_metadata = body.get("project_metadata") or {}
    st.session_state.turns = [
        *turns,
        {
            "transcript": transcript.strip(),
            "prompt_version": body.get("prompt_version"),
            "cached": body.get("cached"),
            "result": body.get("result") or {},
        },
    ]
    # Fresh run: form is idle again (avoids Streamlit "stuck after submit" UI).
    st.rerun()

with st.sidebar:
    st.header("Session")
    st.code(st.session_state.get("session_id") or "(none)", language="text")
    if st.button("Nueva conversación"):
        _reset_session()
        st.rerun()

    st.header("Project metadata")
    metadata = st.session_state.get("project_metadata") or {}
    if _metadata_is_empty(metadata):
        st.caption("Empty — first turn of the session.")
    else:
        st.json(metadata)

    st.header("Service")
    st.code(SESSIONS_ENDPOINT, language="text")
    primary = os.getenv("PRIMARY_MODEL", "gpt-4o-mini")
    fallback = os.getenv("FALLBACK_MODEL", "claude-haiku-4-5-20251001")
    st.markdown(f"**Primary model:** `{primary}`")
    st.markdown(f"**Fallback model:** `{fallback}`")
    st.markdown(f"**Max turns:** `{os.getenv('MAX_CONVERSATION_TURNS', '6')}`")
