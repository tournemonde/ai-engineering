"""Streamlit conversational client for the Session 5 estimator.

UI states
---------
1. First interaction (no turns yet): full typed form at the top
   (transcript + project_type + detail_level + output_format + attachments).
2. After the first successful estimate: hide the typed form; show the
   conversation history and a small composer (text + optional attachments)
   to refine the estimate. Typed options stay in session_state from turn 1.
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
            st.session_state.project_type = ProjectType.WEB_SAAS.value
            st.session_state.detail_level = DetailLevel.MEDIUM.value
            st.session_state.output_format = OutputFormat.PHASES_TABLE.value
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
        # Keep last typed options; user can reset via Nueva conversación + full form.
        st.session_state.project_type = ProjectType.WEB_SAAS.value
        st.session_state.detail_level = DetailLevel.MEDIUM.value
        st.session_state.output_format = OutputFormat.PHASES_TABLE.value
    except httpx.HTTPError as exc:
        st.session_state.last_error = f"Could not reset session: {exc}"


def _render_estimation(result: dict[str, Any]) -> None:
    """Render a structured EstimationResult (summary + phases table)."""
    summary = result.get("summary") or "(no summary)"
    st.text(summary)

    c1, c2, c3 = st.columns(3)
    c1.metric("Duration", f"{result.get('total_duration_weeks', '?')} wk")
    c2.metric("Cost", f"{result.get('total_cost_eur', '?')} EUR")
    c3.metric("Confidence", f"{result.get('confidence_pct', '?')}%")

    phases = result.get("phases") or []
    rows = [
        {
            "Phase": phase.get("name", "?"),
            "Weeks": phase.get("duration_weeks", "?"),
            "Cost (EUR)": phase.get("cost_eur", "?"),
            "Summary": phase.get("summary", ""),
        }
        for phase in phases
        if isinstance(phase, dict)
    ]
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)


def _metadata_is_empty(metadata: dict[str, Any]) -> bool:
    return not metadata or (
        metadata.get("project_name") is None
        and metadata.get("assumed_team_size") is None
        and not metadata.get("mentioned_technologies")
        and metadata.get("agreed_scope") is None
    )


def _post_estimate(*, transcript: str, attachments: list[Any] | None) -> None:
    """POST one turn, append to history, then rerun."""
    turns: list[dict[str, Any]] = list(st.session_state.get("turns") or [])

    if st.session_state.session_id is None:
        st.session_state.last_error = "No active session. Click Nueva conversación."
        st.rerun()
    if len(transcript.strip()) < 20:
        st.session_state.last_error = "The message must be at least 20 characters long."
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
        "project_type": st.session_state.project_type,
        "detail_level": st.session_state.detail_level,
        "output_format": st.session_state.output_format,
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
    st.rerun()


st.set_page_config(page_title="Software Estimator", page_icon="📊")
st.title("Software Estimator")

_ensure_session()

if st.session_state.get("last_error") and not st.session_state.get("session_id"):
    st.error(st.session_state.last_error)
    st.stop()

turns: list[dict[str, Any]] = st.session_state.get("turns") or []
has_conversation = len(turns) > 0

with st.sidebar:
    st.header("Session")
    st.code(st.session_state.get("session_id") or "(none)", language="text")
    if st.button("Nueva conversación"):
        _reset_session()
        st.rerun()

    if has_conversation:
        st.caption("Typed options locked from turn 1:")
        st.markdown(f"**Project type:** `{st.session_state.project_type}`")
        st.markdown(f"**Detail level:** `{st.session_state.detail_level}`")
        st.markdown(f"**Output format:** `{st.session_state.output_format}`")

    st.header("Project metadata")
    metadata = st.session_state.get("project_metadata") or {}
    if _metadata_is_empty(metadata):
        st.caption("Empty — first turn of the session.")
    else:
        st.json(metadata)

    st.header("Service")
    st.code(SESSIONS_ENDPOINT, language="text")
    st.markdown(f"**Primary model:** `{os.getenv('PRIMARY_MODEL', 'gpt-4o-mini')}`")
    st.markdown(f"**Fallback model:** `{os.getenv('FALLBACK_MODEL', 'claude-haiku-4-5-20251001')}`")
    st.markdown(f"**Max turns:** `{os.getenv('MAX_CONVERSATION_TURNS', '6')}`")

if st.session_state.get("last_error"):
    st.error(st.session_state.last_error)

# --- State 1: first interaction — full typed form at the top ----------------
if not has_conversation:
    st.caption(
        "Fill in the typed form to produce the first estimate. "
        "After that you can refine with a short message or attachments."
    )
    with st.form("initial_estimation_form", clear_on_submit=False):
        transcript = st.text_area(
            "Project description / transcript",
            height=200,
            placeholder="Describe the project: goals, key features, constraints…",
            help="Between 20 and 80000 characters.",
        )
        project_type = st.selectbox(
            "Project type",
            options=[t.value for t in ProjectType],
            index=1,
        )
        detail_level = st.radio(
            "Detail level",
            options=[d.value for d in DetailLevel],
            index=1,
            horizontal=True,
        )
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
        submitted = st.form_submit_button("Generate estimation", type="primary")

    if submitted:
        st.session_state.project_type = project_type
        st.session_state.detail_level = detail_level
        st.session_state.output_format = output_format
        _post_estimate(transcript=transcript, attachments=attachments)

# --- State 2: conversation started — results + refine box only --------------
else:
    st.caption(
        "Refine the estimate below, or attach PDF/DOCX. "
        "Typed options stay as in the first turn (see sidebar)."
    )
    st.header("Conversation")
    for idx, turn in enumerate(turns, start=1):
        with st.expander(f"Turn {idx}", expanded=(idx == len(turns))):
            st.caption("You")
            st.text(turn.get("transcript", ""))
            st.caption(f"Prompt `{turn.get('prompt_version', '?')}` · cached={turn.get('cached')}")
            _render_estimation(turn.get("result") or {})

    st.divider()
    with st.form("refine_composer", clear_on_submit=True):
        transcript = st.text_area(
            "Continue the conversation",
            height=120,
            placeholder="Refine scope, correct assumptions, or add details…",
        )
        attachments = st.file_uploader(
            "Attachments (PDF or DOCX)",
            type=["pdf", "docx"],
            accept_multiple_files=True,
        )
        submitted = st.form_submit_button("Send", type="primary")

    if submitted:
        _post_estimate(transcript=transcript, attachments=attachments)
