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
        except httpx.HTTPError as exc:
            st.session_state.session_id = None
            st.session_state.last_error = f"Could not create session: {exc}"


def _reset_session() -> None:
    try:
        st.session_state.session_id = _create_session()
        st.session_state.project_metadata = {}
        st.session_state.last_result = None
        st.session_state.last_error = None
    except httpx.HTTPError as exc:
        st.session_state.last_error = f"Could not reset session: {exc}"


def _render_estimation(result: dict[str, Any]) -> None:
    st.subheader("Estimation")
    st.markdown(result.get("summary", ""))
    st.markdown(
        f"**Total:** {result.get('total_duration_weeks')} weeks · "
        f"{result.get('total_cost_eur')} EUR · "
        f"confidence {result.get('confidence_pct')}%"
    )
    phases = result.get("phases") or []
    if phases:
        st.markdown("**Phases**")
        for phase in phases:
            st.markdown(
                f"- **{phase['name']}** — {phase['duration_weeks']}w / "
                f"{phase['cost_eur']} EUR — {phase['summary']}"
            )


st.set_page_config(page_title="Software Estimator", page_icon="📊")
st.title("Software Estimator")
st.caption(
    "Conversational estimation with session memory and optional PDF/DOCX attachments. "
    "Project facts persist across turns even when the sliding history window drops old messages."
)

_ensure_session()

if st.session_state.get("last_error") and not st.session_state.get("session_id"):
    st.error(st.session_state.last_error)
    st.stop()

with st.form("estimation_form", clear_on_submit=False):
    transcript = st.text_area(
        "Transcript / refinement",
        height=200,
        placeholder="Describe the project, or refine the previous turn…",
        help="At least 20 characters. Attachments enrich this turn's context.",
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
    submitted = st.form_submit_button("Estimate / refine", type="primary")

if submitted:
    if st.session_state.session_id is None:
        st.error("No active session. Click Nueva conversación and try again.")
    elif len(transcript.strip()) < 20:
        st.error("The transcript must be at least 20 characters long.")
    else:
        files = []
        for upload in attachments or []:
            files.append(
                (
                    "attachments",
                    (upload.name, upload.getvalue(), upload.type or "application/octet-stream"),
                )
            )
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
                    f"Service returned {exc.response.status_code}: {exc.response.text}"
                )
                st.error(st.session_state.last_error)
            except httpx.HTTPError as exc:
                st.session_state.last_error = f"Could not reach the estimator: {exc}"
                st.error(st.session_state.last_error)
            else:
                st.session_state.last_error = None
                st.session_state.last_result = body
                st.session_state.project_metadata = body.get("project_metadata") or {}
                st.markdown(f"**Prompt version:** `{body.get('prompt_version', '?')}`")
                st.markdown(f"**Cached:** `{body.get('cached')}`")
                _render_estimation(body.get("result") or {})

elif st.session_state.get("last_result"):
    body = st.session_state.last_result
    st.markdown(f"**Prompt version:** `{body.get('prompt_version', '?')}`")
    _render_estimation(body.get("result") or {})

with st.sidebar:
    st.header("Session")
    st.code(st.session_state.get("session_id") or "(none)", language="text")
    if st.button("Nueva conversación"):
        _reset_session()
        st.rerun()

    st.header("Project metadata")
    metadata = st.session_state.get("project_metadata") or {}
    if not metadata or (
        metadata.get("project_name") is None
        and metadata.get("assumed_team_size") is None
        and not metadata.get("mentioned_technologies")
        and metadata.get("agreed_scope") is None
    ):
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
