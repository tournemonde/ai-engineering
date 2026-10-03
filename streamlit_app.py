"""Streamlit product form for the estimator.

The form collects typed parameters and POSTs them to ``/api/v1/estimate``.
The service still returns the instructor response (estimation, usage, cache,
cost, validation). Streamlit only displays that JSON.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("ESTIMATOR_API_BASE_URL", "http://localhost:8000")
ESTIMATE_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/api/v1/estimate"

PROJECT_TYPES = {
    "Mobile app": "mobile_app",
    "Web SaaS": "web_saas",
    "Internal tool": "internal_tool",
    "Data pipeline": "data_pipeline",
}
DETAIL_LEVELS = {
    "Summary": "summary",
    "Medium": "medium",
    "Detailed": "detailed",
}
OUTPUT_FORMATS = {
    "Phases table": "phases_table",
    "Line items": "line_items",
    "Narrative": "narrative",
}


def _post_estimate(payload: dict[str, str]) -> dict[str, Any]:
    """POST one estimation request and return the JSON body."""
    response = httpx.post(
        ESTIMATE_ENDPOINT,
        json=payload,
        timeout=httpx.Timeout(120.0, connect=10.0),
    )
    response.raise_for_status()
    body: dict[str, Any] = response.json()
    return body


st.set_page_config(page_title="Software Estimator", page_icon="📊")
st.title("Software Estimator")
st.caption(
    "Describe the project and choose how the estimate should look. "
    "The prompt is built on the server."
)

if "last_result" not in st.session_state:
    st.session_state.last_result = None

with st.form("estimate_form"):
    description = st.text_area(
        "Project description",
        height=180,
        placeholder="What should be built, for whom, and any constraints you already know.",
    )
    project_label = st.selectbox("Project type", list(PROJECT_TYPES), index=1)
    detail_label = st.pills(
        "Detail",
        list(DETAIL_LEVELS),
        default="Medium",
        selection_mode="single",
    )
    format_label = st.selectbox("Output format", list(OUTPUT_FORMATS), index=0)
    submitted = st.form_submit_button("Generate estimation")

if submitted:
    text = description.strip()
    if len(text) < 50:
        st.error("The description needs at least 50 characters.")
    elif detail_label is None:
        st.error("Choose a detail level.")
    else:
        payload = {
            "transcription": text,
            "project_type": PROJECT_TYPES[project_label],
            "detail_level": DETAIL_LEVELS[detail_label],
            "output_format": OUTPUT_FORMATS[format_label],
        }
        try:
            st.session_state.last_result = _post_estimate(payload)
        except httpx.HTTPError as exc:
            st.session_state.last_result = None
            st.error(f"Could not reach the estimator at `{ESTIMATE_ENDPOINT}`: {exc}")

result: dict[str, Any] | None = st.session_state.last_result
if result:
    st.markdown(result.get("estimation") or "")
    usage = result.get("usage") or {}
    st.caption(
        f"Model `{result.get('model')}` · "
        f"cache hit `{result.get('cache_hit')}` · "
        f"cost USD `{result.get('cost_usd')}` · "
        f"tokens in/out `{usage.get('input_tokens')}` / `{usage.get('output_tokens')}`"
    )
    validation = result.get("validation")
    if isinstance(validation, dict):
        st.caption(f"Structure score `{validation.get('score')}`")

with st.sidebar:
    st.header("Service")
    st.code(ESTIMATE_ENDPOINT, language="text")
    primary = os.getenv("PRIMARY_MODEL", "gpt-4o-mini")
    fallback = os.getenv("FALLBACK_MODEL", "claude-haiku-4-5-20251001")
    st.markdown(f"**Primary model:** `{primary}`")
    st.markdown(f"**Fallback model:** `{fallback}`")
    st.markdown(f"**Cache TTL:** `{os.getenv('CACHE_TTL', '86400')}s`")
