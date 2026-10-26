"""Streamlit form for the estimator.

Streamlit acts as an HTTP client of the FastAPI service: it submits a typed
``EstimationRequest`` to ``POST /api/v1/estimate`` and renders the response
text. The endpoint URL is read from ``ESTIMATOR_API_BASE_URL`` (loaded from
the same ``.env`` as the API), so the same UI works against a local uvicorn
or against docker-compose.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st
from dotenv import load_dotenv

from app.domain.schemas.estimation import DetailLevel, OutputFormat, ProjectType

load_dotenv()

API_BASE_URL = os.getenv("ESTIMATOR_API_BASE_URL", "http://localhost:8000")
ESTIMATE_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/api/v1/estimate"


def _render_estimation_response(body: dict[str, Any]) -> None:
    """Render both legacy free-text and current structured response payloads."""
    st.markdown(f"**Prompt version:** `{body.get('prompt_version', '?')}`")

    # Backward-compatible path for old payloads.
    text = body.get("text")
    if isinstance(text, str) and text.strip():
        st.markdown(text)
        return

    result = body.get("result")
    if not isinstance(result, dict):
        st.warning("Response received but no renderable content was found.")
        st.json(body)
        return

    summary = result.get("summary", "")
    confidence_pct = result.get("confidence_pct")
    total_duration_weeks = result.get("total_duration_weeks")
    total_cost_eur = result.get("total_cost_eur")

    if summary:
        st.markdown(f"### Summary\n{summary}")

    kpi_1, kpi_2, kpi_3 = st.columns(3)
    kpi_1.metric(
        "Confidence", f"{confidence_pct}%" if confidence_pct is not None else "N/A"
    )
    kpi_2.metric(
        "Total duration",
        f"{total_duration_weeks} weeks" if total_duration_weeks is not None else "N/A",
    )
    kpi_3.metric(
        "Total cost", f"{total_cost_eur} EUR" if total_cost_eur is not None else "N/A"
    )

    phases = result.get("phases", [])
    if isinstance(phases, list) and phases:
        st.markdown("### Phases")
        st.table(
            [
                {
                    "name": phase.get("name"),
                    "duration_weeks": phase.get("duration_weeks"),
                    "cost_eur": phase.get("cost_eur"),
                    "summary": phase.get("summary"),
                }
                for phase in phases
                if isinstance(phase, dict)
            ]
        )


st.set_page_config(page_title="Software Estimator", page_icon="📊")
st.title("Software Estimator")
st.caption(
    "Fill in the form below to estimate a software project. "
    "The service produces a free-text estimation following the chosen format."
)


with st.form("estimation_form", clear_on_submit=False):
    description = st.text_area(
        "Project description",
        height=200,
        placeholder="Describe the project: goals, key features, constraints…",
        help="Between 20 and 2000 characters.",
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
    submitted = st.form_submit_button("Generate estimation", type="primary")


if submitted:
    if len(description.strip()) < 20:
        st.error("The description must be at least 20 characters long.")
    else:
        payload = {
            "description": description.strip(),
            "project_type": project_type,
            "detail_level": detail_level,
            "output_format": output_format,
        }
        with st.spinner("Calling the estimator service…"):
            try:
                response = httpx.post(
                    ESTIMATE_ENDPOINT,
                    json=payload,
                    timeout=httpx.Timeout(120.0, connect=10.0),
                )
                response.raise_for_status()
                body = response.json()
            except httpx.HTTPStatusError as exc:
                st.error(
                    f"Service returned {exc.response.status_code}: {exc.response.text}"
                )
            except httpx.HTTPError as exc:
                st.error(
                    f"Could not reach the estimator at `{ESTIMATE_ENDPOINT}`: {exc}"
                )
            else:
                _render_estimation_response(body)


with st.sidebar:
    st.header("Service")
    st.code(ESTIMATE_ENDPOINT, language="text")
    primary = os.getenv("PRIMARY_MODEL", "gpt-4o-mini")
    fallback = os.getenv("FALLBACK_MODEL", "claude-haiku-4-5-20251001")
    st.markdown(f"**Primary model:** `{primary}`")
    st.markdown(f"**Fallback model:** `{fallback}`")
    st.markdown(f"**Cache TTL:** `{os.getenv('CACHE_TTL', '86400')}s`")
