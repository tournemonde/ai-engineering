"""Step 7 exercise tests using ``httpx.AsyncClient`` + ``ASGITransport``.

Covers the three acceptance criteria from the Session 5 exercise:
1. Two turns accumulate ``project_metadata``.
2. A PDF attachment reaches the LLM as enriched transcript text.
3. Eight turns never inflate the messages array beyond ``MAX_TURNS``.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from app.dependencies import (
    get_estimation_service,
    get_llm_wrapper,
    get_openai_client,
    get_session_store,
)
from app.main import app
from app.schemas.estimation import EstimationResult
from app.services.estimation import EstimationService
from app.sessions.models import ProjectMetadata
from app.sessions.store import SessionStore
from tests.conftest import FakeLLMWrapper, make_canned_result

VALID_FORM = {
    "transcript": "We want a CRM called Nimbus built with React and Postgres for the sales team.",
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "phases_table",
}


@pytest_asyncio.fixture
async def async_client(fake_wrapper: FakeLLMWrapper):
    store = SessionStore(max_turns=3)
    service = EstimationService(
        llm_wrapper=fake_wrapper,
        exact_cache=None,
        semantic_cache=None,
        openai_client=None,
        metadata_extractor_model="gpt-4o-mini",
    )
    app.dependency_overrides[get_estimation_service] = lambda: service
    app.dependency_overrides[get_session_store] = lambda: store
    app.dependency_overrides[get_llm_wrapper] = lambda: fake_wrapper
    app.dependency_overrides[get_openai_client] = lambda: None

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, store, fake_wrapper

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_async_two_turns_accumulate_metadata(
    async_client: tuple[httpx.AsyncClient, SessionStore, FakeLLMWrapper],
) -> None:
    client, _store, fake_wrapper = async_client
    fake_wrapper.scripted = [
        (
            make_canned_result(),
            ProjectMetadata(
                project_name="Nimbus",
                assumed_team_size=3,
                mentioned_technologies=["React", "Postgres"],
                agreed_scope="Phase 1 MVP CRM for sales team.",
            ),
        ),
        (
            make_canned_result(),
            ProjectMetadata(
                project_name=None,
                assumed_team_size=None,
                mentioned_technologies=["Stripe"],
                agreed_scope="Phase 1 MVP CRM with billing.",
            ),
        ),
    ]

    created = await client.post("/sessions")
    assert created.status_code == 201
    session_id = created.json()["session_id"]

    r1 = await client.post(f"/sessions/{session_id}/estimate", data=VALID_FORM)
    assert r1.status_code == 200, r1.text
    assert r1.json()["project_metadata"]["project_name"] == "Nimbus"

    r2 = await client.post(
        f"/sessions/{session_id}/estimate",
        data={**VALID_FORM, "transcript": "Now add Stripe-based billing on top."},
    )
    assert r2.status_code == 200, r2.text
    meta = r2.json()["project_metadata"]
    assert meta["project_name"] == "Nimbus"
    assert sorted(meta["mentioned_technologies"]) == sorted(["React", "Postgres", "Stripe"])
    assert "billing" in meta["agreed_scope"].lower()


@pytest.mark.asyncio
async def test_async_pdf_attachment_reaches_llm(
    async_client: tuple[httpx.AsyncClient, SessionStore, FakeLLMWrapper],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _store, fake_wrapper = async_client
    fake_wrapper.add_turn()

    fake_pages = [
        SimpleNamespace(extract_text=lambda: "Nimbus PDF spec line 1"),
        SimpleNamespace(extract_text=lambda: "Nimbus PDF spec line 2"),
    ]
    import pypdf

    monkeypatch.setattr(pypdf, "PdfReader", lambda _stream: SimpleNamespace(pages=fake_pages))

    session_id = (await client.post("/sessions")).json()["session_id"]
    response = await client.post(
        f"/sessions/{session_id}/estimate",
        data=VALID_FORM,
        files=[("attachments", ("proposal.pdf", b"%PDF-fake-bytes", "application/pdf"))],
    )
    assert response.status_code == 200, response.text

    estimation_call = fake_wrapper.chat_calls[0]
    last_user = next(m for m in reversed(estimation_call["messages"]) if m["role"] == "user")
    assert "--- attachment: proposal.pdf ---" in last_user["content"]
    assert "Nimbus PDF spec line 1" in last_user["content"]


@pytest.mark.asyncio
async def test_async_eight_turns_never_exceed_window(
    async_client: tuple[httpx.AsyncClient, SessionStore, FakeLLMWrapper],
) -> None:
    client, store, fake_wrapper = async_client
    session_id = (await client.post("/sessions")).json()["session_id"]

    for n in range(8):
        response = await client.post(
            f"/sessions/{session_id}/estimate",
            data={
                **VALID_FORM,
                "transcript": f"Turn {n}: refine and add scope details here for clarity.",
            },
        )
        assert response.status_code == 200, response.text

    estimation_calls = [
        call
        for call in fake_wrapper.chat_calls
        if call["response_model"] == EstimationResult.__name__
    ]
    assert len(estimation_calls) == 8
    for idx, call in enumerate(estimation_calls):
        # max_turns=3 → ≤ 1 system + 6 history + 1 current user
        assert len(call["messages"]) <= 8, (
            f"Estimation call {idx} sent {len(call['messages'])} messages"
        )

    session = store.get_or_404(session_id)
    assert len(session.history.messages) <= 3 * 2
