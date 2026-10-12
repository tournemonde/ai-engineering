"""Session 6: aggregated ``turn_observed`` event via GET /sessions/{id}."""

from __future__ import annotations

from typing import Any

TURN_OBSERVED_FIELDS: frozenset[str] = frozenset(
    {
        "turn_index",
        "session_id",
        "enriched_transcript_chars",
        "attachments_total_chars",
        "messages_in_window",
        "anchors_count",
        "summary_chars",
        "tokens_in",
        "tokens_out",
        "cost_usd",
        "latency_ms",
        "cache_hit_kind",
        "last_resolved_tier",
    }
)


def _estimate_form() -> dict[str, str]:
    return {
        "transcript": (
            "We need a SaaS CRM called Nimbus for a 5-person sales team "
            "with React frontend and Postgres backend."
        ),
        "project_type": "web_saas",
        "detail_level": "medium",
        "output_format": "phases_table",
    }


def test_turn_observed_has_thirteen_fields_after_estimate(
    conversational_client: tuple[Any, Any],
) -> None:
    client, store = conversational_client
    created = client.post("/sessions")
    assert created.status_code == 201
    session_id = created.json()["session_id"]

    response = client.post(
        f"/sessions/{session_id}/estimate",
        data=_estimate_form(),
    )
    assert response.status_code == 200

    info = client.get(f"/sessions/{session_id}")
    assert info.status_code == 200
    body = info.json()
    last_turn = body["last_turn"]
    assert last_turn is not None
    assert set(last_turn.keys()) == TURN_OBSERVED_FIELDS
    assert last_turn["turn_index"] == 1
    assert last_turn["session_id"] == session_id
    assert last_turn["cache_hit_kind"] == "none"
    assert last_turn["enriched_transcript_chars"] > 0
    assert last_turn["attachments_total_chars"] == 0
    assert last_turn["tokens_in"] > 0
    assert last_turn["tokens_out"] > 0
    assert last_turn["latency_ms"] >= 0
    assert body["summary"] is None or isinstance(body["summary"], str)
    assert isinstance(body["anchors"], list)

    session = store.get_or_404(session_id)
    assert session.turn_count == 1
    assert session.last_turn == last_turn


def test_turn_index_increments_across_turns(
    conversational_client: tuple[Any, Any],
) -> None:
    client, _store = conversational_client
    session_id = client.post("/sessions").json()["session_id"]
    form = _estimate_form()

    client.post(f"/sessions/{session_id}/estimate", data=form)
    client.post(
        f"/sessions/{session_id}/estimate",
        data={
            **form,
            "transcript": (
                "Also add OAuth2 login and an audit log for compliance "
                "reviews every quarter please."
            ),
        },
    )

    last_turn = client.get(f"/sessions/{session_id}").json()["last_turn"]
    assert last_turn["turn_index"] == 2
