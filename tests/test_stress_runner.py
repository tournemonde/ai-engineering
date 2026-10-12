"""Smoke tests for stress scenarios, PDF fixtures, and runner CSV shape."""

from __future__ import annotations

from pathlib import Path

from evals.stress.fixtures.build_pdfs import MARKERS, TARGET_SIZES_KB, build_all, build_pdf
from evals.stress.run import CSV_COLUMNS, write_csv
from evals.stress.scenarios import SCENARIOS


def test_scenarios_have_twenty_turns_each() -> None:
    assert set(SCENARIOS) == {"growing", "pivot", "contradiction"}
    for profile in SCENARIOS.values():
        assert len(profile.turns) == 20
        assert profile.turns[0].turn_index == 1
        assert profile.turns[-1].turn_index == 20
        for turn in profile.turns:
            assert len(turn.transcript) >= 20
            assert turn.fact_to_remember


def test_pivot_changes_stack_at_turn_five() -> None:
    pivot = SCENARIOS["pivot"]
    assert "Flutter" in pivot.turns[4].transcript
    assert pivot.turns[4].fact_to_remember == "stack includes Flutter"


def test_contradiction_budget_flip() -> None:
    contradiction = SCENARIOS["contradiction"]
    assert "30000" in contradiction.turns[2].fact_to_remember
    assert "80000" in contradiction.turns[7].fact_to_remember


def test_build_pdf_meets_target_and_contains_marker(tmp_path: Path) -> None:
    path = tmp_path / "attach_5kb.pdf"
    marker = MARKERS["attach_5kb.pdf"]
    build_pdf(marker=marker, target_bytes=5 * 1024, path=path)
    assert path.exists()
    assert path.stat().st_size >= 5 * 1024


def test_build_all_creates_expected_files(tmp_path: Path) -> None:
    built = build_all(output_dir=tmp_path)
    assert set(built) == set(TARGET_SIZES_KB)
    for name, path in built.items():
        assert path.exists()
        assert path.stat().st_size >= TARGET_SIZES_KB[name] * 1024


def test_write_csv_header(tmp_path: Path) -> None:
    out = tmp_path / "results.csv"
    write_csv(
        out,
        [
            {
                "scenario": "growing",
                "repeat": 1,
                "attachment_size_kb": 0,
                "turn_index": 1,
                "session_id": "abc",
                "error": "",
            }
        ],
    )
    header = out.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == CSV_COLUMNS
