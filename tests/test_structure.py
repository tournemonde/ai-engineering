"""Verify the Session 2 scaffolding folder layout."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_expected_app_layout_exists() -> None:
    expected = [
        ROOT / "app" / "__init__.py",
        ROOT / "app" / "main.py",
        ROOT / "app" / "config.py",
        ROOT / "app" / "routers" / "__init__.py",
        ROOT / "app" / "routers" / "estimations.py",
        ROOT / "app" / "services" / "__init__.py",
        ROOT / "app" / "services" / "llm_service.py",
        ROOT / "app" / "schemas" / "__init__.py",
        ROOT / "app" / "schemas" / "estimation.py",
        ROOT / "app" / "context" / "__init__.py",
        ROOT / "app" / "context" / "examples.py",
        ROOT / ".env.example",
        ROOT / "pyproject.toml",
        ROOT / "README.md",
        ROOT / "streamlit_app.py",
        ROOT / "data" / "sample_transcription.txt",
    ]
    missing = [str(path.relative_to(ROOT)) for path in expected if not path.exists()]
    assert missing == [], f"Missing scaffolding files: {missing}"
