"""Build deterministic synthetic PDFs for attachment stress sizes.

Usage::

    uv run python -m evals.stress.fixtures.build_pdfs

Produces ``attach_{5,20,50,100}kb.pdf`` under ``evals/stress/fixtures/generated/``
(gitignored). Each PDF embeds a unique marker string used for recall checks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fpdf import FPDF

# snippet: target sizes and recall markers
TARGET_SIZES_KB: dict[str, int] = {
    "attach_5kb.pdf": 5,
    "attach_20kb.pdf": 20,
    "attach_50kb.pdf": 50,
    "attach_100kb.pdf": 100,
}

MARKERS: dict[str, str] = {
    "attach_5kb.pdf": "STRESS_MARKER_ATTACH_5KB",
    "attach_20kb.pdf": "STRESS_MARKER_ATTACH_20KB",
    "attach_50kb.pdf": "STRESS_MARKER_ATTACH_50KB",
    "attach_100kb.pdf": "STRESS_MARKER_ATTACH_100KB",
}

_LOREM = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "generated"


def _write_pdf(path: Path, *, marker: str, line_count: int) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 6, text=f"Stress fixture marker: {marker}", new_x="LMARGIN", new_y="NEXT")
    for i in range(line_count):
        if pdf.get_y() > 270:
            pdf.add_page()
            pdf.set_font("Helvetica", size=10)
        pdf.cell(
            0,
            5,
            text=f"{_LOREM} line={i} marker={marker}",
            new_x="LMARGIN",
            new_y="NEXT",
        )
    pdf.output(str(path))


def build_pdf(*, marker: str, target_bytes: int, path: Path) -> None:
    """Write a PDF whose on-disk size is at least ``target_bytes``."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Exponential growth until we meet the size, then binary-search the line count.
    line_count = 8
    _write_pdf(path, marker=marker, line_count=line_count)
    while path.stat().st_size < target_bytes and line_count < 100_000:
        line_count *= 2
        _write_pdf(path, marker=marker, line_count=line_count)

    if path.stat().st_size < target_bytes:
        # Fallback: append a PDF comment trailer with raw padding (still a valid PDF).
        padding = b"\n% " + (b"PAD" * ((target_bytes - path.stat().st_size) // 3 + 3))
        with path.open("ab") as fh:
            fh.write(padding)
        return

    # Shrink to the smallest line_count that still meets the target.
    low, high = max(1, line_count // 2), line_count
    best = high
    while low <= high:
        mid = (low + high) // 2
        _write_pdf(path, marker=marker, line_count=mid)
        if path.stat().st_size >= target_bytes:
            best = mid
            high = mid - 1
        else:
            low = mid + 1
    _write_pdf(path, marker=marker, line_count=best)


def build_all(output_dir: Path | None = None) -> dict[str, Path]:
    out = output_dir or _default_output_dir()
    built: dict[str, Path] = {}
    for filename, kb in TARGET_SIZES_KB.items():
        path = out / filename
        build_pdf(marker=MARKERS[filename], target_bytes=kb * 1024, path=path)
        built[filename] = path
        print(f"{filename}: {path.stat().st_size} bytes (target {kb} KB)")
    return built


def main() -> int:
    parser = argparse.ArgumentParser(description="Build synthetic stress PDFs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated PDFs (default: fixtures/generated)",
    )
    args = parser.parse_args()
    build_all(output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
