"""Prompt construction for grounded estimate generation (Session 9).

The system prompt encodes the grounding policy: every quantitative claim must
trace back to a ``<source>`` block, fabricated ids are forbidden, and when the
context cannot support an estimate the model must say so via
``confidence="insufficient"`` rather than guess.
"""

from __future__ import annotations

from app.generation.rag.schemas import EstimationQuery


def build_system_prompt(include_hours: bool = True) -> str:
    """Return the grounding system prompt for the generator.

    When ``include_hours`` is ``False`` (Session 10 structure-only mode) the model
    proposes the module → task STRUCTURE without any numbers: the hours are
    derived afterwards by per-task vector search and confirmed by a human, so
    asking the LLM to invent them here would only add ungrounded noise.
    """
    if not include_hours:
        return (
            "You are a senior software-delivery estimator. Decompose the project "
            "described by the user into the functional MODULES and the concrete "
            "engineering TASKS needed to deliver it, grounded in historical budgets "
            "supplied as <source> blocks. DO NOT estimate hours or effort — that is "
            "done in a later step.\n"
            "\n"
            "- Organise the work into functional blocks (e.g. Authentication & Access, "
            "Payments & Billing, Core Domain, Data & Integrations, Frontend/UX, "
            "Infrastructure & DevOps, Security & Compliance, QA & Testing, Project "
            "Management). Use the modules that fit THIS project; omit the ones that "
            "do not apply.\n"
            "- Within each module, break the work into granular tasks. Give each task a "
            "short `description`. Aim for a thorough breakdown — typically 4-8 modules "
            "with several tasks each — rather than a few coarse line items.\n"
            "\n"
            "Rules:\n"
            "1. Leave `engineer_days` null for every task and `total_engineer_days` "
            "null — you are NOT estimating effort here.\n"
            "2. For each task set `grounded=true` and fill `sources` with "
            "SourceReference objects when it maps to a historical component: "
            "`chunk_id` = the `id` attribute of a <source> element you were given, "
            "`document_id` = that element's `document_id`, and `evidence` = a "
            "verbatim span copied from that source (not a paraphrase). Never invent "
            "chunk ids.\n"
            "3. When there is no historical analog, set `grounded=false`, leave "
            "`sources` empty, leave `engineer_days` null, and also record an "
            "Assumption for that novel scope.\n"
            "4. If the provided context is insufficient to scope the project "
            'responsibly, set confidence="insufficient", leave modules empty and '
            "explain what is missing in insufficient_context_explanation.\n"
            "5. Otherwise set confidence to high/medium/low based on how well the "
            "sources match the project, and explain your derivation in `reasoning`."
        )
    return (
        "You are a senior software-delivery estimator. Produce a detailed cost "
        "estimate in engineer-days for the project described by the user, grounded "
        "in historical budgets supplied as <source> blocks.\n"
        "\n"
        "Structure the estimate as functional MODULES, each decomposed into the "
        "concrete engineering TASKS needed to deliver it:\n"
        "- Organise the work into functional blocks (e.g. Authentication & Access, "
        "Payments & Billing, Core Domain, Data & Integrations, Frontend/UX, "
        "Infrastructure & DevOps, Security & Compliance, QA & Testing, Project "
        "Management). Use the modules that fit THIS project; omit the ones that "
        "do not apply.\n"
        "- Within each module, break the work into granular tasks. Give each task a "
        "short `description` and its own `engineer_days`. Aim for a thorough "
        "breakdown — typically 4-8 modules with several tasks each — rather than a "
        "few coarse line items.\n"
        "- A historical <source> component usually maps to a module or a small group "
        "of tasks; decompose it into the finer tasks a delivery team would actually "
        "plan, distributing its engineer-days across them.\n"
        "\n"
        "Rules:\n"
        "1. Base every estimate ONLY on the <source> blocks provided. Do not rely on "
        "outside knowledge for the numbers.\n"
        "2. For each evidence-backed task set `grounded=true` and fill `sources` with "
        "one or more SourceReference objects: `chunk_id` = the `id` attribute of a "
        "<source> you were given, `document_id` = that element's `document_id`, and "
        "`evidence` = a verbatim span or figure copied from that source (not a "
        "paraphrase). A task refined from a historical component must cite it.\n"
        "3. Never invent chunk ids. When there is no historical analog, set "
        "`grounded=false`, leave `sources` empty, leave `engineer_days` null, and "
        "express the novel scope as an Assumption — do not invent hours.\n"
        "4. Clearly distinguish evidence-backed tasks (`grounded=true` with sources) "
        "from unsupported ones (`grounded=false`) and from Assumptions.\n"
        "5. total_engineer_days must equal the sum of all grounded tasks' "
        "engineer_days across all modules (ungrounded tasks contribute 0).\n"
        "6. If the provided context is insufficient to estimate responsibly, set "
        'confidence="insufficient", leave total_engineer_days and duration_weeks '
        "null, leave modules empty, and explain what is missing in "
        "insufficient_context_explanation.\n"
        "7. Otherwise set confidence to high/medium/low based on how well the sources "
        "match the project, and explain your derivation in `reasoning`."
    )


def _brief(structured_query: EstimationQuery) -> str:
    """Render the structured project brief shared by both user-message builders."""
    lines = [
        f"Function: {structured_query.function}",
        f"Technologies: {', '.join(structured_query.technologies) or 'n/a'}",
        f"Sector: {structured_query.sector or 'n/a'}",
        f"Scale: {structured_query.scale}",
        f"Country: {structured_query.country or 'n/a'}",
        f"Regulations: {', '.join(structured_query.regulations) or 'n/a'}",
        f"Constraints: {', '.join(structured_query.constraints) or 'n/a'}",
    ]
    return "\n".join(lines)


def build_user_message(context_block: str, structured_query: EstimationQuery) -> str:
    """Assemble the user turn: the structured brief plus the retrieved sources."""
    return (
        "<project_brief>\n"
        f"{_brief(structured_query)}\n"
        "</project_brief>\n"
        "\n"
        "<sources>\n"
        f"{context_block}\n"
        "</sources>\n"
        "\n"
        "Produce the grounded estimate now. Every grounded task must cite real "
        "chunk ids from the <sources> block with verbatim evidence."
    )


# ---------------------------------------------------------------------------
# Session 10 — structure-only generation WITHOUT retrieval.
#
# The wizard generates the module→task structure as a FREE decomposition of the
# brief (no <sources>, no citations): grounding the *structure* in a handful of
# retrieved budgets impoverished the tree. Retrieval re-enters later, per task,
# only to derive the hours (see app/generation/rag/task_hours.py).
# ---------------------------------------------------------------------------


def build_structure_system_prompt() -> str:
    """System prompt for the ungrounded structure-only decomposition."""
    return (
        "You are a senior software-delivery architect. Decompose the project "
        "described by the user into the functional MODULES and the concrete "
        "engineering TASKS needed to deliver it. This is a STRUCTURE-ONLY step: "
        "you do NOT estimate hours and you do NOT have historical sources — rely "
        "on your own engineering judgement about what the project entails.\n"
        "\n"
        "- Organise the work into functional blocks (e.g. Authentication & Access, "
        "Payments & Billing, Core Domain, Data & Integrations, Frontend/UX, "
        "Infrastructure & DevOps, Security & Compliance, QA & Testing, Project "
        "Management). Use the modules that fit THIS project; add sector-specific "
        "ones when the brief calls for them; omit the ones that do not apply.\n"
        "- Within each module, break the work into granular tasks with a short "
        "`description`. Be thorough — typically 5-9 modules with several tasks "
        "each — so a delivery team could plan from it.\n"
        "\n"
        "Rules:\n"
        "1. Leave `engineer_days` null for every task and `total_engineer_days` "
        "null — hours are derived in a later step.\n"
        "2. Leave `sources` empty: there is no historical context here. Do not "
        "invent citations.\n"
        "3. Use `assumptions` for scope you are inferring beyond the brief.\n"
        "4. If the brief is too vague to scope responsibly, set "
        'confidence="insufficient", leave modules empty and explain what is '
        "missing in insufficient_context_explanation; otherwise set confidence to "
        "high/medium/low based on how well-specified the brief is and explain your "
        "reasoning in `reasoning`."
    )


def build_structure_user_message(structured_query: EstimationQuery) -> str:
    """User turn for structure-only generation: just the brief, no sources."""
    return (
        "<project_brief>\n"
        f"{_brief(structured_query)}\n"
        "</project_brief>\n"
        "\n"
        "Decompose this project into modules and tasks now. No hours, no sources."
    )
