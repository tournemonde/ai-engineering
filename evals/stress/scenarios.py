"""Synthetic multi-turn conversation profiles for CAG stress testing.

Each profile is a list of ``(turn_index, transcript, fact_to_remember)``.
Facts feed ``MemoryDriftMetric`` on later turns (exact case-insensitive match
against summary / anchors / ProjectMetadata).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioTurn:
    turn_index: int
    transcript: str
    fact_to_remember: str


@dataclass(frozen=True)
class ScenarioProfile:
    name: str
    description: str
    turns: list[ScenarioTurn]


def _pad(text: str) -> str:
    """Ensure transcript meets the API min_length=20 floor."""
    if len(text) >= 20:
        return text
    return text + " " + ("please estimate this carefully." * 2)


def _growing_turns() -> list[ScenarioTurn]:
    specs: list[tuple[str, str]] = [
        (
            "We are starting project Nimbus: a B2B SaaS CRM for a five-person sales team. "
            "Stack preference is React on the frontend and Postgres on the backend.",
            "project name: Nimbus",
        ),
        (
            "Add email outreach sequences and a shared team inbox. Keep React and Postgres.",
            "feature: email outreach sequences",
        ),
        (
            "We need OAuth2 login with Google and Microsoft for all users of Nimbus.",
            "auth: OAuth2 Google Microsoft",
        ),
        (
            "Make the product multi-tenant with hard data isolation between customer orgs.",
            "architecture: multi-tenant isolation",
        ),
        (
            "Add a full audit log of every CRM record change for compliance reviews.",
            "feature: audit log",
        ),
        (
            "Support CSV export of contacts, deals, and activity from Nimbus reports.",
            "feature: CSV export",
        ),
        (
            "We need role-based access control with admin, manager, and rep roles.",
            "feature: RBAC admin manager rep",
        ),
        (
            "Add a REST API so partners can create leads programmatically in Nimbus.",
            "feature: public REST API",
        ),
        (
            "Include webhook notifications when a deal stage changes in the pipeline.",
            "feature: deal stage webhooks",
        ),
        (
            "Add Slack notifications for assigned tasks and overdue follow-ups.",
            "integration: Slack notifications",
        ),
        (
            "We need a mobile-responsive UI; desktop-first is fine for v1 of Nimbus.",
            "ui: mobile-responsive",
        ),
        (
            "Store files attached to deals in S3 with signed download URLs.",
            "storage: S3 signed URLs",
        ),
        (
            "Add two-factor authentication as an optional setting per organisation.",
            "security: optional 2FA",
        ),
        (
            "Support custom fields on contacts and deals configured by admins.",
            "feature: custom fields",
        ),
        (
            "Add a reporting dashboard with win rate and pipeline velocity charts.",
            "feature: reporting dashboard",
        ),
        (
            "We need GDPR tools: data export and delete-account flows for Nimbus.",
            "compliance: GDPR export delete",
        ),
        (
            "Add calendar sync with Google Calendar for meeting scheduling.",
            "integration: Google Calendar",
        ),
        (
            "Support bulk import of contacts from CSV with validation and dry-run.",
            "feature: bulk CSV import",
        ),
        (
            "Add email templates with merge fields for outreach sequences.",
            "feature: email templates",
        ),
        (
            "Finalise Nimbus v1 scope: freeze new features; focus estimation on delivery.",
            "scope: Nimbus v1 feature freeze",
        ),
    ]
    return [
        ScenarioTurn(turn_index=i + 1, transcript=_pad(t), fact_to_remember=f)
        for i, (t, f) in enumerate(specs)
    ]


def _pivot_turns() -> list[ScenarioTurn]:
    specs: list[tuple[str, str]] = [
        (
            "Project Atlas is a field-service companion for technicians. "
            "We plan a React web app backed by Node and Postgres.",
            "project name: Atlas",
        ),
        (
            "Atlas needs offline-capable job checklists and photo capture on site.",
            "feature: offline job checklists",
        ),
        (
            "Use React with a responsive layout for tablet use in the field.",
            "stack includes React",
        ),
        (
            "Add GPS check-in when a technician arrives at a customer site.",
            "feature: GPS check-in",
        ),
        (
            "Pivot: drop React web. Rebuild Atlas as a Flutter mobile app for iOS and Android. "
            "Keep the same Node and Postgres backend APIs.",
            "stack includes Flutter",
        ),
        (
            "Flutter UI should reuse the same job checklist domain model as before.",
            "ui: Flutter checklist reuse",
        ),
        (
            "Add push notifications for newly assigned jobs in the Flutter client.",
            "feature: push notifications",
        ),
        (
            "Support camera barcode scanning of asset tags inside Flutter.",
            "feature: barcode scanning",
        ),
        (
            "Sync job updates over a queue API when connectivity returns.",
            "feature: offline sync queue",
        ),
        (
            "Add biometric unlock option on the Flutter app for field security.",
            "security: biometric unlock",
        ),
        (
            "Managers still use a small React admin console; technicians use Flutter only.",
            "stack includes React admin",
        ),
        (
            "Atlas must work on iOS 16+ and Android 12+ for field devices.",
            "platform: iOS16 Android12",
        ),
        (
            "Add deep links so dispatch can open a specific job in Flutter.",
            "feature: deep links",
        ),
        (
            "Store job photos locally first, then upload to S3 when online.",
            "storage: local then S3",
        ),
        (
            "Add a dark mode theme in Flutter for night shifts.",
            "ui: Flutter dark mode",
        ),
        (
            "Support Spanish and English locales in the Flutter client.",
            "i18n: Spanish English",
        ),
        (
            "Add crash reporting via Sentry for the Flutter builds.",
            "observability: Sentry Flutter",
        ),
        (
            "Allow technicians to capture signatures on completed jobs.",
            "feature: signature capture",
        ),
        (
            "Add battery-efficient background location only while a job is active.",
            "feature: background location scoped",
        ),
        (
            "Freeze Atlas mobile scope around Flutter technician workflows for v1.",
            "scope: Atlas Flutter v1 freeze",
        ),
    ]
    return [
        ScenarioTurn(turn_index=i + 1, transcript=_pad(t), fact_to_remember=f)
        for i, (t, f) in enumerate(specs)
    ]


def _contradiction_turns() -> list[ScenarioTurn]:
    specs: list[tuple[str, str]] = [
        (
            "Project Helios is an internal analytics portal for finance. "
            "Stack is Next.js, Postgres, and a small Python ETL.",
            "project name: Helios",
        ),
        (
            "Helios must show monthly P&L and cashflow dashboards for executives.",
            "feature: P&L cashflow dashboards",
        ),
        (
            "Budget locked: 30000 EUR for the Helios v1 delivery including ETL.",
            "budget locked: 30000 EUR",
        ),
        (
            "Add SSO via Okta for all Helios users in the finance department.",
            "auth: Okta SSO",
        ),
        (
            "Include row-level security so regional controllers only see their region.",
            "security: row-level region ACL",
        ),
        (
            "Add scheduled PDF report emails every Monday morning.",
            "feature: Monday PDF reports",
        ),
        (
            "We need a data dictionary page documenting sources for each KPI.",
            "feature: data dictionary",
        ),
        (
            "Budget update: budget locked: 80000 EUR because scope grew with ETL and SSO.",
            "budget locked: 80000 EUR",
        ),
        (
            "Add anomaly alerts when spend spikes more than 20 percent week over week.",
            "feature: anomaly alerts",
        ),
        (
            "Support CSV download of every dashboard table in Helios.",
            "feature: CSV dashboard download",
        ),
        (
            "Add a staging environment that mirrors production schemas.",
            "ops: staging environment",
        ),
        (
            "Finance wants quarterly forecast scenarios on top of actuals.",
            "feature: quarterly forecast scenarios",
        ),
        (
            "Add audit trails for who changed KPI definitions in Helios.",
            "feature: KPI definition audit",
        ),
        (
            "Integrate Snowflake as a read source alongside Postgres for Helios.",
            "stack includes Snowflake",
        ),
        (
            "Add role templates for CFO, controller, and analyst personas.",
            "feature: CFO controller analyst roles",
        ),
        (
            "Support dark mode for late-night finance close work.",
            "ui: dark mode",
        ),
        (
            "Add saved views so users can pin their favourite Helios filters.",
            "feature: saved views",
        ),
        (
            "Include a runbook for monthly close operations using Helios.",
            "deliverable: monthly close runbook",
        ),
        (
            "Add Slack alerts for failed ETL jobs overnight.",
            "integration: Slack ETL alerts",
        ),
        (
            "Freeze Helios v1 after forecast scenarios; no more new modules.",
            "scope: Helios v1 freeze",
        ),
    ]
    return [
        ScenarioTurn(turn_index=i + 1, transcript=_pad(t), fact_to_remember=f)
        for i, (t, f) in enumerate(specs)
    ]


GROWING = ScenarioProfile(
    name="growing",
    description="Project accumulates coherent requirements; track whether project_name survives.",
    turns=_growing_turns(),
)

PIVOT = ScenarioProfile(
    name="pivot",
    description="Stack pivots React→Flutter at turn 5; watch technology list merge behaviour.",
    turns=_pivot_turns(),
)

CONTRADICTION = ScenarioProfile(
    name="contradiction",
    description="Budget flips 30k→80k EUR; watch which fact anchors and which survives.",
    turns=_contradiction_turns(),
)

SCENARIOS: dict[str, ScenarioProfile] = {
    GROWING.name: GROWING,
    PIVOT.name: PIVOT,
    CONTRADICTION.name: CONTRADICTION,
}


def get_scenario(name: str) -> ScenarioProfile:
    try:
        return SCENARIOS[name]
    except KeyError as exc:
        known = ", ".join(sorted(SCENARIOS))
        raise KeyError(f"Unknown scenario {name!r}. Known: {known}") from exc
