"""
Mock tools.

Four tools, backed by static JSON seeded data -- not live integrations.
This is a deliberate scope decision: the point of the demo is the
reasoning and tool-chaining in the orchestrator (Stage 3/4), not
building a real CRM/calendar/email integration. Each tool returns a
dict result and a status, exactly like a real tool call would, so the
orchestrator's handling logic doesn't know or care that these are mocked.

A fifth function, draft_email, doesn't hit any "tool" at all -- it's
an LLM call that drafts the actual content for a client-facing step,
so a human approving it has something real to read, not just a label.
"""

from __future__ import annotations
import json
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _load_json(filename: str) -> dict:
    with open(os.path.join(_DATA_DIR, filename)) as f:
        return json.load(f)


def check_calendar_availability(client_id: str, person: str) -> tuple[dict, str]:
    """Returns (result, status). status is one of 'ok', 'error', 'ambiguous'."""
    calendar = _load_json("mock_calendar.json")
    client_cal = calendar.get(client_id, {})
    person_cal = client_cal.get(person)
    if person_cal is None:
        return (
            {"error": f"No calendar record found for '{person}' under client '{client_id}'"},
            "error",
        )
    return (person_cal, "ok")


def lookup_contact(client_id: str, name: str) -> tuple[dict, str]:
    """Looks across both SOW and CRM contact records. Returns ambiguous
    status if the same name appears with conflicting details in more than
    one place -- this is one of the two halt triggers in Stage 4, distinct
    from a fact contradiction."""
    contacts = _load_json("mock_contacts.json")
    client_contacts = contacts.get(client_id, {})
    sow = client_contacts.get("sow_contacts", [])
    crm = client_contacts.get("crm_contacts", [])

    matches = [c for c in sow + crm if c["name"].lower() == name.lower()]

    if not matches:
        return ({"error": f"No contact record found for '{name}'"}, "error")
    if len(matches) > 1 and len({m.get("email") for m in matches}) > 1:
        return (
            {"matches": matches, "note": "Multiple conflicting records found for this name"},
            "ambiguous",
        )
    return (matches[0], "ok")


def run_data_audit(client_id: str) -> tuple[dict, str]:
    audit = _load_json("mock_data_audit.json")
    result = audit.get(client_id)
    if result is None:
        return ({"error": f"No audit data available for client '{client_id}'"}, "error")
    return (result, "ok")


def create_internal_task(client_id: str, description: str) -> tuple[dict, str]:
    """Purely logs -- there's no real task system behind this. Always succeeds."""
    return (
        {
            "task_created": True,
            "client_id": client_id,
            "description": description,
            "created_at": datetime.utcnow().isoformat(),
        },
        "ok",
    )


DRAFT_EMAIL_SYSTEM_PROMPT = """You draft short, professional onboarding emails for a client onboarding rep named Vasu.

You will be given the plan step being executed and the facts that justify it. Write the actual email: a subject line and a body. Keep it concise, warm but professional, and specific to the facts given -- don't write a generic template. Don't invent details not supported by the facts you're given.

Output ONLY a JSON object: {"subject": "...", "body": "..."}, no preamble, no markdown fences.
"""


def draft_email(step_action: str, justification_text: str, fact_claims: list[str], client: "OpenAI | None" = None) -> dict:
    """Drafts the actual content for a client-facing step, so a human
    approving it sees something real, not a placeholder."""
    if client is None:
        client = OpenAI()

    user_content = (
        f"PLAN STEP: {step_action}\n"
        f"JUSTIFICATION: {justification_text}\n"
        f"RELEVANT FACTS:\n" + "\n".join(f"- {c}" for c in fact_claims)
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": DRAFT_EMAIL_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines)

    return json.loads(raw)