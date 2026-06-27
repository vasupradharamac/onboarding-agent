"""
Stage 2 -- Diff against template.

Takes the facts extracted in Stage 1 plus the fixed standard template,
and produces a plan: each template step is kept, reordered, or removed,
and new steps can be added -- every change justified with a reference
to specific fact_ids (which already carry their transcript line
citations from Stage 1).

This same function IS the replan step (Stage 5). When a conflict gets
resolved by a human, we don't run separate "replanning" logic -- we
call this function again with the updated fact list (the resolved/
superseded fact now reflects the human's decision). Planning and
replanning are the same operation; only the input facts differ.

Design note: this stage deliberately does NOT decide whether to
auto-resolve a contradiction. It only ever sees "active" facts, by
construction (see build_plan's facts param) -- so a fact marked
"contradicted" earlier in Stage 4 simply won't influence this stage
until a human resolves it and the state is updated. That separation
is what keeps "don't auto-resolve" an enforced property of the
pipeline rather than a hope about prompting.
"""

from __future__ import annotations
import json
import os
import sys

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from onboarding_state import Fact, PlanStep, Justification

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


PLAN_SYSTEM_PROMPT = """You are deciding how to adapt a standard client onboarding plan for a specific client, based on facts extracted from their kickoff call.

You will be given:
1. STANDARD TEMPLATE: a fixed list of steps every onboarding normally includes.
2. FACTS: specific things known about this client, each with an id and a source citation.

Your job: decide, for each template step, whether to KEEP it as-is, MODIFY it (same step, but the action wording changes to reflect something client-specific), or REMOVE it (only if a fact makes it clearly inapplicable -- this should be rare). You may also ADD new steps not in the template, only when a fact implies a real need the template doesn't cover.

Rules:
- EVERY step in your output -- whether kept, modified, added, or reordered -- must include a justification. For KEPT steps with no client-specific reasoning, justification can be a short neutral note like "Standard step, no client-specific deviation" with an empty fact_ids list. For MODIFIED, ADDED, or REORDERED steps, justification MUST reference the specific fact_id(s) that drove the change -- never justify a deviation without citing which fact caused it.
- Do not invent facts. Only reference fact_ids that were actually given to you.
- template_origin is "template" for steps that are the same action as their original template step (even if reworded slightly to add client specifics). template_origin is "deviation" for any ADDED step, or any step whose substance (not just wording) changed because of a fact -- e.g. adding a sign-off requirement, or changing WHO an email goes to.
- client_facing should be true for anything the client would see directly: emails sent to them, calls scheduled with them, timeline commitments communicated to them, any check-in or deliverable they receive. It should be false for internal-only actions: internal task creation, internal data audits, internal lookups.
- Be conservative about commitments. If a fact describes something the CLIENT hopes for, wants, or asks for (e.g. a target date, a deadline), do not turn that into a step that COMMUNICATES or COMMITS to it as if it were already agreed. Reflect it as something to track or plan around internally, not as a promise made back to the client, unless a fact explicitly states the commitment was already agreed to on the call.
- depends_on should list the step_id(s) (using YOUR new step_ids, not the template's) that must complete before this step can run. Be conservative and specific: a step depends on another ONLY if it genuinely cannot happen first -- e.g. "configure integration" depends on knowing which tool the client uses, but "create internal onboarding tasks" does NOT need to wait for "send kickoff email" to finish; they can happen independently. Do NOT default to making every step depend on the step before it in your list -- a plan where everything is chained in one straight line is almost always wrong. Most steps should have an empty depends_on list. Ask yourself for each step: "could this realistically start right now, in parallel with everything else?" If yes, leave depends_on empty.- Preserve a sensible order: dependencies should always come before the steps that depend on them in your output list.
- Output ONLY a JSON object of the shape {"plan": [...]}, no preamble, no markdown fences. Each element:
  {"action": "...", "template_origin": "template"|"deviation", "justification_text": "...", "justification_fact_ids": ["F1", ...], "client_facing": true|false, "depends_on": ["S1", ...]}
  (depends_on refers to step_ids you will assign in order: S1, S2, S3... matching the order of this output array)
"""


def build_plan(
    facts: list[Fact],
    template: list[dict],
    client: "OpenAI | None" = None,
) -> list[PlanStep]:
    """Run Stage 2 (and, when called again later with updated facts, Stage 5's
    replan) on a set of active facts and the standard template.
    Returns a list of PlanStep objects with step_ids assigned (S1, S2, ...).

    Only pass ACTIVE facts in -- contradicted facts should be excluded
    by the caller until a human resolves them. This function does not
    filter by status itself, to keep it a pure function of whatever
    facts it's handed; the caller (the orchestrator) is responsible for
    deciding which facts are "currently in play."
    """
    facts_payload = [
        {
            "id": f.id,
            "claim": f.claim,
            "domain": f.domain,
            "confidence": f.confidence,
            "source_span": f.source_span,
        }
        for f in facts
    ]

    user_content = (
        "STANDARD TEMPLATE:\n"
        + json.dumps(template, indent=2)
        + "\n\nFACTS:\n"
        + json.dumps(facts_payload, indent=2)
    )

    if client is None:
        client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw_output = response.choices[0].message.content.strip()
    raw_output = _strip_markdown_fence(raw_output)

    try:
        parsed = json.loads(raw_output)
        items = parsed["plan"] if isinstance(parsed, dict) else parsed
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(
            f"Plan LLM did not return valid JSON in the expected shape. Raw output:\n{raw_output}"
        ) from e

    valid_fact_ids = {f.id for f in facts}

    plan_steps = []
    step_ids = [f"S{i}" for i in range(1, len(items) + 1)]

    for i, item in enumerate(items):
        step_id = step_ids[i]
        cited_ids = item.get("justification_fact_ids", [])
        # defensive: drop any fact_id the model hallucinated that wasn't actually given to it
        cited_ids = [fid for fid in cited_ids if fid in valid_fact_ids]

        plan_steps.append(
            PlanStep(
                step_id=step_id,
                action=item["action"],
                template_origin=item.get("template_origin", "template"),
                justification=Justification(
                    text=item.get("justification_text", ""),
                    fact_ids=cited_ids,
                ),
                status="pending",
                client_facing=item.get("client_facing", False),
                depends_on=item.get("depends_on", []),
            )
        )

    return plan_steps


def _strip_markdown_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


if __name__ == "__main__":
    import argparse
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
    from template import STANDARD_TEMPLATE
    from extract import extract_facts

    parser = argparse.ArgumentParser()
    parser.add_argument("transcript_path")
    args = parser.parse_args()

    with open(args.transcript_path) as f:
        raw = f.read()

    print("--- Extracting facts ---")
    facts = extract_facts(raw)
    for f in facts:
        print(f"[{f.id}] ({f.confidence}, {f.domain}) {f.claim}  <- {f.source_span}")

    print("\n--- Building plan ---")
    plan = build_plan(facts, STANDARD_TEMPLATE)
    for s in plan:
        tag = "DEVIATION" if s.template_origin == "deviation" else "template"
        cf = " [CLIENT-FACING]" if s.client_facing else ""
        print(f"[{s.step_id}] ({tag}){cf} {s.action}")
        print(f"    why: {s.justification.text}  <- facts {s.justification.fact_ids}")
        if s.depends_on:
            print(f"    depends on: {s.depends_on}")