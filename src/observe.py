"""
Stage 4 -- Observe.

After Stage 3 executes tool calls, this stage checks every successful
tool result against the active facts that share its domain. If a
result contradicts a fact, it creates a Conflict, marks that fact
"contradicted", and pushes an approval request -- it does NOT decide
how to resolve the contradiction. That decision is reserved for a
human, by design, not by limitation.

Two distinct things can halt a branch of the plan, and this module
only handles one of them:
  - A CONTRADICTION (this module): a tool result disagrees with
    something already believed. Triggered here, in Stage 4.
  - A TOOL FAILURE (handled back in Stage 3 / act.py): the tool itself
    errored or returned an ambiguous result. Already handled at the
    point of failure, before this stage even runs.
Keeping these separate matters: a contradiction is a fact problem
(something we believed turned out to be wrong), a tool failure is an
execution problem (something we tried to do didn't work). Conflating
them would blur exactly the distinction an FDE needs to reason about:
"is this a data problem or a systems problem."

Domain-scoped checking: a calendar result only gets checked against
scheduling-domain facts, not against every active fact. This keeps the
number of LLM calls proportional to what's actually relevant, not to
the total size of the fact list.
"""

from __future__ import annotations
import json
import os
import sys

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from onboarding_state import OnboardingState, Conflict, ApprovalRequest, ToolLogEntry

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


CONFLICT_CHECK_SYSTEM_PROMPT = """You check whether a tool result contradicts a previously believed fact about a client.

You will be given one fact (a claim believed about the client, with its own confidence level) and one tool result. Decide: does the tool result genuinely contradict, undermine, or cast doubt on the fact? Only say yes if there's a real, specific tension -- not a vague or weak association.

Examples:
- Fact: "client says data is clean." Tool result: data audit shows 40% missing fields. -> CONTRADICTS. The audit directly disputes the clean-data claim.
- Fact: "client uses Salesforce." Tool result: calendar shows person X is available Tuesday. -> NO CONTRADICTION. Unrelated.
- Fact: "client's data is clean" (already weakened to inferred/at-risk from a prior step). Tool result: audit shows minor issues. -> use judgment: if the fact is already flagged as uncertain, a confirming result is not a new contradiction.

Output ONLY a JSON object: {"contradicts": true|false, "reason": "one sentence, specific, citing what in the tool result conflicts with what in the fact"}
No preamble, no markdown fences.
"""


def _next_conflict_id(state: OnboardingState) -> str:
    return f"C{len(state.conflicts) + 1}"


def _next_approval_id(state: OnboardingState) -> str:
    return f"A{len(state.approval_queue) + 1}"


def _relevant_domain_for_tool(tool_log_entry: ToolLogEntry) -> str | None:
    """Maps a tool log entry to the fact domain it could plausibly
    contradict. Simple keyword routing -- mirrors the same style of
    routing Stage 3 uses, so the whole pipeline stays consistent about
    where determinism lives versus where LLM judgment lives."""
    tool_lower = tool_log_entry.tool.lower()
    if "data quality audit" in tool_lower or "data audit" in tool_lower:
        return "data_quality"
    if "calendar" in tool_lower or "schedule" in tool_lower or "discovery call" in tool_lower:
        return "scheduling"
    if "look up" in tool_lower or "contact record" in tool_lower:
        return "contacts"
    if "configure integration" in tool_lower:
        return "scope"
    return None


def _check_fact_against_result(fact, tool_result: dict, client: "OpenAI | None" = None) -> dict:
    if client is None:
        client = OpenAI()

    user_content = (
        f"FACT: {fact.claim} (confidence: {fact.confidence}, status: {fact.status})\n"
        f"TOOL RESULT: {json.dumps(tool_result)}"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": CONFLICT_CHECK_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines)

    return json.loads(raw)


def run_stage4(state: OnboardingState, client: "OpenAI | None" = None) -> OnboardingState:
    """Checks every successful tool_log entry that hasn't been checked yet
    against relevant active facts. Mutates and returns state.

    Only inspects tool_log entries with status "ok" -- a tool_failure was
    already routed to the approval queue back in Stage 3, before this
    stage ever sees it, so there's nothing for this stage to do with it.

    Tracks which tool_log entries it has already checked (by call_id) so
    re-running this stage after new tool calls happen doesn't re-raise
    conflicts that were already surfaced and are sitting in the queue
    or already resolved.
    """
    already_checked_call_ids = {
        c.tool_call_id for c in state.conflicts if c.tool_call_id
    }

    for entry in state.tool_log:
        if entry.status != "ok":
            continue
        if entry.call_id in already_checked_call_ids:
            continue

        domain = _relevant_domain_for_tool(entry)
        if domain is None:
            continue

        relevant_facts = state.active_facts(domain=domain)
        if not relevant_facts:
            continue

        for fact in relevant_facts:
            check = _check_fact_against_result(fact, entry.output, client=client)
            if check.get("contradicts"):
                fact.status = "contradicted"

                conflict_id = _next_conflict_id(state)
                state.conflicts.append(
                    Conflict(
                        id=conflict_id,
                        fact_id=fact.id,
                        trigger="tool_result",
                        tool_call_id=entry.call_id,
                        description=check.get("reason", "Contradiction detected."),
                        status="open",
                    )
                )

                state.approval_queue.append(
                    ApprovalRequest(
                        id=_next_approval_id(state),
                        type="conflict_resolution",
                        ref_id=conflict_id,
                        proposed_action=(
                            f"Fact '{fact.claim}' (cited from {fact.source_span}) is "
                            f"contradicted by the result of '{entry.tool}'. How should this be resolved?"
                        ),
                        justification=check.get("reason", ""),
                    )
                )
                # do not auto-resolve, do not break -- if multiple facts in this
                # domain are each individually contradicted by the same result,
                # each gets its own conflict, surfaced independently.

    return state


if __name__ == "__main__":
    import argparse
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
    from template import STANDARD_TEMPLATE
    from extract import extract_facts
    from plan import build_plan
    from act import run_stage3

    parser = argparse.ArgumentParser()
    parser.add_argument("transcript_path")
    parser.add_argument("--client-id", required=True)
    args = parser.parse_args()

    with open(args.transcript_path) as f:
        raw = f.read()

    print("--- Extracting facts ---")
    facts = extract_facts(raw)

    print("--- Building plan ---")
    plan = build_plan(facts, STANDARD_TEMPLATE)

    state = OnboardingState(client_id=args.client_id, transcript_raw=raw, facts=facts, plan=plan)

    print("--- Running Stage 3 (Act) ---")
    run_stage3(state)

    print("--- Running Stage 4 (Observe) ---")
    run_stage4(state)

    print()
    print(state.pretty_print())

    out_path = os.path.join(os.path.dirname(__file__), "..", "data", f"state_{args.client_id}.json")
    state.save(out_path)
    print(f"\nState saved to {out_path}")