"""
Stage 3 -- Act.

Walks state.plan in order. For each step:
  - If client_facing: draft the actual content (email/message), push to
    approval_queue, mark step "blocked". Do NOT execute. Any step that
    depends on this one also gets marked "blocked" and skipped this pass --
    independent steps keep going. This is the explicit answer to "does the
    whole pipeline freeze while waiting for a human": no, only the
    dependent branch does.
  - Else: execute the matching mock tool, log the result, mark "executed".

This stage does NOT do conflict detection -- that's Stage 4, which runs
on the tool_log entries this stage produces. Stage 3's only job is
executing what's safe to execute and stopping what isn't.

Re-runnable: calling run_stage3 again after some approvals have been
resolved will pick up exactly where it left off, since it only acts on
steps still in "pending" status and respects current "blocked" status
on dependencies.
"""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from onboarding_state import OnboardingState, ToolLogEntry, ApprovalRequest
import tools


def _next_call_id(state: OnboardingState) -> str:
    return f"T{len(state.tool_log) + 1}"


def _next_approval_id(state: OnboardingState) -> str:
    return f"A{len(state.approval_queue) + 1}"


def _is_blocked_by_dependency(state: OnboardingState, step) -> bool:
    """A step is blocked if any step it depends on is not yet executed."""
    for dep_id in step.depends_on:
        dep = state.get_plan_step(dep_id)
        if dep is None:
            continue  # dangling dependency reference, don't let it wedge the whole plan
        if dep.status != "executed":
            return True
    return False


def _extract_candidate_names(state: OnboardingState) -> list[str]:
    """Pull plausible person names out of contacts/stakeholders facts.
    Simple heuristic for the demo: look for capitalized two-word sequences
    in those facts' claims. Good enough for seeded demo data; a production
    version would have named-entity fields on the Fact itself."""
    import re
    names = set()
    for f in state.facts:
        if f.domain in ("contacts", "stakeholders"):
            found = re.findall(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", f.claim)
            names.update(found)
    return sorted(names) or ["Unknown Contact"]


def _execute_tool_step(state: OnboardingState, step) -> None:
    """Routes a non-client-facing step to the matching mock tool based on
    keywords in the step's action text. This is intentionally simple
    keyword routing, not an LLM call -- Stage 3 is meant to be the
    deterministic, auditable part of the pipeline. Every tool call's
    result is logged regardless of outcome."""
    action_lower = step.action.lower()
    call_id = _next_call_id(state)

    if "calendar" in action_lower or "schedule" in action_lower or "discovery call" in action_lower:
        result, status = (
            {"note": "Calendar check step -- see contact-specific lookups for detail"},
            "ok",
        )

    elif "look up" in action_lower or "contact record" in action_lower:
        names_to_check = _extract_candidate_names(state)
        all_results = {}
        overall_status = "ok"
        for name in names_to_check:
            res, st = tools.lookup_contact(state.client_id, name)
            all_results[name] = {"result": res, "status": st}
            if st != "ok":
                overall_status = st
        result, status = all_results, overall_status

    elif "data quality audit" in action_lower or "data audit" in action_lower:
        result, status = tools.run_data_audit(state.client_id)

    elif "internal" in action_lower and "task" in action_lower:
        result, status = tools.create_internal_task(state.client_id, step.action)

    elif "configure integration" in action_lower:
        result, status = (
            {"note": "Integration environment configuration logged (no live system connected)"},
            "ok",
        )

    else:
        result, status = ({"note": "No matching mock tool for this step; logged as a no-op."}, "ok")

    state.tool_log.append(
        ToolLogEntry(
            call_id=call_id,
            tool=step.action,
            input={"client_id": state.client_id},
            output=result,
            status=status,
        )
    )
    step.tool_call_id = call_id

    if status == "ok":
        step.status = "executed"
    else:
        step.status = "blocked"
        state.approval_queue.append(
            ApprovalRequest(
                id=_next_approval_id(state),
                type="tool_failure",
                ref_id=step.step_id,
                proposed_action=f"Tool call for step '{step.action}' returned status '{status}'. How should I proceed?",
                justification=f"Tool result: {result}",
            )
        )


def _draft_and_queue_client_facing_step(state: OnboardingState, step) -> None:
    fact_claims = [
        state.get_fact(fid).claim for fid in step.justification.fact_ids if state.get_fact(fid)
    ]
    draft = tools.draft_email(step.action, step.justification.text, fact_claims)

    drafted_content = f"Subject: {draft['subject']}\n\n{draft['body']}"

    step.status = "blocked"
    state.approval_queue.append(
        ApprovalRequest(
            id=_next_approval_id(state),
            type="client_facing_action",
            ref_id=step.step_id,
            proposed_action=step.action,
            justification=step.justification.text,
            drafted_content=drafted_content,
        )
    )


def run_stage3(state: OnboardingState) -> OnboardingState:
    """Mutates and returns state. Loops over the plan repeatedly until a
    full pass makes no further progress -- this lets independent branches
    run to completion in one call, while branches behind a blocked
    client-facing step or tool failure correctly stop and wait. Calling
    this again later (e.g. after approvals are resolved) picks up exactly
    where it left off, since it only acts on steps still "pending"."""
    while True:
        made_progress = False
        for step in state.plan:
            if step.status != "pending":
                continue  # already executed, blocked, approved, or rejected -- skip

            if _is_blocked_by_dependency(state, step):
                continue  # leave as pending; will be picked up once dependency clears

            if step.client_facing:
                _draft_and_queue_client_facing_step(state, step)
            else:
                _execute_tool_step(state, step)
            made_progress = True

        if not made_progress:
            break

    return state


if __name__ == "__main__":
    import argparse
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
    from template import STANDARD_TEMPLATE
    from extract import extract_facts
    from plan import build_plan

    parser = argparse.ArgumentParser()
    parser.add_argument("transcript_path")
    parser.add_argument("--client-id", required=True, help="e.g. northwind-logistics or solace-health")
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

    print()
    print(state.pretty_print())

    out_path = os.path.join(os.path.dirname(__file__), "..", "data", f"state_{args.client_id}.json")
    state.save(out_path)
    print(f"\nState saved to {out_path}")