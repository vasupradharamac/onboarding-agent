"""
State schema for the onboarding agent.

Single source of truth for one client's onboarding run. Every pipeline
stage reads from and writes to one OnboardingState object. Nothing is
hidden inside an LLM's context window between calls -- if it matters,
it lives here, in a field, where you can print it, diff it, or show it on screen.

Design choice: plain dataclasses + dicts, persisted as JSON. No ORM,
no framework state machine. Easy to inspect, easy to explain line by
line in an interview, easy to debug at 1am the night before a demo.
"""

from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Literal


def new_id(prefix: str) -> str:
    """Short readable ids like F1, S1, C1, A1, T1 instead of UUIDs,
    so transcripts/logs/demo output stay human-readable."""
    return prefix


def make_counter():
    counts = {}
    def _next(prefix: str) -> str:
        counts[prefix] = counts.get(prefix, 0) + 1
        return f"{prefix}{counts[prefix]}"
    return _next


FactDomain = Literal[
    "data_quality", "scheduling", "contacts", "scope", "stakeholders", "risk"
]
FactStatus = Literal["active", "contradicted", "resolved", "superseded"]
PlanStepStatus = Literal["pending", "approved", "blocked", "executed", "rejected"]
PlanOrigin = Literal["template", "deviation"]
ConflictStatus = Literal["open", "escalated", "resolved"]
ApprovalType = Literal[
    "client_facing_action", "timeline_change", "conflict_resolution", "tool_failure"
]
ApprovalStatus = Literal["pending", "approved", "rejected"]


@dataclass
class Fact:
    id: str
    claim: str
    source_span: str          # e.g. "lines 14-15" -- set at extraction time, never invented later
    domain: FactDomain
    confidence: Literal["stated", "inferred"]
    status: FactStatus = "active"


@dataclass
class Justification:
    text: str
    fact_ids: list[str] = field(default_factory=list)


@dataclass
class PlanStep:
    step_id: str
    action: str
    template_origin: PlanOrigin
    justification: Justification
    status: PlanStepStatus = "pending"
    client_facing: bool = False
    depends_on: list[str] = field(default_factory=list)
    tool_call_id: Optional[str] = None


@dataclass
class Conflict:
    id: str
    fact_id: str
    trigger: Literal["tool_result", "tool_failure"]
    tool_call_id: Optional[str]
    description: str
    status: ConflictStatus = "open"
    resolution: Optional[str] = None


@dataclass
class ToolLogEntry:
    call_id: str
    tool: str
    input: dict
    output: dict
    status: Literal["ok", "error", "ambiguous"]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ApprovalRequest:
    id: str
    type: ApprovalType
    ref_id: str                      # points to a plan step_id or conflict id
    proposed_action: str
    justification: str
    status: ApprovalStatus = "pending"
    decision_note: Optional[str] = None


@dataclass
class OnboardingState:
    client_id: str
    transcript_raw: str = ""
    facts: list[Fact] = field(default_factory=list)
    plan: list[PlanStep] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    tool_log: list[ToolLogEntry] = field(default_factory=list)
    approval_queue: list[ApprovalRequest] = field(default_factory=list)

    # --- convenience accessors -------------------------------------------------

    def active_facts(self, domain: Optional[str] = None) -> list[Fact]:
        fs = [f for f in self.facts if f.status == "active"]
        if domain:
            fs = [f for f in fs if f.domain == domain]
        return fs

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        return next((f for f in self.facts if f.id == fact_id), None)

    def get_plan_step(self, step_id: str) -> Optional[PlanStep]:
        return next((s for s in self.plan if s.step_id == step_id), None)

    def pending_approvals(self) -> list[ApprovalRequest]:
        return [a for a in self.approval_queue if a.status == "pending"]

    def steps_blocked_by(self, step_id: str) -> list[PlanStep]:
        """Any step that depends on step_id, directly."""
        return [s for s in self.plan if step_id in s.depends_on]

    # --- persistence -------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "OnboardingState":
        with open(path) as f:
            raw = json.load(f)
        return cls(
            client_id=raw["client_id"],
            transcript_raw=raw.get("transcript_raw", ""),
            facts=[Fact(**f) for f in raw["facts"]],
            plan=[
                PlanStep(
                    step_id=s["step_id"],
                    action=s["action"],
                    template_origin=s["template_origin"],
                    justification=Justification(**s["justification"]),
                    status=s["status"],
                    client_facing=s["client_facing"],
                    depends_on=s["depends_on"],
                    tool_call_id=s.get("tool_call_id"),
                )
                for s in raw["plan"]
            ],
            conflicts=[Conflict(**c) for c in raw["conflicts"]],
            tool_log=[ToolLogEntry(**t) for t in raw["tool_log"]],
            approval_queue=[ApprovalRequest(**a) for a in raw["approval_queue"]],
        )

    def pretty_print(self) -> str:
        """Human-readable dump for demo / debugging -- show this on screen."""
        lines = [f"=== State for {self.client_id} ==="]
        lines.append(f"\nFACTS ({len(self.facts)}):")
        for f in self.facts:
            lines.append(f"  [{f.id}] ({f.status}, {f.domain}) {f.claim}  <- {f.source_span}")
        lines.append(f"\nPLAN ({len(self.plan)}):")
        for s in self.plan:
            tag = "DEVIATION" if s.template_origin == "deviation" else "template"
            cf = " [CLIENT-FACING]" if s.client_facing else ""
            lines.append(f"  [{s.step_id}] ({s.status}, {tag}){cf} {s.action}")
            lines.append(f"      why: {s.justification.text}  <- facts {s.justification.fact_ids}")
        lines.append(f"\nCONFLICTS ({len(self.conflicts)}):")
        for c in self.conflicts:
            lines.append(f"  [{c.id}] ({c.status}) fact={c.fact_id}: {c.description}")
        lines.append(f"\nPENDING APPROVALS ({len(self.pending_approvals())}):")
        for a in self.pending_approvals():
            lines.append(f"  [{a.id}] ({a.type}) {a.proposed_action}")
            lines.append(f"      why: {a.justification}")
        return "\n".join(lines)