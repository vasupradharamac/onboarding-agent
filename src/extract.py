"""
Stage 1 -- Extract.

Turns a raw kickoff transcript into a list of structured Facts, each
carrying a source_span that points back to specific transcript lines.

Design choice that matters for the interview: line numbers are injected
into the transcript IN CODE before it ever reaches the LLM. The LLM
never has to count lines itself -- it just copies the line numbers that
are already sitting right next to the text it's citing. This is what
makes "justification tied to a specific line" a property you can trust,
not a hope about the model's arithmetic.

Every fact created here keeps its source_span forever. Nothing
downstream (the plan diff, the justification text shown to a human)
invents a citation -- it only ever references a fact_id, and the
fact_id already carries its citation from this stage.
"""

from __future__ import annotations
import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from onboarding_state import Fact

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


ALLOWED_DOMAINS = ["data_quality", "scheduling", "contacts", "scope", "stakeholders", "risk"]

EXTRACT_SYSTEM_PROMPT = f"""You are extracting structured facts from a client onboarding kickoff call transcript.

The transcript has been pre-numbered, one line number per line. Every fact you extract MUST cite the exact line number(s) it came from, copied directly from the numbering already in the text. Do not count lines yourself beyond reading the number printed at the start of each line.

Rules:
- Extract only facts that would matter to planning or running the onboarding -- not small talk or filler.
- A "fact" can be a stated claim (client said it directly), or an inferred risk/implication (you noticed it, the client didn't say it outright). Mark confidence accordingly: "stated" or "inferred".
- domain must be exactly one of: {", ".join(ALLOWED_DOMAINS)}
- source_span should look like "lines 7" or "lines 4, 9" or "lines 14-15" -- whatever line numbers actually support the claim, copied from the transcript's own numbering.
- claim should be a short, specific sentence -- not a quote, your own paraphrase of what matters.
- If two lines together imply something neither says alone (e.g. a "clean data" claim plus a mention of a recent migration), extract that as a separate "inferred" fact with both line numbers cited, in addition to the stated facts.
- Output ONLY a JSON object of the shape {{"facts": [...]}}, no preamble, no markdown fences. Each element of the array:
  {{"claim": "...", "source_span": "...", "domain": "...", "confidence": "stated"|"inferred"}}
"""

def number_transcript(raw_text: str) -> str:
    """Inject line numbers into raw transcript text, one per line.
    This is the step that makes citations trustworthy -- numbering
    happens here in code, never left to the LLM to count itself.

    Strips any pre-existing leading "N: " numbering first, then
    renumbers everything fresh from 1. Doing it this way (rather than
    trying to detect "already numbered" line by line) avoids a mixed
    state where some lines in a file are pre-numbered and others
    aren't -- e.g. a header line with no number followed by dialogue
    lines that do have one.
    """
    import re

    lines = raw_text.strip().split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # strip a leading "123: " if present, regardless of what follows
        stripped = re.sub(r"^\d+:\s*", "", stripped)
        cleaned.append(stripped)

    return "\n".join(f"{i}: {line}" for i, line in enumerate(cleaned, start=1))


def extract_facts(transcript_raw: str, client: "OpenAI | None" = None) -> list[Fact]:
    """Run Stage 1 on a raw transcript. Returns a list of Fact objects
    with ids assigned (F1, F2, ...) ready to drop into OnboardingState.facts.
    """
    numbered = number_transcript(transcript_raw)

    if client is None:
        client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": numbered + "\n\nRespond with a JSON object: {\"facts\": [...]}"},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw_output = response.choices[0].message.content.strip()
    raw_output = _strip_markdown_fence(raw_output)

    try:
        parsed = json.loads(raw_output)
        items = parsed["facts"] if isinstance(parsed, dict) else parsed
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(
            f"Extraction LLM did not return valid JSON in the expected shape. Raw output:\n{raw_output}"
        ) from e

    facts = []
    for i, item in enumerate(items, start=1):
        domain = item.get("domain")
        if domain not in ALLOWED_DOMAINS:
            domain = "risk"  # safe fallback rather than crashing the whole pipeline on one bad field
        facts.append(
            Fact(
                id=f"F{i}",
                claim=item["claim"],
                source_span=item["source_span"],
                domain=domain,
                confidence=item.get("confidence", "stated"),
                status="active",
            )
        )
    return facts


def _strip_markdown_fence(text: str) -> str:
    """Models sometimes wrap JSON in ```json fences despite instructions not to.
    Strip defensively rather than trusting the prompt to be obeyed every time."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


if __name__ == "__main__":
    # quick manual run against one of the seeded transcripts
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("transcript_path")
    args = parser.parse_args()

    with open(args.transcript_path) as f:
        raw = f.read()

    facts = extract_facts(raw)
    for f in facts:
        print(f"[{f.id}] ({f.confidence}, {f.domain}) {f.claim}  <- {f.source_span}")