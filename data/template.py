"""
The standard onboarding template. Fixed, hardcoded, on purpose --
this is the baseline. The agent's whole job in Stage 2 is to decide
how much THIS client's plan should deviate from this list, and why.

Each step has an id, a description, and whether it's normally
client-facing. depends_on is left empty here -- dependencies are
assigned at plan-generation time based on the actual client situation,
since a deviation can introduce new dependencies the template doesn't know about.
"""

STANDARD_TEMPLATE = [
    {
        "template_step_id": "T1",
        "action": "Send kickoff confirmation email summarizing scope and next steps",
        "client_facing": True,
    },
    {
        "template_step_id": "T2",
        "action": "Send stakeholder introduction email to all client-side contacts",
        "client_facing": True,
    },
    {
        "template_step_id": "T3",
        "action": "Schedule discovery calls with client technical team",
        "client_facing": True,
    },
    {
        "template_step_id": "T4",
        "action": "Look up existing contact records in CRM/SOW for secondary stakeholders",
        "client_facing": False,
    },
    {
        "template_step_id": "T5",
        "action": "Create internal onboarding tasks for implementation team",
        "client_facing": False,
    },
    {
        "template_step_id": "T6",
        "action": "Run data quality audit on client-provided datasets",
        "client_facing": False,
    },
    {
        "template_step_id": "T7",
        "action": "Configure integration environment based on client's existing tools",
        "client_facing": False,
    },
    {
        "template_step_id": "T8",
        "action": "Send go-live readiness checklist to client for sign-off",
        "client_facing": True,
    },
]