"""Scoring the system against cases it must not get wrong.

Built after the components rather than before, which the design explicitly warns
against — "build the eval harnesses before the models, otherwise 'is this better?'
gets asserted instead of answered". This closes that gap.

Two sets carry the weight: `guardrail` (where copying the user is the wrong move) and
`sycophancy` (where agreeing with the user is the wrong move).
"""

from aura.evaluation.runner import (
    EvalReport,
    ScenarioResult,
    format_report,
    run_all,
    run_scenario,
    run_set,
)
from aura.evaluation.scenarios import (
    ALL_SETS,
    GUARDRAIL_SET,
    SYCOPHANCY_SET,
    Expectation,
    Scenario,
)

__all__ = [
    "ALL_SETS",
    "GUARDRAIL_SET",
    "SYCOPHANCY_SET",
    "EvalReport",
    "Expectation",
    "Scenario",
    "ScenarioResult",
    "format_report",
    "run_all",
    "run_scenario",
    "run_set",
]
