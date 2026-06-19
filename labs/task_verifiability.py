"""AIN-542 Step 1 · task-type verifiability map (the Tier A anchor scope).

Tags each of the 7 canonical ``task_type``s (api ``services.section16``
``VALID_TASK_TYPES`` — kept in sync by test) with how much of its outcome is
checkable by execution, plus the default verifier family. This map *sizes the
verifiable anchor*: the fraction of real traffic where reward = ``verify()``
(Tier A) instead of an Ainfera-Council / judge soft-label (Tier B).

Three tiers
-----------
- ``verifiable`` — outcome is substantively checkable (code parses/compiles,
  structured output is schema-valid, an answer matches a reference).
- ``partial``    — checkable only structurally or only sometimes (a tool call
  can be schema-validated, but "did the *right* tool run" needs a side-effect
  observational rows may lack; a reasoning answer is checkable only when a gold
  answer exists).
- ``subjective`` — no execution anchor (open chat tone, embedding "quality").
  These fall to the Council (Tier B) and bound the residual no-anchor risk.

Honesty (v2 memo §5)
--------------------
A ``verifiable`` tag is necessary, not sufficient. On observational fleet rows
**without a gold answer**, only the INTRINSIC verifiers (parse / compile /
schema-valid) apply; REFERENCE verifiers (answer / result-set match) need the
gold the synthetic / canary stream supplies. ``intrinsic`` records that split;
``verify_harness`` enforces it. This map only says *which family to reach for*.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Verifiability(str, Enum):
    VERIFIABLE = "verifiable"
    PARTIAL = "partial"
    SUBJECTIVE = "subjective"


# Verifier families implemented (or honestly stubbed) by verify_harness.
FAMILY_CODE = "code_exec"        # intrinsic subset shipped = AST parse/compile
FAMILY_SCHEMA = "schema_match"   # JSON well-formed + structural schema conformance
FAMILY_TOOL = "tool_result"      # well-formed tool call + args schema-valid
FAMILY_ANSWER = "answer_check"   # reference: final answer vs gold (exact/numeric)
FAMILY_RETRIEVAL = "citation"    # reference: citations resolve / answer grounded
FAMILY_NONE = "none"             # no verifier — Council (Tier B)


@dataclass(frozen=True)
class TaskVerifiability:
    tier: Verifiability
    family: str
    # True => a no-gold INTRINSIC check exists, usable on live fleet rows.
    # False => only a REFERENCE check (needs gold) or no check at all.
    intrinsic: bool


# The 7 canonical task_types. MIRRORS api services.section16.VALID_TASK_TYPES —
# if that set changes, update here + the lockstep test.
CANONICAL_TASK_TYPES: frozenset[str] = frozenset(
    {"reasoning", "code", "extraction", "chat", "tool_use", "embed", "general"}
)

TASK_VERIFIABILITY: dict[str, TaskVerifiability] = {
    # hard-verifiable, intrinsic check on live traffic
    "code": TaskVerifiability(Verifiability.VERIFIABLE, FAMILY_CODE, intrinsic=True),
    "extraction": TaskVerifiability(Verifiability.VERIFIABLE, FAMILY_SCHEMA, intrinsic=True),
    # partial — structural-only intrinsic check (side-effect / right-tool needs gold)
    "tool_use": TaskVerifiability(Verifiability.PARTIAL, FAMILY_TOOL, intrinsic=True),
    # partial — answer is checkable only against a reference (no intrinsic check)
    "reasoning": TaskVerifiability(Verifiability.PARTIAL, FAMILY_ANSWER, intrinsic=False),
    # subjective — no execution anchor → Council (Tier B)
    "chat": TaskVerifiability(Verifiability.SUBJECTIVE, FAMILY_NONE, intrinsic=False),
    "embed": TaskVerifiability(Verifiability.SUBJECTIVE, FAMILY_NONE, intrinsic=False),
    "general": TaskVerifiability(Verifiability.SUBJECTIVE, FAMILY_NONE, intrinsic=False),
}

# Default for unknown / NULL task_type: subjective default-deny (assume NO anchor
# rather than over-claim verifiability on un-classifiable traffic).
_UNKNOWN = TaskVerifiability(Verifiability.SUBJECTIVE, FAMILY_NONE, intrinsic=False)


def verifiability_of(task_type: str | None) -> TaskVerifiability:
    """Tier + verifier family for a task_type. Unknown/NULL → subjective."""
    return TASK_VERIFIABILITY.get(task_type or "", _UNKNOWN)


def is_verifiable(task_type: str | None) -> bool:
    """True iff the tier is verifiable or partial (i.e. some anchor exists)."""
    return verifiability_of(task_type).tier is not Verifiability.SUBJECTIVE
