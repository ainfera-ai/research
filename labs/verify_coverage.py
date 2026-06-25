"""AIN-624 · κ-coverage screening harness — SCREENING-ONLY, never a promotion gate.

The verify harness (``labs.verify_harness``) turns each eligible row into a Tier-A
reward ``∈ {0, 1, None}``. This module sizes the *reach* of that screen on a body
of traffic — the **κ-coverage** question the anchor needs answered before anyone
trusts it:

  Of the rows the screen *could* substantively grade with no gold answer (the
  INTRINSIC-eligible population — code parse, schema-valid, tool well-formed),
  what fraction did it actually grade rather than defer? And of the rows it
  graded, how many were execution-verifiable FAILURES (reward == 0)?

Coverage is a measurement, not a verdict. The informative anchor (Gwet AC1 vs the
Council, in ``api scripts/bulk_judge_worker.py``) only becomes trustworthy once a
real slice of live failures flows through it; this harness reports how much of the
gradable traffic the screen actually reaches so a low/empty anchor can be told
apart from a screen that simply never fired.

SCREENING-ONLY invariant (the load-bearing rule)
-------------------------------------------------
This module deliberately exposes **no** ``promotion_hold`` / ``gate`` / ``kappa``
hold of any kind, and imports nothing from the promotion path. It reads verifier
output and emits evidence. Promotion stays gated by ``v_anchor_health.kappa_valid``
(AIN-547 Gwet-AC1 gate) and is the founder's D4 tap — never a function of this
coverage number. The unit tests pin that absence.

Live traffic is the tap
-----------------------
``compute_coverage`` is pure (stdlib + the harness). It runs offline on fixtures
for validation; the *informative* number needs live rows — feed it the same
inferences-joined corpus that ``labs.reward_writer.VERIFY_REWARD_SELECT_SQL``
selects (``{task_type, request_payload, response_payload, expected?}``). With no
failures in the corpus, ``fail_rate`` is honestly ``None`` (0/0), not 0.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from labs.task_verifiability import verifiability_of
from labs.verify_harness import VerifySample, verify

# This harness measures; it never promotes. Asserted by test_verify_coverage.
SCREENING_ONLY = True


@dataclass(frozen=True)
class FamilyCoverage:
    """Per-verifier-family reach over the rows dispatched to it."""

    family: str
    tier: str  # 'verifiable' | 'partial' | 'subjective'
    intrinsic: bool  # family has a no-gold INTRINSIC check (gradable on live rows)
    n_eligible: int  # rows dispatched to this family
    n_graded: int  # verify() returned a reward (not deferred)
    n_deferred: int  # verify() deferred (reward is None → Council)
    n_fail: int  # graded rows scoring 0.0 — execution-verifiable failures caught
    n_pass: int  # graded rows scoring 1.0

    @property
    def coverage(self) -> float | None:
        """Graded fraction of rows reaching this family; None if none eligible."""
        return None if self.n_eligible == 0 else self.n_graded / self.n_eligible

    @property
    def fail_rate(self) -> float | None:
        """Failure fraction among graded rows; None (not 0) when nothing graded."""
        return None if self.n_graded == 0 else self.n_fail / self.n_graded


@dataclass(frozen=True)
class CoverageReport:
    """Screening evidence over a body of traffic. Carries NO promotion verdict."""

    families: dict[str, FamilyCoverage]
    n_rows: int
    n_intrinsic_eligible: int  # rows in an intrinsic-gradable family (the denom)
    n_intrinsic_graded: int  # of those, how many the screen actually graded
    n_graded_total: int  # graded rows across all families (incl. reference w/ gold)
    n_fail_total: int  # execution-verifiable failures caught across all families

    @property
    def intrinsic_coverage(self) -> float | None:
        """The headline κ-coverage: graded / intrinsic-eligible. None if denom 0."""
        if self.n_intrinsic_eligible == 0:
            return None
        return self.n_intrinsic_graded / self.n_intrinsic_eligible

    @property
    def fail_rate_total(self) -> float | None:
        """Caught-failure fraction over all graded rows; None when nothing graded."""
        return None if self.n_graded_total == 0 else self.n_fail_total / self.n_graded_total


def _sample_of(row: Mapping[str, Any]) -> VerifySample:
    return VerifySample(
        task_type=row.get("task_type"),
        request_payload=row.get("request_payload"),
        response_payload=row.get("response_payload"),
        expected=row.get("expected"),
    )


def compute_coverage(rows: Iterable[Mapping[str, Any]]) -> CoverageReport:
    """Screen a corpus and report κ-coverage per family + overall.

    ``rows``: ``{task_type, request_payload, response_payload, expected?}`` — the
    inferences-joined corpus (offline fixtures for tests; live traffic for the
    informative number). Pure: one ``verify()`` call per row, no I/O, no promotion.
    """
    # family -> mutable tallies
    tally: dict[str, dict[str, Any]] = {}
    n_rows = 0
    n_intrinsic_eligible = 0
    n_intrinsic_graded = 0
    n_graded_total = 0
    n_fail_total = 0

    for row in rows:
        n_rows += 1
        sample = _sample_of(row)
        tv = verifiability_of(sample.task_type)
        result = verify(sample)
        # Bucket by the dispatched verifier family (the harness's own label).
        fam = result.verifier
        slot = tally.setdefault(
            fam,
            {
                "tier": tv.tier.value,
                "intrinsic": tv.intrinsic,
                "n_eligible": 0,
                "n_graded": 0,
                "n_deferred": 0,
                "n_fail": 0,
                "n_pass": 0,
            },
        )
        slot["n_eligible"] += 1

        graded = result.reward is not None
        if graded:
            slot["n_graded"] += 1
            n_graded_total += 1
            if result.reward == 0.0:
                slot["n_fail"] += 1
                n_fail_total += 1
            else:
                slot["n_pass"] += 1
        else:
            slot["n_deferred"] += 1

        # Headline coverage denominator = rows with a no-gold intrinsic anchor.
        if tv.intrinsic:
            n_intrinsic_eligible += 1
            if graded:
                n_intrinsic_graded += 1

    families = {
        fam: FamilyCoverage(
            family=fam,
            tier=slot["tier"],
            intrinsic=slot["intrinsic"],
            n_eligible=slot["n_eligible"],
            n_graded=slot["n_graded"],
            n_deferred=slot["n_deferred"],
            n_fail=slot["n_fail"],
            n_pass=slot["n_pass"],
        )
        for fam, slot in tally.items()
    }
    return CoverageReport(
        families=families,
        n_rows=n_rows,
        n_intrinsic_eligible=n_intrinsic_eligible,
        n_intrinsic_graded=n_intrinsic_graded,
        n_graded_total=n_graded_total,
        n_fail_total=n_fail_total,
    )


def _pct(x: float | None) -> str:
    return "  n/a" if x is None else f"{100.0 * x:5.1f}%"


def format_report(report: CoverageReport) -> str:
    """Human-readable screening report (no verdict line — screening-only)."""
    lines = [
        "── AIN-624 κ-coverage (SCREENING-ONLY · no promotion gate) ──",
        f"  rows screened              : {report.n_rows}",
        f"  intrinsic-eligible (denom) : {report.n_intrinsic_eligible}",
        f"  intrinsic graded (reach)   : {report.n_intrinsic_graded}"
        f"  → coverage {_pct(report.intrinsic_coverage)}",
        f"  graded total / failures    : {report.n_graded_total} / {report.n_fail_total}"
        f"  → fail-rate {_pct(report.fail_rate_total)}",
        "  per family (family · tier · intrinsic · cov · graded/elig · fail/graded):",
    ]
    for fam in sorted(report.families):
        fc = report.families[fam]
        flag = "intrinsic" if fc.intrinsic else "reference"
        lines.append(
            f"    {fam:<13} {fc.tier:<10} {flag:<9} cov {_pct(fc.coverage)}"
            f"  {fc.n_graded}/{fc.n_eligible}  fail {fc.n_fail}/{fc.n_graded}"
        )
    return "\n".join(lines)
