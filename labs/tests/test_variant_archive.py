"""B5 / AIN-565 — variant archive + DGM auto-keep, SHADOW + fenced.

Deterministic, no DB / no network. Proves: governance fence (B4 reuse + precedence), κ-gated
empirical auto-keep, append-only lineage archive, the offline DR-OPE gate, and the hard cage
invariant that nothing is ever live-promoted (D4 only).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from labs.dr_ope import OPEResult
from labs.variant_archive import (
    GOVERNANCE_DENY_GLOBS,
    KAPPA_GATE,
    VARIANT_ALLOWED_DOMAINS,
    ScoredVariant,
    Variant,
    VariantMetrics,
    append_receipt,
    decision_receipt,
    evaluate_variant,
    fence_variant,
    governance_hit,
    lineage,
    offline_promote_gate,
    read_archive,
    selftest,
)

NOW = datetime(2026, 6, 20, 12, 0, 0, tzinfo=UTC)

# The B4 GOVERNANCE_DENY_GLOBS, verbatim (valinor agents/aule/native_code.py, PR #264 f52bf80).
# The ported fence must be a SUPERSET — extend, never narrow.
_B4_VERBATIM = (
    "substrate/ainfera_core/governance/**",
    "agents/_killswitch.py",
    "*operator_policy*",
    "*d2_cosign*", "*cosign*", "*co_sign*",
    "*cage_runtime*", "*cage_integrity*", "*killswitch*", "*certificates*",
    "scripts/cage_integrity_check.py",
    ".github/workflows/cage-integrity.yml",
    "*eonwe*", "*broker*",
    "*secret*",
)


def _anchor() -> VariantMetrics:
    return VariantMetrics(n=140, successes=112, cost_usd=2.80)  # TSR 0.800, CPST $0.025


def _good_metrics() -> VariantMetrics:
    return VariantMetrics(n=120, successes=104, cost_usd=2.16)  # TSR 0.867 (+6.7pp), cheaper


# ── DoD: the composed drill ───────────────────────────────────────────────────────────


def test_selftest_meets_dod(tmp_path):
    receipts, failures = selftest(NOW, tmp_path / "archive.jsonl")
    assert failures == [], failures
    assert len(receipts) == 3
    decisions = {r["variant_id"]: r["decision"] for r in receipts}
    assert "kept" in decisions.values()
    assert "refused" in decisions.values()


# ── governance fence: B4 reuse + precedence ───────────────────────────────────────────


def test_ported_fence_is_superset_of_b4():
    for g in _B4_VERBATIM:
        assert g in GOVERNANCE_DENY_GLOBS, f"B4 glob dropped from the fence: {g}"


@pytest.mark.parametrize("target", [
    "substrate/ainfera_core/governance/cage_runtime.py",
    "agents/aule/d2_cosign_engine.py",
    "config/eonwe_broker.py",
    "agents/_killswitch.py",
    "scripts/cage_integrity_check.py",
    "lib/my_secret_loader.py",
    "governance/certificates.py",
    "routing/enforce_flip.py",     # B5 extension
    "policy/l1_caps.py",           # B5 extension
    "agents/arming_switch.py",     # B5 extension
])
def test_governance_targets_are_denied(target):
    assert governance_hit(target) is not None


def test_governance_precedence_over_allowed_domain():
    # domain is allowed ('tools') but the target is the co-sign engine → governance, not ok.
    v = Variant("g1", "c::x", "tools",
                "substrate/ainfera_core/governance/d2_cosign_engine.py", None)
    fv = fence_variant(v)
    assert not fv.ok and fv.kind == "governance"


def test_cross_domain_refused():
    v = Variant("x1", "c::x", "infra", "infra/deploy.tf", None)
    fv = fence_variant(v)
    assert not fv.ok and fv.kind == "cross_domain"


@pytest.mark.parametrize("domain,target", [
    ("tools", "tools/web_search.py"),
    ("prompts", "prompts/reasoning/system.md"),
    ("routing_policy", "routing/policy/reasoning.yaml"),
])
def test_capability_domains_allowed(domain, target):
    assert fence_variant(Variant("ok", "c::x", domain, target, None)).ok


def test_allowed_domains_are_exactly_capabilities():
    assert VARIANT_ALLOWED_DOMAINS == frozenset({"tools", "prompts", "routing_policy"})


# ── empirical auto-keep gate (DGM validation) ─────────────────────────────────────────


def test_kept_requires_kappa_gate():
    v = Variant("k1", "extraction::m", "prompts", "prompts/extraction.md", None)
    # below the κ gate → held even with great metrics
    d = evaluate_variant(ScoredVariant(v, _good_metrics(), _anchor(), KAPPA_GATE - 0.01), now=NOW)
    assert d.decision == "held" and d.gates["kappa_ge_gate"] is False
    # at/above the gate → kept
    d2 = evaluate_variant(ScoredVariant(v, _good_metrics(), _anchor(), KAPPA_GATE), now=NOW)
    assert d2.decision == "kept" and d2.gates["kappa_ge_gate"] is True


def test_kept_requires_sufficient_n():
    v = Variant("n1", "extraction::m", "prompts", "prompts/extraction.md", None)
    tiny = VariantMetrics(n=10, successes=10, cost_usd=0.1)  # great TSR but n<30
    d = evaluate_variant(ScoredVariant(v, tiny, _anchor(), 0.9), now=NOW)
    assert d.decision == "held" and d.gates["sufficient_n"] is False


def test_kept_requires_measurable_tsr_improvement():
    v = Variant("t1", "extraction::m", "prompts", "prompts/extraction.md", None)
    flat = VariantMetrics(n=100, successes=80, cost_usd=1.0)  # TSR 0.800 == anchor → +0.0pp
    d = evaluate_variant(ScoredVariant(v, flat, _anchor(), 0.9), now=NOW)
    assert d.decision == "held" and d.gates["improves_tsr"] is False


def test_kept_requires_cpst_not_regressed():
    v = Variant("c1", "extraction::m", "prompts", "prompts/extraction.md", None)
    pricey = VariantMetrics(n=120, successes=110, cost_usd=10.0)  # TSR up but far pricier
    d = evaluate_variant(ScoredVariant(v, pricey, _anchor(), 0.9), now=NOW)
    assert d.decision == "held" and d.gates["cpst_not_regressed"] is False


# ── the cage: never a live promote ────────────────────────────────────────────────────


def test_no_decision_ever_live_promotes():
    v = Variant("p1", "extraction::m", "prompts", "prompts/extraction.md", None)
    d = evaluate_variant(ScoredVariant(v, _good_metrics(), _anchor(), 0.9), now=NOW)
    assert d.live_promoted is False and d.requires_d4_to_promote is True
    rec = decision_receipt(d, now=NOW)
    assert rec["live_promoted"] == "0" and rec["excluded_from_moat"] == "1" and rec["shadow"] == "1"


def test_assert_shadow_tripwire_blocks_live_promote(tmp_path):
    v = Variant("p2", "extraction::m", "prompts", "prompts/extraction.md", None)
    rec = decision_receipt(evaluate_variant(ScoredVariant(v, _good_metrics(), _anchor(), 0.9), now=NOW), now=NOW)
    rec["live_promoted"] = "1"  # forge a live promote
    with pytest.raises(AssertionError):
        append_receipt(rec, tmp_path / "a.jsonl")


def test_offline_gate_is_precondition_not_a_promote():
    healthy = OPEResult(v_dr=0.85, v_logged=0.80, lift=0.05, ci_low=0.01, ci_high=0.09,
                        n=60, ess=40.0, mean_weight=1.0)
    assert offline_promote_gate(healthy) is True
    neg = OPEResult(0.80, 0.80, 0.0, -0.02, 0.02, 60, 40.0, 1.0)
    assert offline_promote_gate(neg) is False
    assert offline_promote_gate(None) is False
    # eligible OFFLINE, but the decision still does not live-promote (D4)
    v = Variant("o1", "extraction::m", "prompts", "prompts/extraction.md", None)
    d = evaluate_variant(ScoredVariant(v, _good_metrics(), _anchor(), 0.9, healthy), now=NOW)
    assert d.offline_promote_eligible is True and d.live_promoted is False


# ── append-only archive + lineage ─────────────────────────────────────────────────────


def test_archive_append_only_and_lineage(tmp_path):
    path = tmp_path / "archive.jsonl"
    root = Variant("v1", "c::x", "prompts", "prompts/a.md", None)
    child = Variant("v2", "c::x", "prompts", "prompts/b.md", "v1")
    grand = Variant("v3", "c::x", "prompts", "prompts/c.md", "v2")
    for var in (root, child, grand):
        d = evaluate_variant(ScoredVariant(var, _good_metrics(), _anchor(), 0.9), now=NOW)
        append_receipt(decision_receipt(d, now=NOW), path)
    archive = read_archive(path)
    assert len(archive) == 3
    assert lineage(archive, "v3") == ["v3", "v2", "v1"]
    assert lineage(archive, "v1") == ["v1"]


def test_metrics_cpst_tsr():
    m = VariantMetrics(n=100, successes=80, cost_usd=2.0)
    assert m.tsr == pytest.approx(0.80)
    assert m.cpst == pytest.approx(0.025)
    assert VariantMetrics(n=10, successes=0, cost_usd=1.0).cpst is None
