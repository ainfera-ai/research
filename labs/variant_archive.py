"""variant_archive.py — Darwin–Gödel within a fence (B5 / AIN-565), SHADOW + propose-only.

The empirical self-improvement loop on Ainfera's *own governed substrate*, ratified
2026-06-20 (vault lock `77d89a8`, research `4b675d4`): an append-only archive of per-cell
routing/prompt/tool **variants** with parent lineage, each scored against the
verifiable-reward **anchor** + Council + Tulkas, auto-kept **only if it measurably improves
CPST/TSR** — κ-gated, sufficient-n. This is DGM's "propose → validate against reward → keep
if better," but **fenced**: a variant may evolve capability (tools / prompts / routing-policy)
and **never** a constraint (governance / cage / L1 / arming / secrets / co-sign / broker).

Cage — by construction, every invariant ratified in `77d89a8`:
  - **Self-mod is fenced.** Every variant is run through :func:`fence_variant`, which ports
    valinor's B4 ``GOVERNANCE_DENY_GLOBS`` (PR #264 `f52bf80`) verbatim and **extends** it for
    the constraint surface. Governance precedence over own-domain → a governance hit is REFUSED
    + pages the founder, regardless of the declared domain. Evolve capability, never constraints.
  - **Empirical validation is the gate.** Keep-in-SHADOW only when κ ≥ anchor gate (0.60),
    n ≥ sufficient-n (30/cell, the frozen replay-gate threshold), CPST not regressed, and TSR
    improves by ≥ the frozen +0.5pp margin — all measured vs the anchor.
  - **Auto-keep is SHADOW only.** A kept variant is an archive/dashboard fact; ``live_promoted``
    is ALWAYS 0. :func:`_assert_shadow` is a tripwire — appending a receipt with
    ``live_promoted=1`` raises, so "auto-keep ≠ live promote" is enforced in the code path.
  - **DR-OPE before any live promote.** :func:`offline_promote_gate` (composing ``labs.dr_ope``)
    reports whether a variant *would* be promote-eligible offline (ci_low(lift) > 0 + healthy
    ESS) — but **live customer-facing promotion stays a founder D4 tap**; this module never does it.

Pure (labs idiom): SQL/IO-free logic + dataclasses, deterministic, stdlib-only. The orchestrator
binds the live reward/judge signal (Step 0) and the archive file; the drill supplies them offline.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from fnmatch import fnmatch
from pathlib import Path

from labs.anchor_kappa import ANCHOR_KAPPA_GATE, Vote, cohen_kappa
from labs.dr_ope import LoggedSample, OPEResult, evaluate_policy
from labs.replay_gate import MIN_DELTA_PCT, MIN_SAMPLE_PER_CELL
from labs.verify_harness import VerifySample, verify

# ── the fence: ported from valinor B4 GOVERNANCE_DENY_GLOBS (native_code.py, PR #264 f52bf80) ──
# A variant may NEVER touch any of these, regardless of its declared domain (governance
# precedence). Ported VERBATIM, then EXTENDED for the constraint surface — extend, never narrow.
GOVERNANCE_DENY_GLOBS: tuple[str, ...] = (
    # ── B4 verbatim ──
    "substrate/ainfera_core/governance/**",  # cage_runtime, certificates, budget, d2_cosign engine
    "agents/_killswitch.py",
    "*operator_policy*",                      # operator policy / D-ladder
    "*d2_cosign*", "*cosign*", "*co_sign*",   # the co-sign engine
    "*cage_runtime*", "*cage_integrity*", "*killswitch*", "*certificates*",
    "scripts/cage_integrity_check.py",
    ".github/workflows/cage-integrity.yml",
    "*eonwe*", "*broker*",                    # broker / EONWE_BROKER_MODE
    "*secret*",                               # secrets / credential material
    # ── B5 extension (AIN-565): the DGM loop must never reach the constraint surface ──
    "*arming*", "*enforce*",                  # arming / ENFORCE flip
    "*hard_deny*", "*l1_*", "*_l1*",          # L1 hard-deny capabilities
    "*killswitch*", "*guardian*",             # negative-authority / kill paths
)

# Capability domains a variant MAY evolve. Anything else is cross-domain → REFUSED.
VARIANT_ALLOWED_DOMAINS: frozenset[str] = frozenset({"tools", "prompts", "routing_policy"})

# Empirical-validation thresholds — reuse the FROZEN labs gates (Discipline #12 / anchor gate).
KAPPA_GATE: float = ANCHOR_KAPPA_GATE          # 0.60 — panel calibrated vs the verifiable anchor
KEEP_MIN_N: int = MIN_SAMPLE_PER_CELL          # 30 — sufficient-n per cell
KEEP_MIN_TSR_DELTA_PP: float = MIN_DELTA_PCT   # +0.5pp — measurable TSR improvement vs anchor

DEFAULT_ARCHIVE_PATH = "preprint/variant-archive.jsonl"  # append-only SHADOW archive


# ── variant + measured metrics ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Variant:
    """One proposed self-modification. ``parent_id`` gives the lineage; ``domain``/``target``
    are what it would touch (fenced); ``origin`` is the proposer (native-propose / kaizen)."""

    variant_id: str
    cell: str                  # "<task_type>::<candidate>" — the per-cell scope
    domain: str                # "tools" | "prompts" | "routing_policy"
    target: str                # the file/identifier the variant would change (fence-checked)
    parent_id: str | None      # lineage: the variant this descends from (None = root)
    origin: str = "native_propose"   # "native_propose" | "kaizen"
    description: str = ""


@dataclass(frozen=True)
class VariantMetrics:
    """Measured on a SHADOW holdout. In production these come from the live reward/judge loop
    (Step 0); the drill supplies them deterministically. CPST = cost per successful task,
    TSR = task success rate (the canonical eval-harness definitions)."""

    n: int
    successes: int
    cost_usd: float

    @property
    def tsr(self) -> float:
        return (self.successes / self.n) if self.n else 0.0

    @property
    def cpst(self) -> float | None:
        return (self.cost_usd / self.successes) if self.successes else None


@dataclass(frozen=True)
class ScoredVariant:
    """A variant bundled with everything the gate needs: its metrics, the anchor's metrics, the
    panel's κ vs the verifiable anchor, and (optionally) a DR-OPE result for the offline gate."""

    variant: Variant
    metrics: VariantMetrics
    anchor: VariantMetrics
    kappa: float | None
    ope: OPEResult | None = None


# ── the governance fence (B4 reuse + extend) ──────────────────────────────────────────


@dataclass(frozen=True)
class FenceVerdict:
    ok: bool
    kind: str          # "ok" | "governance" | "cross_domain"
    reason: str
    hit: str = ""      # the deny glob or bad domain that triggered


def _matches(path: str, glob: str) -> bool:
    """B4 ``_in_domain`` matching: prefix for ``/**`` globs, else fnmatch (+ ``/*`` suffix)."""
    if glob.endswith("/**"):
        return path.startswith(glob[:-2])
    return fnmatch(path, glob) or fnmatch(path, glob + "/*")


def governance_hit(target: str) -> str | None:
    """The first ``GOVERNANCE_DENY_GLOBS`` glob ``target`` hits, else None. Case-insensitive and
    over-matching — biased toward deny (fail-closed), exactly like the B4 fence + cosign needles."""
    t = target.replace("\\", "/").lower()
    for g in GOVERNANCE_DENY_GLOBS:
        if _matches(t, g.lower()):
            return g
    return None


def fence_variant(variant: Variant) -> FenceVerdict:
    """Governance precedence: a target hitting a governance deny glob is REFUSED + pages the
    founder even if its declared domain is allowed. Otherwise the domain must be a capability
    domain (tools/prompts/routing_policy); anything else is cross-domain → REFUSED."""
    g = governance_hit(variant.target)
    if g is not None:
        return FenceVerdict(
            False, "governance",
            f"variant target {variant.target!r} hits governance deny glob {g!r} — "
            "REFUSED + page founder (constraint surface is uncrossable)", g,
        )
    if variant.domain not in VARIANT_ALLOWED_DOMAINS:
        return FenceVerdict(
            False, "cross_domain",
            f"variant domain {variant.domain!r} not in allowed "
            f"{sorted(VARIANT_ALLOWED_DOMAINS)} — REFUSED", variant.domain,
        )
    return FenceVerdict(True, "ok", "within capability domain; no governance hit")


# ── the DR-OPE offline gate (composes labs.dr_ope) ────────────────────────────────────


def offline_promote_gate(ope: OPEResult | None, *, min_ess_frac: float = 0.30) -> bool:
    """Whether a variant *would* be promote-eligible OFFLINE — dr_ope's documented rule:
    ci_low(lift) > 0 AND a healthy effective sample size (ESS ≥ min_ess_frac · n). This is a
    PRECONDITION recorded on the receipt; live customer-facing promotion is still a founder D4 tap."""
    if ope is None or ope.n == 0:
        return False
    return ope.ci_low > 0.0 and ope.ess >= min_ess_frac * ope.n


# ── the empirical auto-keep decision (DGM validation gate) ─────────────────────────────


@dataclass(frozen=True)
class KeepDecision:
    variant_id: str
    parent_id: str | None
    cell: str
    domain: str
    decision: str               # "kept" | "held" | "refused"
    reason: str
    gates: dict[str, bool]
    kappa: float | None
    tsr_delta_pp: float
    cpst_delta: float | None
    n: int
    offline_promote_eligible: bool   # DR-OPE precondition (live promote still D4)
    page_founder: bool = False
    # cage invariants — never mutated off these values here
    shadow: bool = True
    live_promoted: bool = False
    requires_d4_to_promote: bool = True


def _deltas(vm: VariantMetrics, am: VariantMetrics) -> tuple[float, float | None]:
    """(TSR delta in percentage points, CPST delta in USD/success). CPST delta < 0 = cheaper."""
    tsr_delta_pp = (vm.tsr - am.tsr) * 100.0
    cpst_delta = None if (vm.cpst is None or am.cpst is None) else (vm.cpst - am.cpst)
    return tsr_delta_pp, cpst_delta


def evaluate_variant(scored: ScoredVariant, *, now: datetime) -> KeepDecision:
    """Fence FIRST (governance precedence), then the DGM empirical gate. Keep-in-SHADOW only if
    every gate passes; never live-promotes (D4). Deterministic."""
    v, vm, am = scored.variant, scored.metrics, scored.anchor
    tsr_delta_pp, cpst_delta = _deltas(vm, am)
    eligible = offline_promote_gate(scored.ope)

    fence = fence_variant(v)
    if not fence.ok:
        return KeepDecision(
            v.variant_id, v.parent_id, v.cell, v.domain, "refused", fence.reason,
            {"fence": False}, scored.kappa, tsr_delta_pp, cpst_delta, vm.n,
            offline_promote_eligible=eligible, page_founder=(fence.kind == "governance"),
        )

    gates = {
        "fence": True,
        "kappa_ge_gate": scored.kappa is not None and scored.kappa >= KAPPA_GATE,
        "sufficient_n": vm.n >= KEEP_MIN_N,
        "improves_tsr": tsr_delta_pp >= KEEP_MIN_TSR_DELTA_PP,
        "cpst_not_regressed": cpst_delta is not None and cpst_delta <= 0.0,
    }
    keep = all(gates.values())
    if keep:
        reason = (
            f"KEPT in SHADOW: TSR +{tsr_delta_pp:.2f}pp vs anchor, "
            f"CPST Δ${cpst_delta:.6f}/success, κ={scored.kappa:.2f}≥{KAPPA_GATE}, "
            f"n={vm.n}≥{KEEP_MIN_N} — archive only, live promote still D4"
        )
    else:
        failed = [k for k, ok in gates.items() if not ok]
        reason = f"HELD: empirical gate not met ({', '.join(failed)}) — not kept"
    return KeepDecision(
        v.variant_id, v.parent_id, v.cell, v.domain, "kept" if keep else "held", reason,
        gates, scored.kappa, tsr_delta_pp, cpst_delta, vm.n, offline_promote_eligible=eligible,
    )


# ── the append-only SHADOW archive (mirrors delta_logger) ──────────────────────────────


def decision_receipt(d: KeepDecision, *, now: datetime) -> dict[str, str]:
    """A flat, append-only archive receipt — the 'by receipt' proof. ``excluded_from_moat=1``:
    a variant row is never counted in customer/fleet moat metrics."""
    return {
        "kind": "variant_autokeep",
        "variant_id": d.variant_id,
        "parent_id": d.parent_id or "",
        "cell": d.cell,
        "domain": d.domain,
        "decision": d.decision,
        "reason": d.reason[:240],
        "kappa": f"{d.kappa:.3f}" if d.kappa is not None else "",
        "tsr_delta_pp": f"{d.tsr_delta_pp:.3f}",
        "cpst_delta": f"{d.cpst_delta:.6f}" if d.cpst_delta is not None else "",
        "n": str(d.n),
        "gates": json.dumps(d.gates),
        "offline_promote_eligible": "1" if d.offline_promote_eligible else "0",
        "requires_d4_to_promote": "1" if d.requires_d4_to_promote else "0",
        "live_promoted": "1" if d.live_promoted else "0",   # invariant: ALWAYS 0 here
        "shadow": "1" if d.shadow else "0",
        "excluded_from_moat": "1",
        "page_founder": "1" if d.page_founder else "0",
        "ts": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _assert_shadow(rec: dict[str, str]) -> None:
    """Tripwire: refuse to archive anything that claims a live promote. Makes 'auto-keep ≠ live
    promote' a code invariant — a receipt with live_promoted=1 aborts before it can be written."""
    if rec.get("live_promoted") != "0":
        raise AssertionError(f"live promote in SHADOW archive (D4 required): {rec.get('variant_id')!r}")
    if rec.get("shadow") != "1":
        raise AssertionError(f"non-shadow receipt: {rec.get('variant_id')!r}")


def append_receipt(rec: dict[str, str], path: str | Path | None = None) -> Path:
    """Append one receipt to the append-only archive (JSONL). Shadow-checked first."""
    _assert_shadow(rec)
    p = Path(path) if path else Path(DEFAULT_ARCHIVE_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return p


def read_archive(path: str | Path | None = None) -> list[dict[str, str]]:
    """Read the archive back (oldest→newest). Fail-closed → []."""
    p = Path(path) if path else Path(DEFAULT_ARCHIVE_PATH)
    if not p.exists():
        return []
    out: list[dict[str, str]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def lineage(receipts: list[dict[str, str]], variant_id: str) -> list[str]:
    """The ancestor chain for ``variant_id`` (self → parent → …root), following ``parent_id``."""
    parent_of = {r["variant_id"]: (r.get("parent_id") or "") for r in receipts}
    chain = [variant_id]
    seen = {variant_id}
    cur = parent_of.get(variant_id, "")
    while cur and cur not in seen:
        chain.append(cur)
        seen.add(cur)
        cur = parent_of.get(cur, "")
    return chain


# ── orchestrator ──────────────────────────────────────────────────────────────────────


def run_autokeep(scored: list[ScoredVariant], *, now: datetime,
                 archive_path: str | Path | None = None) -> list[dict[str, str]]:
    """Evaluate each scored variant and append its receipt to the SHADOW archive. Records only;
    no live promote (D4). Returns the receipts in order."""
    receipts: list[dict[str, str]] = []
    for s in scored:
        d = evaluate_variant(s, now=now)
        rec = decision_receipt(d, now=now)
        append_receipt(rec, archive_path)
        receipts.append(rec)
    return receipts


def render_report(receipts: list[dict[str, str]]) -> str:
    """A single human-readable report over an auto-keep run's receipts."""
    kept = [r for r in receipts if r["decision"] == "kept"]
    refused = [r for r in receipts if r["decision"] == "refused"]
    lines = ["═══ B5 variant archive — DGM auto-keep (SHADOW, fenced) ═══",
             f"variants: {len(receipts)} · kept {len(kept)} · "
             f"held {sum(1 for r in receipts if r['decision']=='held')} · refused {len(refused)}"]
    for r in receipts:
        extra = ""
        if r["decision"] == "kept":
            extra = (f" κ={r['kappa']} TSR+{r['tsr_delta_pp']}pp "
                     f"offline_eligible={r['offline_promote_eligible']} live_promoted={r['live_promoted']}")
        elif r["decision"] == "refused":
            extra = f" page_founder={r['page_founder']}"
        lines.append(
            f"  {r['decision']:<7} {r['variant_id']:<22} cell={r['cell']:<22} "
            f"domain={r['domain']:<13} parent={r['parent_id'] or '∅'}{extra}"
        )
    lines.append("invariant: live_promoted=0 on every receipt (auto-keep ≠ live promote; D4 required)")
    return "\n".join(lines)


# ── offline drill (CLI + tests) ───────────────────────────────────────────────────────


def _kept_anchor_kappa() -> float:
    """A real κ via labs.anchor_kappa.cohen_kappa over council-vs-anchor agreement pairs — high
    agreement → κ ≥ gate, so the kept variant is genuinely κ-gated, not a hard-coded number."""
    # 9 agree (A/B mix to avoid trivial single-class κ), 1 disagree → κ ≈ 0.78 ≥ 0.60.
    pairs: list[tuple[Vote, Vote]] = (
        [(Vote.A, Vote.A)] * 5 + [(Vote.B, Vote.B)] * 4 + [(Vote.A, Vote.B)]
    )
    k = cohen_kappa(pairs)
    return k if k is not None else 0.0


def _selftest_scored(now: datetime) -> list[ScoredVariant]:
    """Deterministic scenario covering the DoD: ≥2 scored variants with lineage, one auto-kept
    (κ-gated), and a governance-touching variant refused."""
    kappa = _kept_anchor_kappa()
    anchor = VariantMetrics(n=140, successes=112, cost_usd=2.80)  # TSR 0.800, CPST $0.025

    # V1 (root, prompts) — improves TSR + cheaper → KEPT. Anchored by a REAL verify() reward.
    vr = verify(VerifySample(
        task_type="extraction",
        request_payload={"messages": [{"role": "user", "content": "Return JSON {\"sum\": 4}"}],
                         "response_format": {"type": "json_object"}},
        response_payload={"choices": [{"message": {"content": "{\"sum\": 4}"}}]},
    ))
    v1 = Variant("var-20260620-001", "extraction::gpt-5-5", "prompts",
                 "prompts/extraction/system.md", None, "native_propose",
                 f"tighter extraction prompt (anchor verify reward={vr.reward})")
    m1 = VariantMetrics(n=120, successes=104, cost_usd=2.16)        # TSR 0.867 (+6.7pp), CPST $0.0208

    # DR-OPE: candidate concentrates on the better action → positive lift, healthy ESS.
    samples = (
        [LoggedSample("extraction", "gpt-5-5", 1.0, 0.5)] * 40
        + [LoggedSample("extraction", "gpt-5-5", 0.0, 0.5)] * 8
        + [LoggedSample("extraction", "alt", 0.0, 0.5)] * 12
    )
    ope = evaluate_policy(
        samples,
        target_probs={"extraction": {"gpt-5-5": 0.9, "alt": 0.1}},
        q_hat={("extraction", "gpt-5-5"): 0.83, ("extraction", "alt"): 0.40},
        seed=0,
    )

    # V2 (child of V1, routing_policy) — TSR gain below the +0.5pp margin → HELD (lineage proof:
    # a descendant of V1 that the empirical gate declines, so "keep only if it improves" has teeth).
    v2 = Variant("var-20260620-002", "extraction::gpt-5-5", "routing_policy",
                 "routing/policy/extraction.yaml", v1.variant_id, "kaizen",
                 "nudge cell weight (marginal)")
    m2 = VariantMetrics(n=90, successes=72, cost_usd=1.78)          # TSR 0.800 (+0.0pp) → below margin

    # V3 (governance-touching) — domain says 'tools' but the target is the co-sign engine → REFUSED + page.
    v3 = Variant("var-20260620-003", "tooling::any", "tools",
                 "substrate/ainfera_core/governance/d2_cosign_engine.py", None, "native_propose",
                 "(disallowed) attempt to edit the co-sign engine")
    m3 = VariantMetrics(n=200, successes=200, cost_usd=1.0)         # great metrics — must NOT matter

    return [
        ScoredVariant(v1, m1, anchor, kappa, ope),
        ScoredVariant(v2, m2, anchor, kappa, None),
        ScoredVariant(v3, m3, anchor, kappa, None),
    ]


def selftest(now: datetime, archive_path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    """Run the deterministic scenario; return (receipts, DoD failures). Empty failures = PASS."""
    receipts = run_autokeep(_selftest_scored(now), now=now, archive_path=archive_path)
    archive = read_archive(archive_path)
    by_id = {r["variant_id"]: r for r in archive}
    failures: list[str] = []

    scored_with_lineage = [r for r in archive if r["parent_id"]]
    if len(archive) < 2:
        failures.append(f"archive has {len(archive)} variants, want ≥2 scored")
    if not scored_with_lineage:
        failures.append("no variant carries lineage (parent_id)")
    else:
        child = scored_with_lineage[0]
        chain = lineage(archive, child["variant_id"])
        if len(chain) < 2:
            failures.append(f"lineage chain for {child['variant_id']} did not resolve a parent: {chain}")

    kept = [r for r in archive if r["decision"] == "kept"]
    if not kept:
        failures.append("no variant auto-kept in SHADOW")
    else:
        for r in kept:
            if float(r["kappa"] or 0) < KAPPA_GATE:
                failures.append(f"kept variant {r['variant_id']} not κ-gated (κ={r['kappa']})")

    refused_gov = [r for r in archive if r["decision"] == "refused" and r["page_founder"] == "1"]
    if not refused_gov:
        failures.append("no governance-touching variant refused + paged")

    if any(r["live_promoted"] != "0" for r in archive):
        failures.append("a variant was live-promoted (must be D4 only)")
    if any(r["excluded_from_moat"] != "1" for r in archive):
        failures.append("a variant row is not excluded from moat metrics")

    _ = by_id  # (kept for readability of the archive map)
    return receipts, failures


def _cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="variant_archive",
        description="B5 DGM variant archive — score variants vs anchor, auto-keep in SHADOW "
                    "(κ-gated, fenced); refuse governance-touching variants; never live-promote.",
    )
    p.add_argument("--selftest", action="store_true",
                   help="deterministic offline drill (tmp archive); exit 1 unless the DoD holds")
    p.add_argument("--archive", default=None, help="archive JSONL path (default: a temp file)")
    args = p.parse_args(argv)

    now = datetime(2026, 6, 20, 12, 0, 0, tzinfo=UTC)
    if args.archive:
        archive_path: str = args.archive
    else:
        archive_path = str(Path(tempfile.mkdtemp(prefix="b5-variant-")) / "variant-archive.jsonl")

    receipts, failures = selftest(now, archive_path)
    print(render_report(receipts))
    print(f"archive: {archive_path}")
    if failures:
        print("DoD: FAIL")
        for f in failures:
            print(f"  ✗ {f}")
        return 1
    print("DoD: PASS — ≥2 scored w/ lineage · 1 auto-kept (κ-gated) · 1 governance refused · 0 live-promoted")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_cli())


__all__ = [
    "DEFAULT_ARCHIVE_PATH",
    "GOVERNANCE_DENY_GLOBS",
    "KAPPA_GATE",
    "KEEP_MIN_N",
    "KEEP_MIN_TSR_DELTA_PP",
    "VARIANT_ALLOWED_DOMAINS",
    "FenceVerdict",
    "KeepDecision",
    "ScoredVariant",
    "Variant",
    "VariantMetrics",
    "append_receipt",
    "decision_receipt",
    "evaluate_variant",
    "fence_variant",
    "governance_hit",
    "lineage",
    "offline_promote_gate",
    "read_archive",
    "render_report",
    "run_autokeep",
    "selftest",
]
