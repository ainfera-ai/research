"""snapshot.py — the weekly snapshot artifacts.

Two outputs per cycle:
  - **web snapshot** (sanitized) — exactly the shape the marketing site reads
    (`content/proof-snapshot.fixture.json`): labels / kinds / rates /
    cost_per_success only. No model slugs, no prompts, no secret values. This is
    what passes the FOUNDER sanitization gate and then replaces the web fixture.
  - **private results** — the rich per-arm metrics + win verdict, written to the
    Labs-private artifact dir (never the public repo / git).

`assert_sanitized` is a defence-in-depth check the founder gate can rely on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from labs.eval_harness import config
from labs.eval_harness.metrics import ArmMetrics, TypeStat

# Web-facing labels/kinds — the sanitized identity (never the real model slug).
ARM_LABEL: dict[str, tuple[str, str]] = {
    "C": ("Ainfera", "router"),
    "A": ("Premium pin", "pin"),
    "B": ("Cheap pin", "pin"),
    "D": ("Fusion panel", "panel"),
}
_ALLOWED_ARM_KEYS = {"id", "label", "kind", "cost_per_success", "success_rate"}


def build_web_snapshot(
    *,
    version: str,
    generated_at: str,
    arm_metrics: dict[str, ArmMetrics],
    type_table: dict[tuple[str, str], TypeStat],
) -> dict[str, Any]:
    """Sanitized, web-shaped snapshot. Mirrors lib/proof-snapshot.ts ProofSnapshot."""
    arms_out: list[dict[str, Any]] = []
    for aid in ("C", "A", "B", "D"):
        m = arm_metrics.get(aid)
        if m is None:
            continue
        label, kind = ARM_LABEL[aid]
        arms_out.append({
            "id": aid,
            "label": label,
            "kind": kind,
            "cost_per_success": round(m.cost_per_success, 6) if m.cost_per_success is not None else None,
            "success_rate": round(m.success_rate, 4),
        })

    types = sorted({tt for (_, tt) in type_table})
    tt_out: list[dict[str, Any]] = []
    for tt in types:
        arms_rate: dict[str, float] = {}
        for aid in ("C", "A", "B", "D"):
            stat = type_table.get((aid, tt))
            if stat is not None:
                arms_rate[aid] = round(stat.success_rate, 4)
        tt_out.append({"task_type": tt, "arms": arms_rate})

    return {
        "version": version,
        "generated_at": generated_at,
        "state": "measured",
        "source": "Ainfera Labs eval — measured weekly snapshot (sanitized; founder-gated).",
        "arms": arms_out,
        "task_types": tt_out,
    }


def assert_sanitized(web_snapshot: dict[str, Any]) -> None:
    """Fail closed if the web snapshot carries anything beyond the allowed shape
    (e.g. a leaked model slug, prompt, or secret value)."""
    for arm in web_snapshot.get("arms", []):
        extra = set(arm) - _ALLOWED_ARM_KEYS
        if extra:
            raise AssertionError(f"web snapshot arm carries non-sanitized keys: {sorted(extra)}")
        label, _ = ARM_LABEL.get(arm["id"], ("", ""))
        if arm["label"] != label:
            raise AssertionError(f"arm {arm['id']} label {arm['label']!r} is not the sanitized label.")
    if web_snapshot.get("state") not in {"measured", "illustrative"}:
        raise AssertionError("web snapshot state must be 'measured' or 'illustrative'.")


def write_artifacts(
    *, web_snapshot: dict[str, Any], private_results: dict[str, Any], out_dir: str | None = None
) -> dict[str, str]:
    """Write both artifacts to the Labs-private dir. Returns the paths."""
    assert_sanitized(web_snapshot)
    d = Path(out_dir or config.artifact_dir())
    d.mkdir(parents=True, exist_ok=True)
    ver = web_snapshot["version"]
    web_path = d / f"proof-snapshot-{ver}.json"   # → founder sanitization gate → web
    priv_path = d / f"eval-results-{ver}.json"     # rich, Labs-private
    web_path.write_text(json.dumps(web_snapshot, indent=2, sort_keys=True))
    priv_path.write_text(json.dumps(private_results, indent=2, sort_keys=True))
    return {"web": str(web_path), "private": str(priv_path)}
