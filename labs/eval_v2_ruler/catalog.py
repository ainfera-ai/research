"""catalog.py — fetch the live model catalog from the gateway.

Returns a list of model slugs that are active and routable. The router
entry (ainfera-inference) is excluded — it's a router, not a model.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass

from labs.eval_v2_ruler import config


@dataclass(frozen=True)
class CatalogModel:
    slug: str
    display_name: str
    provider: str
    type: str  # "model" | "router"
    input_cost_per_million: float | None
    output_cost_per_million: float | None
    context_window: int | None
    capabilities: tuple[str, ...]
    routable_status: str


def fetch_catalog() -> list[CatalogModel]:
    """Fetch the live model catalog. Returns active, routable models only."""
    req = urllib.request.Request(
        config.CATALOG_URL,
        headers={
            "Authorization": f"Bearer {os.environ[config.GATEWAY_KEY_ENV]}",
            "Accept": "application/json",
            "User-Agent": "eval-v2-ruler/0.1",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = json.loads(r.read())

    out: list[CatalogModel] = []
    for m in raw:
        if m.get("routable_status") != "active":
            continue
        if m.get("type") == "router":
            continue  # skip ainfera-inference (the router itself)
        out.append(
            CatalogModel(
                slug=m["slug"],
                display_name=m.get("display_name", m["slug"]),
                provider=m.get("provider", "unknown"),
                type=m.get("type", "model"),
                input_cost_per_million=_safe_float(m.get("input_cost_per_million_usd")),
                output_cost_per_million=_safe_float(m.get("output_cost_per_million_usd")),
                context_window=m.get("context_window"),
                capabilities=tuple(m.get("capabilities", [])),
                routable_status=m.get("routable_status", "active"),
            )
        )
    return out


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None
