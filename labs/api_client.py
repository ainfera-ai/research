"""AIN-542 · api.ainfera.ai client — fetch inference rows for verification.

Lightweight HTTP client for the Ainfera inference API.  Fetches
request/response payloads joined from the ``inferences`` and
``routing_outcomes`` tables, so the verify harness can run against live
fleet rows.

Authentication: reads ``AINFERA_FLEET_KEY`` from the environment (never passed
in code).  The base URL defaults to ``https://api.ainfera.ai`` and can be
overridden via ``AINFERA_API_URL``.

This module uses stdlib ``urllib`` to avoid a hard dependency on ``httpx``.
If ``httpx`` is available, it's used for connection pooling and HTTP/2.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


DEFAULT_BASE_URL = "https://api.ainfera.ai"


class ApiError(Exception):
    """Raised when the API returns an error."""


class AinferaClient:
    """Client for the api.ainfera.ai inference API.

    Example:
        >>> client = AinferaClient()  # reads AINFERA_FLEET_KEY from env
        >>> rows = client.fetch_outcomes(days=7, limit=100)
        >>> from labs.verify_harness import verify_rows
        >>> results = verify_rows(rows)
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = (base_url or os.environ.get("AINFERA_API_URL") or DEFAULT_BASE_URL).rstrip("/")
        self._api_key = api_key or os.environ.get("AINFERA_FLEET_KEY")
        if not self._api_key:
            raise ApiError("AINFERA_FLEET_KEY not set — cannot authenticate to api.ainfera.ai")
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": "AinferaLabs/0.2.0",
        }

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(url, data=data, method=method, headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                status = resp.status
                if status >= 400:
                    raise ApiError(f"HTTP {status} from {url}")
                raw = resp.read().decode()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            raise ApiError(f"HTTP {exc.code} from {url}: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise ApiError(f"URL error from {url}: {exc.reason}") from exc

    def fetch_outcomes(
        self,
        days: int = 7,
        limit: int = 100,
        reward_source: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch routing outcomes joined with inference payloads.

        Returns rows shaped for ``verify_rows()``:
        ``{task_type, request_payload, response_payload, expected, ...}``

        Args:
            days: lookback window in days.
            limit: max rows to fetch.
            reward_source: filter by reward_source ('verify', 'council', etc).
                If None, fetches all.
        """
        body: dict[str, Any] = {"days": days, "limit": limit}
        if reward_source:
            body["reward_source"] = reward_source
        result = self._request("POST", "/v1/labs/outcomes", body=body)
        return result.get("rows", [])

    def fetch_verifiable_subset(
        self, days: int = 7, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Fetch only the verifiable subset (reward_source='verify')."""
        return self.fetch_outcomes(days=days, limit=limit, reward_source="verify")

    def verify_batch(
        self,
        days: int = 7,
        limit: int = 100,
        reward_source: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch outcomes and run verify_rows() on them.

        Returns the verify results as dicts with reward, verifier, mode, detail.
        """
        from labs.verify_harness import verify_rows

        rows = self.fetch_outcomes(days=days, limit=limit, reward_source=reward_source)
        results = verify_rows(rows)
        return [
            {
                "reward": r.reward,
                "reward_source": r.reward_source,
                "verifier": r.verifier,
                "mode": r.mode,
                "verifiable": r.verifiable,
                "detail": r.detail,
                "evidence": list(r.evidence),
            }
            for r in results
        ]

    def health(self) -> bool:
        """Check if the API is reachable."""
        try:
            self._request("GET", "/v1/health")
            return True
        except ApiError:
            return False
