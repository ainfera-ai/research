"""AIN-542 · CLI and API client tests."""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import patch, MagicMock

import pytest

from labs.cli import main, _parse_json_or_string


# ── _parse_json_or_string ────────────────────────────────────────────────────


class TestParseJsonOrString:
    def test_parses_json_dict(self) -> None:
        result = _parse_json_or_string('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parses_json_list(self) -> None:
        result = _parse_json_or_string('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_returns_raw_string_on_failure(self) -> None:
        result = _parse_json_or_string("not json")
        assert result == "not json"

    def test_returns_int(self) -> None:
        result = _parse_json_or_string("42")
        assert result == 42


# ── CLI verify command ───────────────────────────────────────────────────────


class TestCLIVerify:
    def test_verify_code(self, capsys: pytest.CaptureFixture[str]) -> None:
        resp = json.dumps({"choices": [{"message": {"content": "```python\nx = 1\n```"}}]})
        rc = main(["verify", "--task-type", "code", "--response", resp])
        assert rc == 0
        out = capsys.readouterr().out
        assert "reward:" in out
        assert "1.0" in out

    def test_verify_math_with_steps(self, capsys: pytest.CaptureFixture[str]) -> None:
        resp = json.dumps({
            "content": [{"type": "text", "text": "2*x + 3 = 7\n2*x = 4\nx = 2\nFinal answer: 2"}]
        })
        rc = main(["verify", "--task-type", "math", "--response", resp, "--expected", "2"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "1.0" in out

    def test_verify_qa(self, capsys: pytest.CaptureFixture[str]) -> None:
        resp = json.dumps({
            "content": [{"type": "text", "text": "Paris is the capital of France."}]
        })
        expected = json.dumps({"evidence": "Paris is the capital of France.", "answer": "Paris"})
        rc = main(["verify", "--task-type", "qa", "--response", resp, "--expected", expected])
        assert rc == 0
        out = capsys.readouterr().out
        assert "1.0" in out

    def test_verify_defers_on_no_gold(self, capsys: pytest.CaptureFixture[str]) -> None:
        resp = json.dumps({"choices": [{"message": {"content": "hello"}}]})
        rc = main(["verify", "--task-type", "chat", "--response", resp])
        assert rc == 1  # deferred → exit 1
        out = capsys.readouterr().out
        assert "None" in out


# ── CLI list-verifiers ───────────────────────────────────────────────────────


class TestCLIListVerifiers:
    def test_list_verifiers(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["list-verifiers"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "code" in out
        assert "math" in out
        assert "qa" in out
        assert "sql" in out


# ── CLI benchmark ────────────────────────────────────────────────────────────


class TestCLIBenchmark:
    def test_benchmark_runs(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["benchmark"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "ainfera_learned" in out
        assert "save%" in out
        # The fix should produce positive savings
        assert "+1" in out or "+1" in out  # at least positive


# ── API client ───────────────────────────────────────────────────────────────


class TestAinferaClient:
    def test_requires_api_key(self) -> None:
        from labs.api_client import AinferaClient, ApiError

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ApiError, match="AINFERA_FLEET_KEY"):
                AinferaClient()

    def test_init_with_key(self) -> None:
        from labs.api_client import AinferaClient

        client = AinferaClient(api_key="test-key")
        assert client.base_url == "https://api.ainfera.ai"
        assert client._api_key == "test-key"

    def test_init_with_custom_url(self) -> None:
        from labs.api_client import AinferaClient

        client = AinferaClient(api_key="test-key", base_url="https://staging.ainfera.ai")
        assert client.base_url == "https://staging.ainfera.ai"

    def test_headers(self) -> None:
        from labs.api_client import AinferaClient

        client = AinferaClient(api_key="test-key")
        headers = client._headers()
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"

    def test_health_returns_bool(self) -> None:
        from labs.api_client import AinferaClient

        client = AinferaClient(api_key="test-key")
        # Should return False when the API is unreachable
        assert client.health() is False
