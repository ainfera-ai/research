"""Tests for the ainfera SDK-native seat caller (AIN-542 live wiring)."""

from __future__ import annotations

from labs.council_seats import COUNCIL_SEATS
from labs.seat_caller import ainfera_health_check, ainfera_seat_caller, parse_pick


class _FakeInferenceResponse:
    """Mimics ainfera InferenceResponse — has .content (not .choices)."""
    def __init__(self, content: str):
        self.content = content


class _FakeAgent:
    """Mimics ainfera SDK Agent.inference(). Records calls."""
    def __init__(self, content: str = "FIRST", fail_times: int = 0):
        self._content = content
        self._fail_times = fail_times
        self.calls = 0

    def inference(self, *, model, messages, max_tokens=None, **extra):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise RuntimeError("502 gateway")
        return _FakeInferenceResponse(self._content)


def test_ainfera_seat_caller_parses_and_abstains():
    seat = COUNCIL_SEATS[0]  # Námo (anthropic)
    ok = ainfera_seat_caller(_FakeAgent("SECOND"))
    assert ok(seat, "T", "a", "b") == "second"


def test_ainfera_seat_caller_tie_on_unparseable():
    seat = COUNCIL_SEATS[0]
    caller = ainfera_seat_caller(_FakeAgent("blah blah"))
    assert caller(seat, "T", "a", "b") == "tie"


def test_ainfera_seat_caller_abstains_on_exhausted_retries():
    seat = COUNCIL_SEATS[0]
    flaky = ainfera_seat_caller(_FakeAgent(fail_times=99), retries=2, backoff_base=0.0)
    assert flaky(seat, "T", "a", "b") == "tie"


def test_ainfera_seat_caller_retries_transient_failure():
    seat = COUNCIL_SEATS[0]
    agent = _FakeAgent(fail_times=1, content="FIRST")
    caller = ainfera_seat_caller(agent, retries=2, backoff_base=0.0)
    assert caller(seat, "T", "a", "b") == "first"
    assert agent.calls == 2  # failed once → retried → succeeded


def test_ainfera_health_check_partitions():
    agent = _FakeAgent(content="ok")
    reach, unreach = ainfera_health_check(agent, COUNCIL_SEATS[:3], retries=0)
    assert len(reach) == 3 and unreach == []


def test_ainfera_health_check_quarantines_flaky():
    agent = _FakeAgent(fail_times=99)
    reach, unreach = ainfera_health_check(agent, COUNCIL_SEATS[:2], retries=0)
    assert reach == [] and len(unreach) == 2


def test_ainfera_seat_caller_uses_correct_message_format():
    """The caller must build pairwise messages (system + user with TASK, FIRST, SECOND)."""
    from labs.seat_caller import build_pairwise_messages
    msgs = build_pairwise_messages("Extract the order number", "48291", "The number is 48291")
    assert msgs[0]["role"] == "system"
    assert "TASK" in msgs[1]["content"]
    assert "Extract the order number" in msgs[1]["content"]
    assert "FIRST" in msgs[1]["content"]
    assert "SECOND" in msgs[1]["content"]
