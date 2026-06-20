"""AIN-542 Step 4 · live seat caller (parse, family map, gateway wrapper)."""

from __future__ import annotations

from labs.council_seats import COUNCIL_SEATS
from labs.seat_caller import family_of_slug, gateway_seat_caller, parse_pick


def test_parse_pick() -> None:
    assert parse_pick("FIRST") == "first"
    assert parse_pick("SECOND") == "second"
    assert parse_pick("TIE") == "tie"
    assert parse_pick("The first response is better.") == "first"
    assert parse_pick("They are equal") == "tie"
    assert parse_pick("") == "tie"
    assert (
        parse_pick("uhh both have merit and FIRST... SECOND") == "tie"
    )  # ambiguous → tie


def test_family_of_slug_real_roster() -> None:
    assert family_of_slug("claude-opus-4-7") == "anthropic"
    assert family_of_slug("gpt-5-5") == "openai"
    assert family_of_slug("gemini-3-1-pro") == "google"
    assert family_of_slug("grok-4") == "xai"
    assert family_of_slug("llama-4-405b-together") == "meta"
    assert family_of_slug("deepseek-v4-pro-deepinfra") == "deepseek"
    assert family_of_slug("minimax-m3-novita") == "minimax"
    assert family_of_slug("qwen-3-5-397b-novita") == "alibaba"
    assert family_of_slug(None) == "unknown"
    assert family_of_slug("some-mystery-model") == "unknown"


class _FakeResp:
    def __init__(self, text):
        self.choices = [
            type("C", (), {"message": type("M", (), {"content": text})()})()
        ]


class _FakeClient:
    def __init__(self, text=None, raise_exc=False):
        self._text, self._raise = text, raise_exc
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs):
        if self._raise:
            raise RuntimeError("gateway 500")
        return _FakeResp(self._text)


def test_gateway_seat_caller_parses_and_abstains() -> None:
    seat = COUNCIL_SEATS[0]
    ok = gateway_seat_caller(_FakeClient("SECOND"))
    assert ok(seat, "T", "a", "b") == "second"
    # a flaky seat abstains, never crashes
    flaky = gateway_seat_caller(_FakeClient(raise_exc=True))
    assert flaky(seat, "T", "a", "b") == "tie"


# ── AIN-546: retry/backoff + health_check (quarantine, don't silent-drop) ─────


class _FlakyClient:
    def __init__(self, fail_times=0, text="ok"):
        self.fail_times, self.calls, self._text = fail_times, 0, text
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("502")
        return _FakeResp(self._text)


def test_retry_recovers_after_transient_failure() -> None:
    from labs.seat_caller import gateway_seat_caller

    c = _FlakyClient(fail_times=1)
    caller = gateway_seat_caller(c, retries=2, backoff_base=0.0)  # no sleep in tests
    assert caller(COUNCIL_SEATS[0], "T", "a", "b") in ("first", "second", "tie")
    assert c.calls == 2  # failed once → retried → succeeded


def test_health_check_partitions_reachable_unreachable() -> None:
    from labs.seat_caller import health_check

    reach, unreach = health_check(_FakeClient(text="ok"), COUNCIL_SEATS[:2], retries=0)
    assert len(reach) == 2 and unreach == []
    reach2, unreach2 = health_check(
        _FlakyClient(fail_times=99), COUNCIL_SEATS[:2], retries=0
    )
    assert reach2 == [] and len(unreach2) == 2


def test_pairwise_messages_include_the_task() -> None:
    from labs.seat_caller import build_pairwise_messages

    msgs = build_pairwise_messages(
        "Extract the order number", "48291", "The number is 48291"
    )
    user = msgs[-1]["content"]
    assert "Extract the order number" in user and "TASK" in user
