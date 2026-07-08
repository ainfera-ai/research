"""AIN-542 Tier A · verify_math (SymPy step checking) + verify_qa (evidence) tests."""

from __future__ import annotations

import pytest

from labs.verify_harness import (
    FAMILY_MATH_STEPS,
    FAMILY_QA,
    VerifySample,
    verify,
)


def _anthropic_text(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}]}


def _openai(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


# ── verify_math: SymPy step/derivation checking ──────────────────────────────


class TestVerifyMathSteps:
    """Tests for the SymPy-based math step verifier."""

    def test_correct_derivation_scores_1(self) -> None:
        resp = _anthropic_text(
            "Let me solve this step by step.\n"
            "2x + 3 = 7\n"
            "2x = 4\n"
            "x = 2\n"
            "Final answer: 2"
        )
        r = verify(VerifySample("math", response_payload=resp, expected=2))
        assert r.reward is not None
        assert r.reward == 1.0
        assert r.verifier == FAMILY_MATH_STEPS
        assert r.mode == "reference"

    def test_wrong_step_scores_partial(self) -> None:
        resp = _anthropic_text(
            "2x + 3 = 7\n"
            "2x = 5\n"  # WRONG: should be 4
            "x = 5/2\n"
            "Final answer: 2.5"
        )
        r = verify(VerifySample("math", response_payload=resp, expected=2.5))
        assert r.reward is not None
        assert 0.0 < r.reward < 1.0  # partial credit

    def test_correct_steps_wrong_final_answer_scores_half(self) -> None:
        """Steps verify but final answer doesn't match expected → 0.5."""
        resp = _anthropic_text(
            "2x + 3 = 7\n"
            "2x = 4\n"
            "x = 2\n"
            "Final answer: 3"  # wrong final answer
        )
        r = verify(VerifySample("math", response_payload=resp, expected=5))
        assert r.reward is not None
        assert r.reward == 0.5

    def test_algebraic_identity_verifies(self) -> None:
        """SymPy should verify algebraic identities like (x+1)^2 = x^2 + 2x + 1."""
        resp = _anthropic_text(
            "(x+1)^2 = x^2 + 2*x + 1\n"
            "Final answer: x^2 + 2*x + 1"
        )
        r = verify(VerifySample("math", response_payload=resp))
        # No expected → still checks steps
        assert r.reward is not None
        assert r.reward == 1.0

    def test_no_steps_falls_back_to_math_exact(self) -> None:
        """If no derivation steps found, falls back to final-answer checking."""
        resp = _anthropic_text("Final answer: 42")
        r = verify(VerifySample("math", response_payload=resp, expected=42))
        assert r.reward == 1.0

    def test_no_steps_no_gold_defers(self) -> None:
        """No steps and no gold → defer to Council."""
        resp = _anthropic_text("The answer is 42.")
        r = verify(VerifySample("math", response_payload=resp))
        assert r.reward is None

    def test_empty_output_defers(self) -> None:
        r = verify(VerifySample("math", response_payload=_anthropic_text("")))
        assert r.reward is None

    def test_arrow_prefixed_steps(self) -> None:
        """Steps with =>, →, ⇒ prefixes are stripped and verified."""
        resp = _anthropic_text(
            "x + 2 = 5\n"
            "=> x = 3\n"
            "Final answer: 3"
        )
        r = verify(VerifySample("math", response_payload=resp, expected=3))
        assert r.reward is not None
        assert r.reward == 1.0

    def test_fraction_steps(self) -> None:
        """Fraction arithmetic verified by SymPy."""
        resp = _anthropic_text(
            "1/2 + 1/3 = 5/6\n"
            "Final answer: 5/6"
        )
        r = verify(VerifySample("math", response_payload=resp, expected="5/6"))
        assert r.reward is not None
        assert r.reward == 1.0

    def test_substantive_not_liveness(self) -> None:
        """A 200-OK with empty body never scores 1."""
        r = verify(VerifySample("math", response_payload={}, expected=42))
        assert r.reward in (None, 0.0)

    def test_boxed_answer_verified(self) -> None:
        """\\boxed{} answer is extracted and verified."""
        resp = _anthropic_text(
            "3x = 12\n"
            "x = 4\n"
            "\\boxed{4}"
        )
        r = verify(VerifySample("math", response_payload=resp, expected=4))
        assert r.reward is not None
        assert r.reward == 1.0


# ── verify_qa: evidence/factuality checking ──────────────────────────────────


class TestVerifyQA:
    """Tests for the evidence/factuality verifier."""

    def test_grounded_correct_answer_scores_1(self) -> None:
        evidence = (
            "Paris is the capital of France. "
            "The Eiffel Tower is located in Paris."
        )
        resp = _anthropic_text(
            "Paris is the capital of France. "
            "The Eiffel Tower is located in Paris."
        )
        r = verify(VerifySample(
            "qa",
            response_payload=resp,
            expected={"evidence": evidence, "answer": "Paris"},
        ))
        assert r.reward is not None
        assert r.reward == 1.0
        assert r.verifier == FAMILY_QA

    def test_ungrounded_answer_scores_below_1(self) -> None:
        """Response with fabricated claims not in evidence → grounding fails."""
        evidence = "Paris is the capital of France."
        resp = _anthropic_text(
            "The capital of France is Lyon. "  # wrong, not in evidence
            "Lyon is the largest city in France."  # fabricated
        )
        r = verify(VerifySample(
            "qa",
            response_payload=resp,
            expected={"evidence": evidence, "answer": "Lyon"},
        ))
        # Answer matches gold but grounding fails → 0.5
        assert r.reward is not None
        assert r.reward == 0.5

    def test_grounded_wrong_answer_scores_half(self) -> None:
        evidence = "Paris is the capital of France."
        resp = _anthropic_text(
            "Paris is the capital of France."
        )
        r = verify(VerifySample(
            "qa",
            response_payload=resp,
            expected={"evidence": evidence, "answer": "Lyon"},
        ))
        # Grounding OK but answer wrong → 0.5
        assert r.reward is not None
        assert r.reward == 0.5

    def test_no_evidence_no_gold_defers(self) -> None:
        resp = _anthropic_text("Some answer.")
        r = verify(VerifySample("qa", response_payload=resp))
        assert r.reward is None

    def test_evidence_only_no_answer(self) -> None:
        """With evidence but no gold answer, only grounding is checked."""
        evidence = "The speed of light is 299,792,458 meters per second."
        resp = _anthropic_text(
            "The speed of light is 299,792,458 meters per second."
        )
        r = verify(VerifySample(
            "qa",
            response_payload=resp,
            expected={"evidence": evidence},
        ))
        assert r.reward is not None
        assert r.reward == 1.0  # grounding passes, no answer to check

    def test_string_expected_treated_as_gold_answer(self) -> None:
        """Bare string expected → answer-match only (no grounding)."""
        resp = _anthropic_text("The answer is 42.")
        r = verify(VerifySample("qa", response_payload=resp, expected="42"))
        assert r.reward is not None
        assert r.reward == 1.0

    def test_empty_output_defers(self) -> None:
        r = verify(VerifySample("qa", response_payload=_anthropic_text("")))
        assert r.reward is None

    def test_substantive_not_liveness(self) -> None:
        r = verify(VerifySample("qa", response_payload={}, expected="x"))
        assert r.reward in (None, 0.0)

    def test_partial_grounding(self) -> None:
        """Mixed grounded and ungrounded sentences → partial if <70% grounded."""
        evidence = "The Earth orbits the Sun. The Moon orbits the Earth."
        resp = _anthropic_text(
            "The Earth orbits the Sun. "
            "Mars is made of cheese. "  # unsupported
            "Jupiter has 79 moons. "  # unsupported
            "The Moon orbits the Earth."
        )
        r = verify(VerifySample(
            "qa",
            response_payload=resp,
            expected={"evidence": evidence},
        ))
        # 2/4 sentences grounded = 50% < 70% → grounding fails
        # but no answer to match → reward should be < 1
        assert r.reward is not None
        assert r.reward < 1.0

    def test_qa_dispatches_correctly(self) -> None:
        """The verify() dispatcher routes 'qa' task_type to verify_qa."""
        resp = _anthropic_text("The answer is Paris.")
        r = verify(VerifySample("qa", response_payload=resp, expected="Paris"))
        assert r.verifier == FAMILY_QA

    def test_evidence_alias_dispatches(self) -> None:
        """The 'evidence' task_type alias also dispatches to verify_qa."""
        resp = _anthropic_text("The answer is Paris.")
        r = verify(VerifySample("evidence", response_payload=resp, expected="Paris"))
        assert r.verifier == FAMILY_QA
