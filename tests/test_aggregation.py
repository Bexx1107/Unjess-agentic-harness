"""Tests for unjess.aggregation — result aggregation strategies."""

from typing import Any

import pytest

from unjess.aggregation import AggregatedResult, best_pick, merge, synthesize, vote


class TestAggregatedResult:
    """AggregatedResult dataclass construction and summary output."""

    def test_defaults(self) -> None:
        r = AggregatedResult(strategy="merge", output="hello")
        assert r.source_count == 0
        assert r.metadata is None

    def test_all_fields(self) -> None:
        r = AggregatedResult(
            strategy="vote",
            output="yes",
            source_count=3,
            metadata={"winner_count": 2},
        )
        assert r.strategy == "vote"
        assert r.output == "yes"
        assert r.source_count == 3
        assert r.metadata == {"winner_count": 2}

    def test_summary_format(self) -> None:
        r = AggregatedResult(strategy="best_pick", output="x", source_count=5)
        assert r.summary() == "Aggregated (best_pick) from 5 sources"

    def test_summary_zero_sources(self) -> None:
        r = AggregatedResult(strategy="merge", output="")
        assert "0 sources" in r.summary()


class TestMerge:
    """merge() concatenation strategy."""

    def test_single_result(self) -> None:
        res = merge({"agent-1": "answer one"})
        assert res.strategy == "merge"
        assert res.source_count == 1
        assert "agent-1" in res.output
        assert "answer one" in res.output

    def test_multiple_results_default_separator(self) -> None:
        results = {"aaa": "first", "bbb": "second"}
        res = merge(results)
        assert res.source_count == 2
        assert "\n\n---\n\n" in res.output
        assert "first" in res.output
        assert "second" in res.output

    def test_custom_separator(self) -> None:
        results = {"a1": "hello", "a2": "world"}
        res = merge(results, separator=" | ")
        assert " | " in res.output
        assert "\n\n---\n\n" not in res.output

    def test_empty_results(self) -> None:
        res = merge({})
        assert res.source_count == 0
        assert res.output == ""

    def test_agent_id_truncated_to_8_chars(self) -> None:
        long_id = "abcdefghijklmnop"
        res = merge({long_id: "text"})
        assert f"**[{long_id[:8]}]:**" in res.output
        assert long_id[9:] not in res.output.split(":**")[0]

    def test_agent_id_shorter_than_8_chars(self) -> None:
        short_id = "ab"
        res = merge({short_id: "text"})
        assert f"**[{short_id}]:**" in res.output

    def test_preserves_order(self) -> None:
        from collections import OrderedDict

        ordered = OrderedDict([("first", "AAA"), ("second", "BBB"), ("third", "CCC")])
        res = merge(ordered)
        parts = res.output.split("\n\n---\n\n")
        assert "AAA" in parts[0]
        assert "BBB" in parts[1]
        assert "CCC" in parts[2]

    def test_metadata_is_none(self) -> None:
        res = merge({"a": "b"})
        assert res.metadata is None


class TestBestPick:
    """best_pick() scoring strategy."""

    def test_default_scorer_picks_longest(self) -> None:
        results = {"short": "hi", "long": "hello world this is long"}
        res = best_pick(results)
        assert res.strategy == "best_pick"
        assert res.output == "hello world this is long"
        assert res.metadata is not None
        assert res.metadata["winner"] == "long"

    def test_custom_scorer(self) -> None:
        results = {"a": "short", "b": "longer text", "c": "x"}
        # Custom scorer: prefer shorter text
        res = best_pick(results, scorer=lambda t: -float(len(t)))
        assert res.output == "x"
        assert res.metadata["winner"] == "c"

    def test_empty_results(self) -> None:
        res = best_pick({})
        assert res.output == ""
        assert res.source_count == 0
        assert res.metadata is None

    def test_single_result(self) -> None:
        res = best_pick({"only": "one result"})
        assert res.output == "one result"
        assert res.source_count == 1
        assert res.metadata["winner"] == "only"

    def test_scorer_exception_gives_zero_score(self) -> None:
        def bad_scorer(text: str) -> float:
            if text == "bad":
                raise ValueError("nope")
            return 10.0

        results = {"good": "ok", "fail": "bad"}
        res = best_pick(results, scorer=bad_scorer)
        assert res.output == "ok"
        assert res.metadata["winner"] == "good"
        assert res.metadata["score"] == 10.0

    def test_all_scorers_fail_picks_first_with_zero(self) -> None:
        def always_fail(text: str) -> float:
            raise RuntimeError("boom")

        results = {"a": "one", "b": "two"}
        res = best_pick(results, scorer=always_fail)
        # All get score 0.0; one of them is picked
        assert res.metadata["score"] == 0.0
        assert res.source_count == 2

    def test_score_in_metadata(self) -> None:
        results = {"x": "hello"}
        res = best_pick(results, scorer=lambda t: 42.5)
        assert res.metadata["score"] == 42.5

    def test_tied_scores_picks_one(self) -> None:
        results = {"a": "xxx", "b": "yyy"}
        res = best_pick(results, scorer=lambda t: 1.0)
        assert res.output in ("xxx", "yyy")
        assert res.source_count == 2


class TestSynthesize:
    """synthesize() combination strategy."""

    def test_no_synthesizer_dedup_merge(self) -> None:
        results = {
            "a": "line one\nline two",
            "b": "line two\nline three",
        }
        res = synthesize(results)
        assert res.strategy == "synthesize"
        assert res.source_count == 2
        lines = res.output.splitlines()
        # "line two" should appear only once (dedup)
        assert lines.count("line two") == 1
        assert "line one" in res.output
        assert "line three" in res.output

    def test_dedup_preserves_original_whitespace(self) -> None:
        results = {"a": "  indented line\nnormal"}
        res = synthesize(results)
        assert "  indented line" in res.output

    def test_dedup_strips_for_comparison(self) -> None:
        results = {
            "a": "  hello  ",
            "b": "hello",
        }
        res = synthesize(results)
        # Both are "hello" stripped, so only one should appear
        lines = [l for l in res.output.splitlines() if l.strip()]
        assert len(lines) == 1

    def test_dedup_skips_empty_lines(self) -> None:
        results = {"a": "line\n\n\nanother"}
        res = synthesize(results)
        # Blank lines are stripped to empty and skipped
        assert "\n\n" not in res.output

    def test_custom_synthesizer_called(self) -> None:
        def my_synth(texts: list[str]) -> str:
            return " + ".join(texts)

        results = {"a": "one", "b": "two"}
        res = synthesize(results, synthesizer=my_synth)
        assert res.output == "one + two"
        assert res.source_count == 2

    def test_synthesizer_failure_falls_back(self) -> None:
        def failing_synth(texts: list[str]) -> str:
            raise RuntimeError("synthesis broke")

        results = {"a": "alpha", "b": "beta"}
        res = synthesize(results, synthesizer=failing_synth)
        # Fallback: join with "\n\n"
        assert "alpha" in res.output
        assert "beta" in res.output
        assert "\n\n" in res.output

    def test_empty_results(self) -> None:
        res = synthesize({})
        assert res.source_count == 0
        assert res.output == ""

    def test_single_result_no_synthesizer(self) -> None:
        res = synthesize({"only": "just one line"})
        assert res.output == "just one line"

    def test_none_synthesizer_uses_dedup(self) -> None:
        results = {"a": "x", "b": "y"}
        res = synthesize(results, synthesizer=None)
        assert "x" in res.output
        assert "y" in res.output

    def test_metadata_is_none(self) -> None:
        res = synthesize({"a": "text"})
        assert res.metadata is None


class TestVote:
    """vote() majority-vote strategy."""

    def test_clear_majority(self) -> None:
        results = {"a": "yes", "b": "yes", "c": "no"}
        res = vote(results)
        assert res.strategy == "vote"
        assert res.output == "yes"
        assert res.source_count == 3
        assert res.metadata is not None
        assert res.metadata["winner_count"] == 2

    def test_case_insensitive_voting(self) -> None:
        results = {"a": "Yes", "b": "YES", "c": "no"}
        res = vote(results)
        assert res.output.lower() == "yes"
        assert res.metadata["winner_count"] == 2

    def test_whitespace_normalization(self) -> None:
        results = {"a": "  yes  ", "b": "yes", "c": "no"}
        res = vote(results)
        assert res.output == "yes"
        assert res.metadata["winner_count"] == 2

    def test_empty_results(self) -> None:
        res = vote({})
        assert res.output == ""
        assert res.source_count == 0
        assert res.metadata is None

    def test_single_vote(self) -> None:
        res = vote({"a": "only"})
        assert res.output == "only"
        assert res.metadata["winner_count"] == 1

    def test_tie_picks_one(self) -> None:
        results = {"a": "foo", "b": "bar"}
        res = vote(results)
        assert res.output in ("foo", "bar")
        assert res.metadata["winner_count"] == 1

    def test_votes_dict_in_metadata(self) -> None:
        results = {"a": "yes", "b": "no", "c": "yes"}
        res = vote(results)
        votes_map = res.metadata["votes"]
        assert votes_map["yes"] == 2
        assert votes_map["no"] == 1

    def test_preserves_original_case_in_output(self) -> None:
        results = {"a": "Accept", "b": "accept", "c": "ACCEPT"}
        res = vote(results)
        # Output should be original-cased from first match found
        assert res.output == "Accept"

    def test_unanimous_vote(self) -> None:
        results = {"a": "agree", "b": "agree", "c": "agree"}
        res = vote(results)
        assert res.output == "agree"
        assert res.metadata["winner_count"] == 3

    def test_many_candidates(self) -> None:
        results = {
            "a": "opt1", "b": "opt2", "c": "opt2",
            "d": "opt3", "e": "opt2", "f": "opt1",
        }
        res = vote(results)
        assert res.output == "opt2"
        assert res.metadata["winner_count"] == 3
