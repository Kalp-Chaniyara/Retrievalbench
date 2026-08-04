"""Recommendation engine: does it apply the documented ranking policy?

The policy (recommend.py docstring): rank by correctness PASS RATE, tie-break on
faithfulness, then prefer cheaper. These tests fabricate runs so that only the
correct policy produces the expected winner — a test that would pass regardless
of the ranking key proves nothing.
"""

import pytest

from retrievalbench.model import Budget, FailureMode
from retrievalbench.recommend import (
    ComparabilityError,
    config_diff,
    latest_per_config,
    recommend,
)

from .conftest import make_config, make_run

NONE = FailureMode.NONE
F1 = FailureMode.RETRIEVAL_MISS
F_GEN = FailureMode.GENERATION_FAILURE


def test_abstainer_loses_despite_perfect_metrics() -> None:
    """THE load-bearing test.

    An "I don't know" pipeline scores faithfulness 1.0 (deepeval hardcodes 1 when
    the answer yields zero claims), and is cheaper and faster because it emits
    almost no tokens. It is rigged here to win on THREE of four axes. Only
    ranking on pass rate rejects it. If someone "optimises" the sort key to use
    faithfulness or cost, this test goes red.
    """
    abstainer = make_run(
        make_config("abstainer"),
        modes=[F_GEN] * 8,  # 0/8 correct
        latency=900,  # fastest
        cost=0.002,  # cheapest
        faithfulness=1.0,  # perfect metrics
    )
    real = make_run(
        make_config("real", retrieval="hybrid"),
        modes=[NONE] * 6 + [F_GEN, F1],  # 6/8 correct
        latency=1500,
        cost=0.004,
        faithfulness=0.72,
    )

    result = recommend([abstainer, real])

    assert result.winner is not None
    assert result.winner.config_name == "real"
    assert result.winner.pass_rate == 0.75


def test_faithfulness_only_breaks_ties() -> None:
    """Equal pass rate -> faithfulness decides. It is a tie-break, never the key."""
    low = make_run(
        make_config("low"), modes=[NONE] * 4, latency=100, cost=0.001, faithfulness=0.60
    )
    high = make_run(
        make_config("high"),
        modes=[NONE] * 4,
        latency=100,
        cost=0.001,
        faithfulness=0.95,
    )
    result = recommend([low, high])
    assert result.winner is not None
    assert result.winner.config_name == "high"


def test_pareto_dominated_config_is_flagged() -> None:
    """Same quality, 5x cost, 3x latency -> strictly worse on every axis."""
    cheap = make_run(
        make_config("cheap"), modes=[NONE] * 6 + [F_GEN] * 2, latency=1000, cost=0.002
    )
    lavish = make_run(
        make_config("lavish", retrieval="hybrid", rerank=True),
        modes=[NONE] * 6 + [F_GEN] * 2,
        latency=3000,
        cost=0.010,
    )
    result = recommend([cheap, lavish])
    assert "lavish" in result.dominated
    assert "cheap" not in result.dominated
    assert result.winner is not None and result.winner.config_name == "cheap"


def test_quality_reported_in_points_not_percent() -> None:
    """50% -> 83% is +33 POINTS. Reporting it as '+66%' would inflate a modest
    gain; cost/latency stay ratios, where a multiplier is the honest unit."""
    small = make_run(
        make_config("small"),
        modes=[NONE] * 6 + [F1] * 4 + [F_GEN] * 2,
        latency=1420,
        cost=0.0031,
    )
    big = make_run(
        make_config("big", retrieval="hybrid", rerank=True),
        modes=[NONE] * 10 + [F1, F_GEN],
        latency=2840,
        cost=0.0074,
    )
    result = recommend([big, small])

    assert result.winner is not None and result.winner.config_name == "big"
    assert result.reference is not None and result.reference.config_name == "small"
    assert result.quality_gain_points == pytest.approx(33.33, abs=0.1)
    assert result.cost_ratio == pytest.approx(2.39, abs=0.01)
    assert result.latency_ratio == pytest.approx(2.0, abs=0.01)
    assert not result.diminishing_returns


def test_diminishing_returns_flagged_when_gain_is_small_but_costly() -> None:
    """Rule: gain < 5 points AND (cost or latency) > 1.5x -> call it out.

    Sized at n=24 so ONE extra passing query is 4.17 points, i.e. genuinely
    below the threshold. (At n=12 one query is 8.3 points and the rule correctly
    does NOT fire — the gain is real at that resolution.)
    """
    cheap = make_run(
        make_config("cheap"), modes=[NONE] * 12 + [F_GEN] * 12, latency=1000, cost=0.002
    )
    pricey = make_run(
        make_config("pricey", retrieval="hybrid"),
        modes=[NONE] * 13 + [F_GEN] * 11,
        latency=1100,
        cost=0.008,
    )
    result = recommend([cheap, pricey])
    assert result.winner is not None and result.winner.config_name == "pricey"
    assert result.quality_gain_points == pytest.approx(4.17, abs=0.01)
    assert result.diminishing_returns


def test_large_gain_is_not_flagged_as_diminishing() -> None:
    """The complement: a real quality jump must NOT be discouraged."""
    cheap = make_run(
        make_config("cheap"), modes=[NONE] * 12 + [F_GEN] * 12, latency=1000, cost=0.002
    )
    better = make_run(
        make_config("better", retrieval="hybrid"),
        modes=[NONE] * 20 + [F_GEN] * 4,
        latency=1100,
        cost=0.008,
    )
    result = recommend([cheap, better])
    assert result.winner is not None and result.winner.config_name == "better"
    assert not result.diminishing_returns


def test_budget_excludes_over_limit_configs() -> None:
    small = make_run(
        make_config("small"), modes=[NONE] * 6 + [F1] * 6, latency=1420, cost=0.0031
    )
    big = make_run(
        make_config("big", retrieval="hybrid"),
        modes=[NONE] * 10 + [F1] * 2,
        latency=2840,
        cost=0.0074,
    )
    result = recommend([big, small], Budget(max_latency_ms=2000))
    assert result.winner is not None and result.winner.config_name == "small"
    assert [c.config_name for c in result.infeasible] == ["big"]


def test_no_config_fits_budget_returns_no_winner() -> None:
    """Must degrade to a clear message, not crash or silently pick something."""
    run = make_run(make_config("only"), modes=[NONE] * 4, latency=1000, cost=0.01)
    result = recommend([run], Budget(max_latency_ms=10))
    assert result.winner is None
    assert result.notes and "budget" in result.notes[0].lower()


def test_rejects_runs_from_different_corpora() -> None:
    a = make_run(make_config("a"), modes=[NONE] * 4, latency=1, cost=1)
    b = make_run(
        make_config("b"), modes=[NONE] * 4, latency=1, cost=1, corpus_id="other"
    )
    with pytest.raises(ComparabilityError, match="multiple corpora"):
        recommend([a, b])


def test_rejects_runs_with_different_golden_set_sizes() -> None:
    """Pass rates over different denominators are not comparable — 3/4 vs 6/12
    look equal but rest on different evidence."""
    a = make_run(make_config("a"), modes=[NONE] * 4, latency=1, cost=1)
    b = make_run(make_config("b"), modes=[NONE] * 12, latency=1, cost=1)
    with pytest.raises(ComparabilityError, match="golden-set sizes"):
        recommend([a, b])


def test_latest_run_per_config_wins() -> None:
    """Re-running a config refines its number; it must not get two rows."""
    old = make_run(
        make_config("dup"), modes=[NONE] * 8, latency=100, cost=0.001, minutes_offset=0
    )
    new = make_run(
        make_config("dup"),
        modes=[F_GEN] * 8,
        latency=100,
        cost=0.001,
        minutes_offset=99,
    )
    assert [r.id for r in latest_per_config([old, new])] == [new.id]
    result = recommend([old, new])
    assert len(result.candidates) == 1
    assert result.candidates[0].pass_rate == 0.0


def test_resolution_points_exposes_small_golden_sets() -> None:
    """With n=4 one query is 25 points, so no smaller difference is real."""
    run = make_run(make_config("a"), modes=[NONE] * 4, latency=1, cost=1)
    assert recommend([run]).resolution_points == 25.0


def test_config_diff_lists_every_differing_dimension() -> None:
    """Confound detection: a winner differing in 4 dimensions cannot have its
    gain attributed to any single one."""
    diff = config_diff(
        make_config("a"),
        make_config("b", retrieval="hybrid", rerank=True, chunk_type="recursive"),
    )
    assert "retrieval.type" in diff
    assert "chunking.type" in diff
    assert "reranker.type" in diff
    assert "name" not in diff  # ignored: renaming isn't a pipeline change
    assert len(diff) > 1


def test_empty_runs_raises() -> None:
    with pytest.raises(ComparabilityError, match="No saved runs"):
        recommend([])
