"""The wedge: F1 vs F_GEN attribution.

`classify_failure` is the one part of the diagnostics engine with NO LLM in it —
deterministic by design (Design §5.10), which is exactly why it can be gated in
CI for free. The correctness trigger (`is_failed`) is a judge call and is
therefore NOT tested here; it belongs to the Tier-2 eval.
"""

from retrievalbench.cli import failures_by_query_type
from retrievalbench.eval.diagnostics import classify_failure, summarize
from retrievalbench.model import EvalScores, FailureMode, QueryEvaluation, QueryResult

from .conftest import make_golden, make_retrieved, score


def _result(*chunk_texts: str) -> QueryResult:
    return QueryResult(
        golden_item_id="g1",
        retrieved=make_retrieved(*chunk_texts),
        answer="whatever",
        latency_ms=1.0,
    )


def test_f1_when_no_retrieved_chunk_contains_a_snippet() -> None:
    """Evidence never reached the generator -> no prompt change can fix it."""
    item = make_golden("roasted on the next roast day")
    result = _result("something about brewing", "something about returns")
    assert classify_failure(result, item) is FailureMode.RETRIEVAL_MISS


def test_f_gen_when_evidence_was_retrieved() -> None:
    """The cascade's second branch: F1 ruled out -> the failure is downstream."""
    item = make_golden("roasted on the next roast day")
    result = _result("noise", "orders are roasted on the next roast day instead")
    assert classify_failure(result, item) is FailureMode.GENERATION_FAILURE


def test_snippet_match_ignores_whitespace_and_case() -> None:
    """Source docs hard-wrap, so a snippet spans newlines once chunked. Both
    sides are normalised or every real hit would be missed."""
    item = make_golden("roasted on the next roast day")
    result = _result("Orders are ROASTED\non   the\tnext ROAST day.")
    assert classify_failure(result, item) is FailureMode.GENERATION_FAILURE


def test_any_snippet_is_enough_to_pass_the_f1_gate() -> None:
    """expected_snippets is ANY-of: one hit means evidence reached the pipeline."""
    item = make_golden("first fact", "second fact")
    assert classify_failure(_result("only the second fact here"), item) is (
        FailureMode.GENERATION_FAILURE
    )


def test_f1_reads_the_prererank_shortlist() -> None:
    """F1 asks whether evidence EVER reached the pipeline, so it reads
    `retrieved` (top_k_retrieve), not the reranked top_k_final. A snippet the
    reranker dropped is still not a retrieval miss."""
    item = make_golden("the key fact")
    result = QueryResult(
        golden_item_id="g1",
        retrieved=make_retrieved("noise", "the key fact is here"),
        reranked=make_retrieved("noise"),  # reranker discarded the evidence
        answer="wrong",
        latency_ms=1.0,
    )
    assert classify_failure(result, item) is FailureMode.GENERATION_FAILURE


def test_empty_retrieval_is_f1() -> None:
    item = make_golden("anything")
    result = QueryResult(golden_item_id="g1", retrieved=[], answer="", latency_ms=1.0)
    assert classify_failure(result, item) is FailureMode.RETRIEVAL_MISS


def _ev(mode: FailureMode) -> QueryEvaluation:
    s = EvalScores(
        faithfulness=score(1.0),
        answer_relevancy=score(1.0),
        context_precision=score(1.0),
        context_recall=score(1.0),
    )
    return QueryEvaluation(golden_item_id="x", scores=s, failure_mode=mode)


def test_summary_counts_and_shares() -> None:
    evaluations = [
        _ev(FailureMode.NONE),
        _ev(FailureMode.NONE),
        _ev(FailureMode.RETRIEVAL_MISS),
        _ev(FailureMode.GENERATION_FAILURE),
    ]
    summary = summarize(evaluations)
    assert (summary.total_queries, summary.failed_count) == (4, 2)
    assert (summary.f1_count, summary.f_gen_count) == (1, 1)
    # share is of FAILED queries, not all queries
    assert summary.f1_share == 0.5
    assert "2/4 queries failed" in summary.headline


def test_summary_with_no_failures_does_not_divide_by_zero() -> None:
    summary = summarize([_ev(FailureMode.NONE)])
    assert summary.f1_share == 0.0
    assert summary.headline == "0/1 queries failed."


def _ev_for(item_id: str, mode: FailureMode) -> QueryEvaluation:
    return QueryEvaluation(
        golden_item_id=item_id,
        scores=EvalScores(
            faithfulness=score(1.0),
            answer_relevancy=score(1.0),
            context_precision=score(1.0),
            context_recall=score(1.0),
        ),
        failure_mode=mode,
    )


def test_failures_split_by_query_type() -> None:
    """The finding: F1 concentrated in exact_match, absent from semantic. A
    single aggregate F1 rate would average that signal away."""
    golden = {
        "a": make_golden("s", item_id="a"),
        "b": make_golden("s", item_id="b"),
        "c": make_golden("s", item_id="c"),
    }
    golden["a"].query_type = "exact_match"
    golden["b"].query_type = "exact_match"
    golden["c"].query_type = "semantic"

    breakdown = failures_by_query_type(
        [
            _ev_for("a", FailureMode.RETRIEVAL_MISS),
            _ev_for("b", FailureMode.NONE),
            _ev_for("c", FailureMode.NONE),
        ],
        golden,
    )
    assert breakdown["exact_match"][FailureMode.RETRIEVAL_MISS] == 1
    assert breakdown["exact_match"][FailureMode.NONE] == 1
    assert breakdown["semantic"][FailureMode.RETRIEVAL_MISS] == 0
    assert breakdown["semantic"][FailureMode.NONE] == 1


def test_breakdown_skips_evaluations_with_no_matching_golden_item() -> None:
    """A run scored against a golden set that has since changed must not crash
    or invent a type."""
    assert failures_by_query_type([_ev_for("gone", FailureMode.NONE)], {}) == {}


def test_golden_item_query_type_defaults_to_semantic() -> None:
    assert make_golden("s").query_type == "semantic"
