"""Shared factories for building domain objects without touching OpenAI/Qdrant.

Every Tier-1 test runs on FABRICATED data. That is deliberate: these tests check
whether the *logic* is right, not whether the pipeline's numbers are good. The
numbers are chosen to construct a specific scenario (see test_recommend.py,
where the "abstainer" is cheaper AND faster AND has better faithfulness, so only
ranking on pass rate can reject it).
"""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from retrievalbench.config import (
    ChunkingConfig,
    EmbeddingConfig,
    GenerationConfig,
    RerankerConfig,
    RetrievalConfig,
    RetrieverConfig,
)
from retrievalbench.model import (
    Chunk,
    EvalScores,
    ExperimentRun,
    FailureMode,
    GoldenItem,
    MetricScore,
    QueryEvaluation,
    QueryResult,
    RetrievedChunk,
)

BASE_TIME = datetime(2026, 8, 4, 12, 0, 0)
CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "corpora" / "sample_data1"


@pytest.fixture
def corpus_dir() -> Path:
    """The REAL corpus the pipeline runs against — committed, not a copy.

    Using the real thing (rather than a fixture duplicate) is deliberate: the
    golden set's expected_snippets must exist verbatim in THESE files, so a
    copy could drift and let a broken snippet pass CI while real runs fail."""
    return CORPUS_DIR


def make_config(
    name: str = "cfg",
    *,
    retrieval: str = "dense",
    rerank: bool = False,
    chunk_type: str = "fixed",
    size: int = 200,
    overlap: int = 64,
) -> RetrievalConfig:
    return RetrievalConfig(
        name=name,
        chunking=ChunkingConfig(type=chunk_type, size=size, overlap=overlap),
        embedding=EmbeddingConfig(),
        retrieval=RetrieverConfig(type=retrieval),
        reranker=RerankerConfig() if rerank else None,
        generation=GenerationConfig(),
        top_k_retrieve=20,
        top_k_final=5,
    )


def score(value: float) -> MetricScore:
    return MetricScore(score=value, reason="")


def make_run(
    config: RetrievalConfig,
    *,
    modes: list[FailureMode],
    latency: float,
    cost: float,
    faithfulness: float = 0.9,
    minutes_offset: int = 0,
    corpus_id: str = "sample_data1",
) -> ExperimentRun:
    """Assemble the same shape `rbench run` persists, but by hand.

    `modes` drives everything: one entry per golden query, each the failure_mode
    the diagnostics engine would have assigned. len(modes) is the golden-set size.
    """
    evaluations, results = [], []
    for index, mode in enumerate(modes):
        evaluations.append(
            QueryEvaluation(
                golden_item_id=f"q{index}",
                scores=EvalScores(
                    faithfulness=score(faithfulness),
                    answer_relevancy=score(faithfulness),
                    context_precision=score(faithfulness),
                    context_recall=score(faithfulness),
                ),
                failure_mode=mode,
            )
        )
        results.append(
            QueryResult(
                golden_item_id=f"q{index}",
                retrieved=[],
                answer="answer",
                latency_ms=latency,
                cost_usd=cost / len(modes),
            )
        )
    return ExperimentRun(
        id=f"{config.name}_{minutes_offset}",
        corpus_id=corpus_id,
        config=config,
        query_results=results,
        evaluations=evaluations,
        aggregate={
            "faithfulness": faithfulness,
            "answer_relevancy": faithfulness,
            "context_precision": faithfulness,
            "context_recall": faithfulness,
            "mean_latency_ms": latency,
            "total_cost_usd": cost,
        },
        created_at=BASE_TIME + timedelta(minutes=minutes_offset),
    )


def make_retrieved(*texts: str) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            score=1.0 - i / 100,
            chunk_id=f"doc_{i:04d}",
            text=text,
            document_id="doc",
        )
        for i, text in enumerate(texts)
    ]


def make_golden(*snippets: str, item_id: str = "g1") -> GoldenItem:
    return GoldenItem(
        id=item_id,
        query="q",
        expected_snippets=list(snippets),
        expected_answer="expected",
    )


def make_chunk(text: str, index: int = 0) -> Chunk:
    return Chunk(
        id=f"doc_{index:04d}", document_id="doc", text=text, index=index, token_count=1
    )
