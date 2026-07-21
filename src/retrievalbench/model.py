from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from retrievalbench.config import RetrievalConfig


class Document(BaseModel):
    id: str
    source_path: str
    title: str | None
    text: str | None
    metadata: dict[str, str] = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str
    document_id: str
    text: str
    index: int  # order within the document
    token_count: int
    metadata: dict[str, str] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    score: float
    chunk_id: str
    text: str
    document_id: str
    metadata: dict[str, str] = Field(default_factory=dict)


class GoldenItem(BaseModel):
    id: str  # stable id so QueryResult/QueryEvaluation can link back here
    query: str
    # Retrieval ground truth as verbatim, answer-bearing SOURCE snippets — not
    # chunk ids. A chunk_id encodes position (`docid_0007`), which shifts with
    # chunk size, so it can't be ground truth for a benchmark that compares
    # chunking configs. Source text is config-stable: a chunk is a retrieval
    # "hit" if it contains any of these snippets, so the matching chunk id is
    # resolved per config at eval time (see golden.chunk_matches_snippets).
    # This drives the F1 (retrieval-miss) gate; F2/F3 are gated on it.
    expected_snippets: list[str]
    expected_answer: str


class MetricScore(BaseModel):
    """One metric's score (0..1) plus the judge's human-readable reason."""

    score: float
    reason: str


class EvalScores(BaseModel):
    """All four RAG metrics for a single query."""

    faithfulness: MetricScore
    answer_relevancy: MetricScore
    context_precision: MetricScore
    context_recall: MetricScore


class FailureMode(StrEnum):
    """Canonical RAG failure modes (the wedge). Logic that assigns these is
    Phase 2; the field exists now so persistence needs no later migration."""

    NONE = "none"  # query passed
    RETRIEVAL_MISS = "f1"  # right chunk never retrieved
    GENERATION_IGNORE = "f2"  # retrieved but the answer ignored it
    GENERATION_ERROR = "f3"  # used the context but still answered wrong


class QueryResult(BaseModel):
    """What retrieval + generation produced for one golden query."""

    golden_item_id: str
    retrieved: list[RetrievedChunk]
    answer: str
    latency_ms: float = 0.0
    cost_usd: float = 0.0  # 0 until the generator returns token usage


class QueryEvaluation(BaseModel):
    """The scores for one query. `scores` reuses EvalScores (keeps the judge's
    reasons); failure_mode/diagnosis_note stay empty until Phase 2."""

    golden_item_id: str
    scores: EvalScores
    failure_mode: FailureMode = FailureMode.NONE
    diagnosis_note: str | None = None


class ExperimentRun(BaseModel):
    """One config executed over the whole golden set — the unit of comparison."""

    id: str
    corpus_id: str
    config: RetrievalConfig
    query_results: list[QueryResult]
    evaluations: list[QueryEvaluation]
    aggregate: dict[str, float]  # mean metrics / totals — what `compare` reads
    created_at: datetime
