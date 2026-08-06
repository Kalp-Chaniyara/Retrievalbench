from datetime import datetime
from enum import StrEnum
from typing import Literal

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


# One place defining the allowed query kinds — GoldenItem, golden.py and the
# gen-golden review prompt all read from here so they cannot drift apart.
QueryType = Literal["exact_match", "semantic", "negation", "multi_hop"]
QUERY_TYPES: tuple[str, ...] = ("exact_match", "semantic", "negation", "multi_hop")


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
    # What KIND of retrieval this query stresses. The per-type F1 breakdown is
    # the actual finding ("F1 fires on 40% of exact_match and 0% of semantic");
    # a single aggregate F1 rate hides which retrieval mode is failing.
    # Defaults to "semantic" so GoldenStore rows written before this field
    # deserialize without a migration.
    query_type: QueryType = "semantic"


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
    """Canonical RAG failure modes (the wedge). Shipped two-class engine
    (design §5.10): NONE / RETRIEVAL_MISS (F1) / GENERATION_FAILURE (F_GEN),
    assigned by eval/diagnostics.py. F2/F3/ABSTAIN are the deferred F2/F3 split
    (§5.10.1, post-ship-gate) — declared now so persistence needs no later
    migration, but nothing assigns them yet."""

    NONE = "none"  # query passed (not a failure)
    RETRIEVAL_MISS = "f1"  # expected evidence never retrieved — deterministic
    GENERATION_FAILURE = "f_gen"  # evidence retrieved but answer wrong — unattributed
    # --- Deferred: sub-classes of GENERATION_FAILURE (§5.10.1) ---
    GENERATION_IGNORE = "f2"  # retrieved but the answer ignored it
    GENERATION_ERROR = "f3"  # used the context but still answered wrong
    ABSTAIN = "abstain"  # failed but unattributable (F2/F3 split only)


class QueryResult(BaseModel):
    """What retrieval + generation produced for one golden query."""

    golden_item_id: str
    # Full retrieval output (top_k_retrieve), pre-rerank — the F1 (retrieval-miss)
    # gate reads this: a snippet missing here means it was never retrieved at all.
    retrieved: list[RetrievedChunk]
    # The reranked top_k_final the generator actually saw, or None when no
    # reranker is configured (then the generator saw retrieved[:top_k_final]).
    # F2/F3 gate on the context the generator saw, so they read this when set.
    reranked: list[RetrievedChunk] | None = None
    answer: str
    latency_ms: float = 0.0
    # PIPELINE cost only — the generation call for this query, priced from the
    # response's token usage. Deliberately excludes the judge/diagnostics spend:
    # that is measurement overhead you don't pay in production, so folding it in
    # would make a pipeline look pricier just for switching judge models.
    # Also excludes indexing (one-time, amortized) and the local reranker (no $,
    # but its real cost — latency — is already in latency_ms).
    cost_usd: float = 0.0


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


class DiagnosticsSummary(BaseModel):
    """Aggregate over one run's failures — what `rbench report` prints.
    Computed by eval/diagnostics.summarize()."""

    total_queries: int
    failed_count: int
    f1_count: int
    f_gen_count: int

    @property
    def f1_share(self) -> float:
        """Share of FAILED queries (not all queries) attributed to F1."""
        return self.f1_count / self.failed_count if self.failed_count else 0.0

    @property
    def headline(self) -> str:
        if self.failed_count == 0:
            return f"0/{self.total_queries} queries failed."
        return (
            f"{self.failed_count}/{self.total_queries} queries failed — "
            f"{self.f1_share:.0%} are F1 (retrieval miss), "
            f"{1 - self.f1_share:.0%} are F_GEN (generation failure)."
        )


class Budget(BaseModel):
    """Constraints a config must satisfy to be recommendable. Both optional —
    an empty Budget means "rank everything, exclude nothing"."""

    max_latency_ms: float | None = Field(default=None, gt=0)
    max_cost_usd: float | None = Field(default=None, gt=0)

    def allows(self, candidate: "ConfigCandidate") -> bool:
        if self.max_latency_ms is not None and (
            candidate.mean_latency_ms > self.max_latency_ms
        ):
            return False
        if self.max_cost_usd is not None and (
            candidate.cost_per_run_usd > self.max_cost_usd
        ):
            return False
        return True


class ConfigCandidate(BaseModel):
    """One config's measured result, rolled up to the axes we rank on."""

    run_id: str
    config_name: str
    config: RetrievalConfig
    total_queries: int
    passed: int
    f1_count: int
    f_gen_count: int
    mean_latency_ms: float
    cost_per_run_usd: float
    faithfulness: float  # secondary colour only — never the ranking key

    @property
    def pass_rate(self) -> float:
        """0..1. The ranking key: correctness, not metric means."""
        return self.passed / self.total_queries if self.total_queries else 0.0

    @property
    def failed(self) -> int:
        return self.f1_count + self.f_gen_count


class Recommendation(BaseModel):
    """What `rbench recommend` renders. Pure data — the CLI does the printing."""

    corpus_id: str
    golden_set_size: int
    candidates: list[ConfigCandidate]  # ranked best-first (feasible only)
    infeasible: list[ConfigCandidate] = Field(default_factory=list)
    winner: ConfigCandidate | None = None
    reference: ConfigCandidate | None = None  # cheapest feasible = "do nothing"
    dominated: list[str] = Field(default_factory=list)
    # Winner vs reference. Quality in POINTS (a 50%->83% move is +33 points, not
    # +66%); cost/latency as ratios, where a multiplier is the honest unit.
    quality_gain_points: float | None = None
    cost_ratio: float | None = None
    latency_ratio: float | None = None
    diminishing_returns: bool = False
    confounded_dimensions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def resolution_points(self) -> float:
        """Smallest pass-rate change the golden set can even express. With n=4
        one query flipping is 25 points, so any smaller 'gain' is noise."""
        return 100.0 / self.golden_set_size if self.golden_set_size else 0.0
