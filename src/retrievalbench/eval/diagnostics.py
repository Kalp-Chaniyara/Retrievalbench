import asyncio

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams
from openai import AsyncOpenAI
from pydantic import BaseModel

from retrievalbench.golden import hit_chunk_ids
from retrievalbench.model import FailureMode, GoldenItem, QueryEvaluation, QueryResult

DEFAULT_NOTE_MODEL = "gpt-4o-mini"
CORRECTNESS_THRESHOLD = 0.5

_NOTE_SYSTEM_PROMPT = (
    "You write a one-sentence, plain-language note explaining a RAG failure for "
    "a report reader. The failure stage has already been decided by a separate "
    "deterministic rule — you are never asked to classify it, only to explain it."
)


def _correctness_metric(judge_model: str) -> GEval:
    """The correctness trigger (design §5.10 prerequisite): a GEval check of the
    answer against `expected_answer` — deliberately NOT a RAGAS/DeepEval metric
    threshold, since one low faithfulness/relevancy score on an otherwise-correct
    answer is noise, not a failure. Pinning `judge_model` still leaves judge
    non-determinism at this trigger layer (this is intentional per §5.10); keep
    the model fixed across a run for reproducibility."""
    return GEval(
        name="Correctness",
        criteria=(
            "Determine whether 'actual output' is factually correct and complete "
            "relative to 'expected output'. Paraphrasing and extra harmless detail "
            "are fine; missing key facts, contradictions, or unsupported claims "
            "are not."
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=CORRECTNESS_THRESHOLD,
        model=judge_model,
    )


async def is_failed(
    judge_model: str, query: str, answer: str, expected_answer: str
) -> bool:
    """True if the correctness trigger marks this query failed. `classify_failure`
    (attribution) must only run on queries where this returns True."""
    metric = _correctness_metric(judge_model)
    test_case = LLMTestCase(
        input=query, actual_output=answer, expected_output=expected_answer
    )
    await metric.a_measure(test_case)
    return not metric.is_successful()


def classify_failure(result: QueryResult, item: GoldenItem) -> FailureMode:
    """The deterministic cascade (design §5.10): test F1 first, assign F_GEN only
    if F1 is false — never independent predicates, so a query can't match both.
    Reads `result.retrieved`, the full pre-rerank top_k_retrieve shortlist: F1
    asks whether evidence ever reached the generator at all, not whether
    reranking happened to keep it."""
    if not hit_chunk_ids(result.retrieved, item):
        return FailureMode.RETRIEVAL_MISS
    return FailureMode.GENERATION_FAILURE


async def _write_note(
    client: AsyncOpenAI,
    model: str,
    failure_mode: FailureMode,
    item: GoldenItem,
    result: QueryResult,
) -> str:
    """The LLM writes only the human-readable note — `classify_failure` has
    already fixed the class. The prompt varies by stage so the note explains the
    right thing (missing evidence vs. a wrong answer despite having it)."""
    if failure_mode is FailureMode.RETRIEVAL_MISS:
        stage_hint = (
            "The expected evidence was never retrieved at all (retrieval-stage "
            "failure). Briefly explain why retrieval likely missed it — e.g. an "
            "exact-match/keyword query on a dense-only setup, chunks too small, "
            "or top_k too low — based on the query below."
        )
    else:
        stage_hint = (
            "The expected evidence WAS retrieved, but the generated answer is "
            "still wrong (generation-stage failure). Briefly explain what the "
            "answer got wrong relative to the expected answer."
        )

    user = (
        f"Query: {item.query}\n"
        f"Expected answer: {item.expected_answer}\n"
        f"Generated answer: {result.answer}\n\n"
        f"{stage_hint}\n"
        "Write ONE plain-language sentence. Do not restate the stage label."
    )

    response = await client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": _NOTE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
    )
    return (response.choices[0].message.content or "").strip()


async def diagnose_query(
    client: AsyncOpenAI,
    note_model: str,
    judge_model: str,
    item: GoldenItem,
    result: QueryResult,
) -> tuple[FailureMode, str | None]:
    """One query, gated then attributed then explained. Returns (NONE, None) for
    a query the correctness trigger did not mark failed."""
    if not await is_failed(
        judge_model, item.query, result.answer, item.expected_answer
    ):
        return FailureMode.NONE, None

    failure_mode = classify_failure(result, item)
    note = await _write_note(client, note_model, failure_mode, item, result)
    return failure_mode, note


class DiagnosticsSummary(BaseModel):
    """Aggregate over one run's failures — what `rbench report` prints."""

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


def summarize(evaluations: list[QueryEvaluation]) -> DiagnosticsSummary:
    f1 = sum(1 for e in evaluations if e.failure_mode is FailureMode.RETRIEVAL_MISS)
    f_gen = sum(
        1 for e in evaluations if e.failure_mode is FailureMode.GENERATION_FAILURE
    )
    return DiagnosticsSummary(
        total_queries=len(evaluations),
        failed_count=f1 + f_gen,
        f1_count=f1,
        f_gen_count=f_gen,
    )


async def diagnose_run(
    query_results: list[QueryResult],
    evaluations: list[QueryEvaluation],
    golden_set: list[GoldenItem],
    *,
    judge_model: str,
    note_model: str = DEFAULT_NOTE_MODEL,
) -> tuple[list[QueryEvaluation], DiagnosticsSummary]:
    """Attribute every failed query and summarize. Returns updated
    QueryEvaluation objects (same order as `evaluations`, passing queries
    untouched) plus the aggregate summary; caller decides how to persist/print."""
    golden_by_id = {item.id: item for item in golden_set}
    result_by_id = {r.golden_item_id: r for r in query_results}
    client = AsyncOpenAI()

    # Independent per-query judge calls -> run them concurrently, not in a loop.
    diagnoses = await asyncio.gather(
        *(
            diagnose_query(
                client,
                note_model,
                judge_model,
                golden_by_id[evaluation.golden_item_id],
                result_by_id[evaluation.golden_item_id],
            )
            for evaluation in evaluations
        )
    )

    updated = [
        evaluation.model_copy(
            update={"failure_mode": failure_mode, "diagnosis_note": note}
        )
        for evaluation, (failure_mode, note) in zip(evaluations, diagnoses, strict=True)
    ]

    return updated, summarize(updated)
