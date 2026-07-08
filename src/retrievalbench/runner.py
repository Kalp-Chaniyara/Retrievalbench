import statistics
import time
from contextlib import nullcontext
from datetime import datetime

from qdrant_client import models
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from retrievalbench.config import RetrievalConfig
from retrievalbench.eval.metric import evaluate_query
from retrievalbench.generate import build_generator
from retrievalbench.ingest.index import ensure_indexed
from retrievalbench.model import (
    ExperimentRun,
    GoldenItem,
    QueryEvaluation,
    QueryResult,
)
from retrievalbench.retrieval.embedders import build_embedder, build_sparse_embedder
from retrievalbench.retrieval.retrieval import build_retriever
from retrievalbench.storage import RunStore

DEFAULT_JUDGE_MODEL = "gpt-4o-mini"


def _make_run_id(config_name: str, created_at: datetime) -> str:
    return f"{config_name}_{created_at:%Y%m%d_%H%M%S}"


def _aggregate(
    evaluations: list[QueryEvaluation], query_results: list[QueryResult]
) -> dict[str, float]:
    """Run-level means/totals — what `compare` (step 6) reads to diff two runs."""

    def mean(values: list[float]) -> float:
        return statistics.fmean(values) if values else 0.0

    return {
        "faithfulness": mean([e.scores.faithfulness.score for e in evaluations]),
        "answer_relevancy": mean(
            [e.scores.answer_relevancy.score for e in evaluations]
        ),
        "context_precision": mean(
            [e.scores.context_precision.score for e in evaluations]
        ),
        "context_recall": mean([e.scores.context_recall.score for e in evaluations]),
        "mean_latency_ms": mean([r.latency_ms for r in query_results]),
        "total_cost_usd": sum(r.cost_usd for r in query_results),
    }


async def run_experiment(
    config: RetrievalConfig,
    golden_set: list[GoldenItem],
    *,
    corpus_id: str,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    store: RunStore | None = None,
    console: Console | None = None,
) -> ExperimentRun:
    """Run one config over the whole golden set → ExperimentRun (optionally saved).

    Components are built from `config` via the factories, so swapping chunker/
    embedder/retriever/generator is a YAML change, not a code change. The Qdrant
    collection is derived from and populated by the (corpus, chunking, embedding)
    index cache, so retrieval-only variants reuse vectors instead of re-embedding.
    """
    embedder = build_embedder(config.embedding)
    # Sparse encoder only exists for hybrid runs; dense runs never load it.
    sparse_embedder = (
        build_sparse_embedder(config.sparse_embedding)
        if config.retrieval.type == "hybrid"
        else None
    )
    collection = await ensure_indexed(
        corpus_id, config, embedder, sparse_embedder, console=console
    )
    retriever = build_retriever(
        config.retrieval, collection=collection, dim=embedder.dim
    )
    generator = build_generator(config.generation)

    queries = [item.query for item in golden_set]
    # Batch-embed every query in one HTTP call (async hygiene), not per-loop.
    query_vectors = await embedder.embed(queries)
    # Sparse query vectors run in lockstep; None-filled for dense so the loop's
    # strict zip stays uniform. embed_query (not embed): BM25 query weighting.
    if sparse_embedder is not None:
        query_sparse: list[models.SparseVector | None] = list(
            await sparse_embedder.embed_query(queries)
        )
    else:
        query_sparse = [None] * len(golden_set)

    query_results: list[QueryResult] = []
    evaluations: list[QueryEvaluation] = []

    progress = (
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        )
        if console is not None
        else None
    )

    with progress if progress is not None else nullcontext():
        task = (
            progress.add_task(
                f"Running [bold cyan]{config.name}[/bold cyan]",
                total=len(golden_set),
            )
            if progress
            else None
        )

        for item, vector, sparse in zip(
            golden_set, query_vectors, query_sparse, strict=True
        ):
            # latency covers the RAG pipeline (retrieve + generate), not judging.
            started = time.perf_counter()
            retrieved = await retriever.retrieve(
                vector, limit=config.top_k_retrieve, sparse_vector=sparse
            )
            # No reranker yet: narrow to the top_k_final the generator sees.
            context = retrieved[: config.top_k_final]
            answer = await generator.generate(item.query, context)
            latency_ms = (time.perf_counter() - started) * 1000

            scores = await evaluate_query(
                judge_model, item.query, answer, item.expected_answer, context
            )

            query_results.append(
                QueryResult(
                    golden_item_id=item.id,
                    retrieved=retrieved,
                    answer=answer,
                    latency_ms=latency_ms,
                )
            )
            evaluations.append(QueryEvaluation(golden_item_id=item.id, scores=scores))

            if progress and task is not None:
                progress.advance(task)

    created_at = datetime.now()
    run = ExperimentRun(
        id=_make_run_id(config.name, created_at),
        corpus_id=corpus_id,
        config=config,
        query_results=query_results,
        evaluations=evaluations,
        aggregate=_aggregate(evaluations, query_results),
        created_at=created_at,
    )

    if store is not None:
        store.save_run(run)

    return run
