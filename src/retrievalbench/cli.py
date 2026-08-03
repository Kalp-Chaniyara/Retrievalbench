import asyncio
from collections.abc import Callable

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from retrievalbench.config import load_config
from retrievalbench.eval.diagnostics import summarize
from retrievalbench.golden import (
    DEFAULT_GENERATOR_MODEL,
    GOLDEN_SET,
    candidate_to_golden_item,
    generate_candidates,
    hit_chunk_ids,
)
from retrievalbench.ingest.chunkers import build_chunker
from retrievalbench.ingest.index import CORPORA_DIR
from retrievalbench.ingest.loader import load_corpus
from retrievalbench.model import (
    Chunk,
    ExperimentRun,
    FailureMode,
    GoldenItem,
    QueryEvaluation,
    QueryResult,
    RetrievedChunk,
)
from retrievalbench.runner import run_experiment
from retrievalbench.storage import GoldenStore, RunStore

load_dotenv()

app = typer.Typer(help="RetrievalBench CLI")
console = Console()


@app.callback()
def main() -> None:
    """RetrievalBench: a config-driven retrieval-eval harness."""


CORPUS_ID = "sample_data1"  # folder under data/corpora/ + the index-cache key
# Judge only — see runner.DEFAULT_JUDGE_MODEL for why this one is not the mini.
JUDGE_MODEL = "gpt-4o"


def _golden_set(corpus_id: str) -> list[GoldenItem]:
    """The effective golden set for a corpus: the hand-written, git-versioned
    GOLDEN_SET literal plus whatever `rbench gen-golden` has generated and a
    human kept for this corpus (GoldenStore). Ids never collide — generated
    items are 'gen_'-prefixed — so this is a plain concatenation."""
    store = GoldenStore()
    return GOLDEN_SET + store.load_golden_set(corpus_id)


def _color_score(score: float) -> str:
    color = "green" if score >= 0.8 else "yellow" if score >= 0.5 else "red"
    return f"[{color}]{score:.3f}[/{color}]"


# Badge styling per failure mode — shared by the per-query panel and `report`.
_FAILURE_STYLE: dict[FailureMode, tuple[str, str]] = {
    FailureMode.NONE: ("PASS", "green"),
    FailureMode.RETRIEVAL_MISS: ("F1 · retrieval miss", "red"),
    FailureMode.GENERATION_FAILURE: ("F_GEN · generation failure", "yellow"),
}


def _failure_badge(mode: FailureMode) -> str:
    label, color = _FAILURE_STYLE.get(mode, (mode.value, "magenta"))
    return f"[bold {color}]{label}[/bold {color}]"


def _chunk_table(chunks: list[RetrievedChunk], hits: set[str]) -> Table:
    """One rank-ordered chunk list, each row marked hit/miss vs the golden set.
    `hits` is computed per list (snippet-based), so the same helper renders both
    the pre-rerank shortlist and the reranked context with correct ✓/✗."""
    table = Table(box=None, pad_edge=False, show_header=True, header_style="bold")
    table.add_column("", width=3)
    table.add_column("rank", justify="right", style="dim")
    table.add_column("score", justify="right")
    table.add_column("chunk_id")
    for rank, chunk in enumerate(chunks, start=1):
        hit = chunk.chunk_id in hits
        table.add_row(
            "[green]✓[/green]" if hit else "[red]✗[/red]",
            str(rank),
            f"{chunk.score:.3f}",
            f"[green]{chunk.chunk_id}[/green]" if hit else chunk.chunk_id,
        )
    return table


def _render_query(
    index: int,
    item: GoldenItem,
    result: QueryResult,
    evaluation: QueryEvaluation,
) -> None:
    """Print one clean block: query, retrieved chunks, answer, metric scores."""
    scores = evaluation.scores

    # The pre-rerank shortlist (F1 gate) and, when a reranker ran, the reranked
    # top_k_final the generator actually answered from (F2/F3 gate). Hits are
    # snippet-based and computed per list, so ✓/✗ is correct in each table.
    retrieved_hits = hit_chunk_ids(result.retrieved, item)
    retrieved_table = _chunk_table(result.retrieved, retrieved_hits)
    reranked_table = (
        _chunk_table(result.reranked, hit_chunk_ids(result.reranked, item))
        if result.reranked is not None
        else None
    )

    # The four metrics with the judge's reason underneath each one.
    metrics = Table(box=None, pad_edge=False, show_header=True, header_style="bold")
    metrics.add_column("metric", style="cyan")
    metrics.add_column("score", justify="right")
    metrics.add_column("reason", overflow="fold")
    for name, ms in (
        ("faithfulness", scores.faithfulness),
        ("answer_relevancy", scores.answer_relevancy),
        ("context_precision", scores.context_precision),
        ("context_recall", scores.context_recall),
    ):
        metrics.add_row(name, _color_score(ms.score), f"[dim]{ms.reason}[/dim]")

    body = Table.grid(padding=(0, 0))
    body.add_row("[bold]Retrieved (shortlist)[/bold]")
    body.add_row(retrieved_table)
    if reranked_table is not None:
        body.add_row("")
        body.add_row(f"[bold]Reranked → context (top {len(result.reranked)})[/bold]")
        body.add_row(reranked_table)
    body.add_row("")
    body.add_row(f"[bold]Answer[/bold] [dim]({result.latency_ms:.0f} ms)[/dim]")
    body.add_row(result.answer)
    body.add_row("")
    body.add_row("[bold]Scores[/bold]")
    body.add_row(metrics)
    if evaluation.failure_mode is not FailureMode.NONE:
        body.add_row("")
        body.add_row(
            f"[bold]Diagnosis[/bold] {_failure_badge(evaluation.failure_mode)}"
        )
        if evaluation.diagnosis_note:
            body.add_row(f"[dim]{evaluation.diagnosis_note}[/dim]")

    console.print(
        Panel(
            body,
            title=f"[bold]Q{index}[/bold]  {item.query}",
            title_align="left",
            border_style="blue",
        )
    )


def _render_summary(run: ExperimentRun) -> None:
    agg = run.aggregate
    table = Table(
        title=f"{run.config.name} — means over {len(run.query_results)} queries"
    )
    table.add_column("metric", style="cyan")
    table.add_column("mean", justify="right")
    for name in (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ):
        table.add_row(name, _color_score(agg[name]))
    table.add_row("mean_latency_ms", f"{agg['mean_latency_ms']:.0f}")

    console.print()
    console.print(table)


def _render_run(run: ExperimentRun) -> None:
    golden_by_id = {item.id: item for item in _golden_set(run.corpus_id)}
    eval_by_id = {ev.golden_item_id: ev for ev in run.evaluations}
    for index, result in enumerate(run.query_results, start=1):
        item = golden_by_id[result.golden_item_id]
        _render_query(index, item, result, eval_by_id[result.golden_item_id])
    _render_summary(run)


@app.command()
def run(
    config: str = typer.Option(
        "configs/baseline.yaml", "--config", "-c", help="Experiment YAML."
    ),
) -> None:
    """Run one config over the golden set, score it, persist it, and print results."""
    cfg = load_config(config)
    store = RunStore()
    experiment = asyncio.run(
        run_experiment(
            cfg,
            _golden_set(CORPUS_ID),
            corpus_id=CORPUS_ID,
            judge_model=JUDGE_MODEL,
            store=store,
            console=console,
        )
    )
    _render_run(experiment)
    console.print(
        f"\n[dim]saved run[/dim] [bold cyan]{experiment.id}[/bold cyan] "
        f"[dim]→ {store.path}[/dim]"
    )


def _render_report(run: ExperimentRun) -> None:
    """Per-query failure table + aggregate headline — the diagnosis (design
    §5.10). Recomputed from `run.evaluations` rather than stored, since the
    summary is a pure aggregation of already-persisted failure_mode values."""
    golden_by_id = {item.id: item for item in _golden_set(run.corpus_id)}

    table = Table(title=f"{run.config.name} — diagnosis")
    table.add_column("query", overflow="fold")
    table.add_column("status")
    table.add_column("note", overflow="fold")
    for evaluation in run.evaluations:
        item = golden_by_id.get(evaluation.golden_item_id)
        query = item.query if item is not None else evaluation.golden_item_id
        table.add_row(
            query,
            _failure_badge(evaluation.failure_mode),
            evaluation.diagnosis_note or "[dim]—[/dim]",
        )

    summary = summarize(run.evaluations)
    console.print()
    console.print(table)
    console.print()
    console.print(Panel(summary.headline, border_style="magenta", title="Summary"))


@app.command()
def report(
    run_id: str = typer.Argument(..., help="Saved run id (see `rbench run` output)."),
) -> None:
    """Print the per-query failure diagnosis + aggregate summary for a saved run."""
    store = RunStore()
    run = store.get_run(run_id)
    if run is None:
        console.print(f"[red]run not found:[/red] {run_id}")
        _print_available_runs(store)
        raise typer.Exit(code=1)
    _render_report(run)


def _render_candidate(
    index: int,
    total: int,
    chunk: Chunk,
    question: str,
    expected_answer: str,
    snippets: list[str],
) -> None:
    body = Table.grid(padding=(0, 0))
    body.add_row(f"[bold]Source chunk[/bold] [dim]({chunk.id})[/dim]")
    body.add_row(chunk.text)
    body.add_row("")
    body.add_row(f"[bold]Question[/bold]  {question}")
    body.add_row(f"[bold]Expected answer[/bold]  {expected_answer}")
    body.add_row(f"[bold]Snippets[/bold]  {snippets}")
    console.print(
        Panel(
            body,
            title=f"[bold]Candidate {index}/{total}[/bold]",
            title_align="left",
            border_style="cyan",
        )
    )


@app.command("gen-golden")
def gen_golden(
    config: str = typer.Option(
        "configs/baseline.yaml",
        "--config",
        "-c",
        help="Chunking config to sample from.",
    ),
    corpus_id: str = typer.Option(
        CORPUS_ID, "--corpus-id", help="Corpus to generate from."
    ),
    n: int = typer.Option(
        10, "--n", help="Number of chunks to sample and generate candidates from."
    ),
    seed: int = typer.Option(42, "--seed", help="Sampling seed (reproducibility, G4)."),
    model: str = typer.Option(
        DEFAULT_GENERATOR_MODEL, "--model", help="Generator model."
    ),
) -> None:
    """Generate candidate golden items from the corpus, review each one
    (keep/edit/drop), and persist kept items to the corpus's stored golden set."""
    cfg = load_config(config)
    corpus_dir = CORPORA_DIR / corpus_id
    documents = load_corpus(str(corpus_dir))
    chunker = build_chunker(cfg.chunking)
    chunks: list[Chunk] = [c for doc in documents for c in chunker.chunk(doc)]
    if not chunks:
        console.print(f"[red]no chunks found under {corpus_dir}[/red]")
        raise typer.Exit(code=1)

    console.print(
        f"[dim]generating up to {n} candidate(s) from {len(chunks)} chunks…[/dim]"
    )
    candidates = asyncio.run(generate_candidates(chunks, model=model, n=n, seed=seed))
    console.print(
        f"[dim]{len(candidates)}/{n} candidates passed verbatim validation[/dim]\n"
    )

    kept: list[GoldenItem] = []
    for index, (chunk, candidate) in enumerate(candidates, start=1):
        question, expected_answer = candidate.question, candidate.expected_answer
        _render_candidate(
            index,
            len(candidates),
            chunk,
            question,
            expected_answer,
            candidate.expected_snippets,
        )

        choice = Prompt.ask(
            "[k]eep / [e]dit / [d]rop", choices=["k", "e", "d"], default="k"
        )
        if choice == "d":
            continue
        if choice == "e":
            question = Prompt.ask("Question", default=question)
            expected_answer = Prompt.ask("Expected answer", default=expected_answer)

        kept.append(
            candidate_to_golden_item(
                candidate.model_copy(
                    update={"question": question, "expected_answer": expected_answer}
                )
            )
        )

    if not kept:
        console.print("\n[dim]no items kept.[/dim]")
        return

    store = GoldenStore()
    existing = store.load_golden_set(corpus_id)
    store.save_golden_set(corpus_id, existing + kept)
    total = len(existing) + len(kept)
    console.print(
        f"\n[green]saved {len(kept)} item(s)[/green] to the golden set for "
        f"[cyan]{corpus_id}[/cyan] [dim](total stored: {total})[/dim]"
    )


# Which aggregate keys `compare` diffs, and how to read each: `higher_is_better`
# flips the colour logic (a lower latency/cost is an *improvement*, not a loss);
# `fmt` renders both the value and the delta in that metric's own units.
_COMPARE_METRICS: list[tuple[str, bool, Callable[[float], str]]] = [
    ("faithfulness", True, lambda v: f"{v:.3f}"),
    ("answer_relevancy", True, lambda v: f"{v:.3f}"),
    ("context_precision", True, lambda v: f"{v:.3f}"),
    ("context_recall", True, lambda v: f"{v:.3f}"),
    ("mean_latency_ms", False, lambda v: f"{v:.0f} ms"),
    ("total_cost_usd", False, lambda v: f"${v:.4f}"),
]


def _config_desc(run: ExperimentRun) -> str:
    """One-line "what differs" summary so A vs B isn't just two opaque ids."""
    c = run.config
    return (
        f"{c.chunking.type}/{c.chunking.size}/{c.chunking.overlap} · "
        f"{c.retrieval.type} · {c.embedding.type}"
    )


def _delta_cell(
    a: float, b: float, higher_is_better: bool, fmt: Callable[[float], str]
) -> str:
    """Signed delta B−A, green when it's an improvement, red when a regression.
    Improvement direction is per-metric: up is good for scores, down for
    latency/cost."""
    delta = b - a
    if abs(delta) < 1e-9:
        return "[dim]—[/dim]"
    improved = (delta > 0) == higher_is_better
    color = "green" if improved else "red"
    sign = "+" if delta > 0 else "-"
    return f"[{color}]{sign}{fmt(abs(delta))}[/{color}]"


def _print_available_runs(store: RunStore) -> None:
    """Shown on a not-found so the user can copy a real id instead of guessing."""
    rows = store.list_runs()
    if not rows:
        console.print("[dim]no saved runs yet — run `rbench run` first.[/dim]")
        return
    table = Table(title="available runs")
    table.add_column("id", style="cyan")
    table.add_column("config")
    table.add_column("created_at", style="dim")
    for run_id, config_name, created_at in rows:
        table.add_row(run_id, config_name, created_at)
    console.print(table)


def _render_compare(run_a: ExperimentRun, run_b: ExperimentRun) -> None:
    table = Table(title="compare  (Δ = B − A)")
    table.add_column("metric", style="cyan")
    table.add_column(f"A · {run_a.config.name}", justify="right")
    table.add_column(f"B · {run_b.config.name}", justify="right")
    table.add_column("Δ", justify="right")
    for key, higher_is_better, fmt in _COMPARE_METRICS:
        a = run_a.aggregate[key]
        b = run_b.aggregate[key]
        table.add_row(key, fmt(a), fmt(b), _delta_cell(a, b, higher_is_better, fmt))

    def header(label: str, run: ExperimentRun) -> str:
        return (
            f"[bold]{label}[/bold] [cyan]{run.id}[/cyan] "
            f"[dim]({_config_desc(run)})[/dim]"
        )

    console.print()
    console.print(header("A", run_a))
    console.print(header("B", run_b))
    console.print()
    console.print(table)


@app.command()
def compare(
    run_a: str = typer.Argument(..., help="Baseline run id (A)."),
    run_b: str = typer.Argument(..., help="Candidate run id (B)."),
) -> None:
    """Diff two saved runs on every metric: Δ = B − A, coloured by improvement."""
    store = RunStore()
    a = store.get_run(run_a)
    b = store.get_run(run_b)
    if a is None or b is None:
        missing = ", ".join(rid for rid, run in ((run_a, a), (run_b, b)) if run is None)
        console.print(f"[red]run(s) not found:[/red] {missing}")
        _print_available_runs(store)
        raise typer.Exit(code=1)
    _render_compare(a, b)


if __name__ == "__main__":
    app()
