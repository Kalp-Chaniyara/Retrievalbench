import asyncio
from collections.abc import Callable

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from retrievalbench.config import load_config
from retrievalbench.golden import GOLDEN_SET, hit_chunk_ids
from retrievalbench.model import (
    ExperimentRun,
    GoldenItem,
    QueryEvaluation,
    QueryResult,
)
from retrievalbench.runner import run_experiment
from retrievalbench.storage import RunStore

load_dotenv()

app = typer.Typer(help="RetrievalBench CLI")
console = Console()


@app.callback()
def main() -> None:
    """RetrievalBench: a config-driven retrieval-eval harness."""


CORPUS_ID = "sample_data1"  # folder under data/corpora/ + the index-cache key
JUDGE_MODEL = "gpt-4o-mini"


def _color_score(score: float) -> str:
    color = "green" if score >= 0.8 else "yellow" if score >= 0.5 else "red"
    return f"[{color}]{score:.3f}[/{color}]"


def _render_query(
    index: int,
    item: GoldenItem,
    result: QueryResult,
    evaluation: QueryEvaluation,
) -> None:
    """Print one clean block: query, retrieved chunks, answer, metric scores."""
    hits = hit_chunk_ids(result.retrieved, item)  # resolved per config, not hardcoded
    scores = evaluation.scores

    # Which chunks came back, in rank order, marked hit/miss vs the golden set.
    chunks = Table(box=None, pad_edge=False, show_header=True, header_style="bold")
    chunks.add_column("", width=3)
    chunks.add_column("rank", justify="right", style="dim")
    chunks.add_column("score", justify="right")
    chunks.add_column("chunk_id")
    for rank, chunk in enumerate(result.retrieved, start=1):
        hit = chunk.chunk_id in hits
        chunks.add_row(
            "[green]✓[/green]" if hit else "[red]✗[/red]",
            str(rank),
            f"{chunk.score:.3f}",
            f"[green]{chunk.chunk_id}[/green]" if hit else chunk.chunk_id,
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
    body.add_row("[bold]Retrieved (top-k)[/bold]")
    body.add_row(chunks)
    body.add_row("")
    body.add_row(f"[bold]Answer[/bold] [dim]({result.latency_ms:.0f} ms)[/dim]")
    body.add_row(result.answer)
    body.add_row("")
    body.add_row("[bold]Scores[/bold]")
    body.add_row(metrics)

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
    golden_by_id = {item.id: item for item in GOLDEN_SET}
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
            GOLDEN_SET,
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
