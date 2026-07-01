import asyncio

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from retrievalbench.config import load_config
from retrievalbench.golden import GOLDEN_SET
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
    expected = set(item.expected_chunk_ids)
    scores = evaluation.scores

    # Which chunks came back, in rank order, marked hit/miss vs the golden set.
    chunks = Table(box=None, pad_edge=False, show_header=True, header_style="bold")
    chunks.add_column("", width=3)
    chunks.add_column("rank", justify="right", style="dim")
    chunks.add_column("score", justify="right")
    chunks.add_column("chunk_id")
    for rank, chunk in enumerate(result.retrieved, start=1):
        hit = chunk.chunk_id in expected
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


if __name__ == "__main__":
    app()
