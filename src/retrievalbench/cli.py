import asyncio
import statistics

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from retrievalbench.eval.metric import EvalScores, evaluate_query
from retrievalbench.generate import OpenAIGenerator
from retrievalbench.golden import GOLDEN_SET
from retrievalbench.model import GoldenItem, RetrievedChunk
from retrievalbench.retrieval.embedders import OpenAITextEmbedderSmall
from retrievalbench.retrieval.retrieval import QdrantRetrieval

load_dotenv()

app = typer.Typer(help="RetrievalBench CLI")
console = Console()


@app.callback()
def main() -> None:
    """RetrievalBench: a config-driven retrieval-eval harness."""


COLLECTION = "sample_data1"
JUDGE_MODEL = "gpt-4o-mini"
TOP_K = 5


def _render_query(
    index: int,
    item: GoldenItem,
    answer: str,
    retrieved: list[RetrievedChunk],
    scores: EvalScores,
) -> None:
    """Print one clean block: query, retrieved chunks, answer, metric scores."""
    expected = set(item.expected_chunk_ids)

    # Which chunks came back, in rank order, marked hit/miss vs the golden set.
    chunks = Table(box=None, pad_edge=False, show_header=True, header_style="bold")
    chunks.add_column("", width=3)
    chunks.add_column("rank", justify="right", style="dim")
    chunks.add_column("score", justify="right")
    chunks.add_column("chunk_id")
    for rank, chunk in enumerate(retrieved, start=1):
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
    body.add_row(f"[bold]Answer[/bold]\n{answer}")
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


def _color_score(score: float) -> str:
    color = "green" if score >= 0.8 else "yellow" if score >= 0.5 else "red"
    return f"[{color}]{score:.3f}[/{color}]"


async def _run() -> None:
    embedder = OpenAITextEmbedderSmall()
    retriever = QdrantRetrieval(collection=COLLECTION, dim=embedder.dim)
    generator = OpenAIGenerator()

    # Batch-embed every golden query in one HTTP call (async hygiene).
    query_vectors = await embedder.embed([item.query for item in GOLDEN_SET])

    faithfulness: list[float] = []
    answer_relevancy: list[float] = []
    context_precision: list[float] = []
    context_recall: list[float] = []

    for i, (item, vector) in enumerate(
        zip(GOLDEN_SET, query_vectors, strict=True), start=1
    ):
        retrieved = await retriever.dense_search(vector, limit=TOP_K)
        answer = await generator.generate(item.query, retrieved)
        scores = await evaluate_query(
            JUDGE_MODEL, item.query, answer, item.expected_answer, retrieved
        )

        faithfulness.append(scores.faithfulness.score)
        answer_relevancy.append(scores.answer_relevancy.score)
        context_precision.append(scores.context_precision.score)
        context_recall.append(scores.context_recall.score)

        _render_query(i, item, answer, retrieved, scores)

    _render_summary(faithfulness, answer_relevancy, context_precision, context_recall)


def _render_summary(
    faithfulness: list[float],
    answer_relevancy: list[float],
    context_precision: list[float],
    context_recall: list[float],
) -> None:
    table = Table(title=f"Phase 0 — means over {len(faithfulness)} queries")
    table.add_column("metric", style="cyan")
    table.add_column("mean", justify="right")
    for name, values in (
        ("faithfulness", faithfulness),
        ("answer_relevancy", answer_relevancy),
        ("context_precision", context_precision),
        ("context_recall", context_recall),
    ):
        table.add_row(name, _color_score(statistics.fmean(values)))

    console.print()
    console.print(table)


@app.command()
def run() -> None:
    """Run the golden set end-to-end and print the four mean RAG metrics."""
    asyncio.run(_run())
    # _run()


if __name__ == "__main__":
    app()
