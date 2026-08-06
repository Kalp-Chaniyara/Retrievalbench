"""k/N instrumentation — how much of the corpus enters context per query.

No LLM calls, no Qdrant. This is the measurement that says whether a corpus is
big enough for retrieval to be able to MISS. When k >= N every query retrieves
the whole corpus, `hit_chunk_ids` can never return empty, and F1 is
arithmetically impossible — the diagnostics engine is measuring nothing.

Usage:
    uv run python scripts/corpus_stats.py                    # every corpus
    uv run python scripts/corpus_stats.py sample_data1 techdocs
"""

import sys
from pathlib import Path

from retrievalbench.config import load_config
from retrievalbench.ingest.chunkers import build_chunker
from retrievalbench.ingest.loader import load_corpus

CORPORA = Path("data/corpora")
CONFIGS = Path("configs")


def main(corpus_ids: list[str]) -> int:
    if not corpus_ids:
        corpus_ids = sorted(p.name for p in CORPORA.iterdir() if p.is_dir())

    configs = [load_config(p) for p in sorted(CONFIGS.glob("*.yaml"))]

    for corpus_id in corpus_ids:
        corpus_dir = CORPORA / corpus_id
        if not corpus_dir.is_dir():
            print(f"{corpus_id}: no such corpus at {corpus_dir}")
            continue

        docs = load_corpus(str(corpus_dir))
        print(f"\n{corpus_id} — {len(docs)} documents")
        print(f"  {'config':18s} {'chunks':>7s} {'k':>4s} {'k/N':>8s}")

        for cfg in configs:
            chunker = build_chunker(cfg.chunking)
            chunks = sum(len(chunker.chunk(doc)) for doc in docs)
            k = cfg.top_k_retrieve
            # k > N means every query pulls the entire corpus: retrieval is a
            # no-op and F1 cannot fire. Flagged rather than left to the reader.
            ratio = k / chunks if chunks else float("inf")
            flag = "  <-- k >= N, F1 impossible" if k >= chunks else ""
            print(f"  {cfg.name:18s} {chunks:7d} {k:4d} {ratio:7.2%}{flag}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
