# scratch.py  (throwaway, don't commit)
# End-to-end ingest: load -> chunk -> embed -> store, then dump chunk ids.
import asyncio
from pathlib import Path

from retrievalbench.ingest.chunkers import FixedSizeChunker
from retrievalbench.ingest.loader import load_corpus
from retrievalbench.retrieval.embedders import OpenAITextEmbedderSmall
from retrievalbench.retrieval.store import QdrantStore

CORPUS_DIR = "data/corpora/sample_data1"
COLLECTION = "sample_data1"
DUMP_PATH = "data/corpora/sample_data1_chunks.md"
EMBED_BATCH = 100


async def main() -> None:
    # 1. Load documents
    docs = load_corpus(CORPUS_DIR)
    print(f"loaded {len(docs)} docs from {CORPUS_DIR}")

    # 2. Chunk every document into one flat list
    chunker = FixedSizeChunker(size=300, overlap=50)
    chunks = []
    for d in docs:
        doc_chunks = chunker.chunk(d)
        chunks.extend(doc_chunks)
        print(f"  {d.title}: {len(doc_chunks)} chunks")
    print(f"total chunks: {len(chunks)}")

    # 3. Embed in batches (one HTTP call per batch, not per chunk)
    embedder = OpenAITextEmbedderSmall()
    vectors: list[list[float]] = []
    for i in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[i : i + EMBED_BATCH]
        vectors.extend(await embedder.embed([c.text for c in batch]))
    print(f"embedded {len(vectors)} chunks (dim={embedder.dim})")

    # 4. Store in Qdrant (creates the collection if missing)
    store = QdrantStore(collection=COLLECTION, dim=embedder.dim)
    await store.upsert(chunks, vectors)
    print(f"upserted {len(chunks)} points into collection '{COLLECTION}'")

    # 5. Dump chunk_id + full text to a file so you can write golden items
    lines = ["# Chunk dump for sample_data1\n"]
    for c in chunks:
        lines.append(f"## {c.id}  (doc={c.document_id}, tokens={c.token_count})\n")
        lines.append(c.text.strip() + "\n")
    Path(DUMP_PATH).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote chunk dump -> {DUMP_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
