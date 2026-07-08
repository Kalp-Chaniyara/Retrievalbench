from pathlib import Path

from rich.console import Console

from retrievalbench.config import (
    ChunkingConfig,
    EmbeddingConfig,
    RetrievalConfig,
    SparseEmbeddingConfig,
)
from retrievalbench.ingest.chunkers import build_chunker
from retrievalbench.ingest.loader import load_corpus
from retrievalbench.model import Chunk
from retrievalbench.retrieval.embedders import Embedder, SparseEmbedder
from retrievalbench.retrieval.store import QdrantHybridStore, QdrantStore

# Corpora live at the project root under data/corpora/<corpus_id>/ (gitignored).
# corpus_id doubles as the folder name, so it must match the directory on disk.
CORPORA_DIR = Path("data/corpora")


def collection_name(
    corpus_id: str,
    chunking: ChunkingConfig,
    embedding: EmbeddingConfig,
    sparse: SparseEmbeddingConfig | None = None,
) -> str:
    """The index-cache key = exactly the inputs that determine the stored
    vectors: corpus + chunking + embedding (+ sparse encoder for hybrid). Same
    key -> reuse the collection (retrieval/rerank/generation variants never
    trigger a re-embed); change any input -> a different name, so two chunkings
    can never pollute one collection.

    The sparse suffix is load-bearing: a hybrid collection has a different
    schema (named dense + sparse vectors) than a dense one, so they MUST get
    different names or a dense-built collection would answer sparse queries with
    nothing. A dense config (sparse=None) keeps the old, unsuffixed name.

    Deliberately NOT keyed on corpus *content*: editing/adding/deleting files
    won't invalidate the cache yet — re-index by hand for now (deferred).
    """
    name = (
        f"{corpus_id}__{chunking.type}_{chunking.size}_{chunking.overlap}"
        f"__{embedding.type}"
    )
    if sparse is not None:
        name += f"__{sparse.type}"
    return name


async def ensure_indexed(
    corpus_id: str,
    config: RetrievalConfig,
    embedder: Embedder,
    sparse_embedder: SparseEmbedder | None = None,
    *,
    console: Console | None = None,
) -> str:
    """Guarantee the (corpus, chunking, embedding[, sparse]) collection exists
    and is populated, indexing it exactly once on a cache miss. Returns the
    collection name for the retriever to query.

    `embedder` is passed in rather than built here so chunks and queries share
    the *same* embedder instance — dense search only compares vectors living in
    one shared space (mixing embedders makes distances meaningless). For hybrid
    runs a `sparse_embedder` is passed too, and every chunk is stored with BOTH
    a dense and a sparse vector — sparse search returns nothing unless the
    stored chunks carry sparse vectors, not just the query.
    """
    hybrid = config.retrieval.type == "hybrid"
    if hybrid and sparse_embedder is None:
        raise ValueError("hybrid indexing requires a sparse_embedder")

    name = collection_name(
        corpus_id,
        config.chunking,
        config.embedding,
        config.sparse_embedding if hybrid else None,
    )
    store: QdrantStore | QdrantHybridStore = (
        QdrantHybridStore(name, embedder.dim)
        if hybrid
        else QdrantStore(name, embedder.dim)
    )

    # Cache hit: this exact (corpus, chunking, embedding[, sparse]) is indexed.
    if await store.is_populated():
        if console:
            console.print(f"[dim]index cache hit[/dim] [cyan]{name}[/cyan]")
        return name

    # Cache miss: load -> chunk -> embed -> upsert, once.
    corpus_dir = CORPORA_DIR / corpus_id
    documents = load_corpus(str(corpus_dir))
    chunker = build_chunker(config.chunking)
    chunks: list[Chunk] = [chunk for doc in documents for chunk in chunker.chunk(doc)]
    if not chunks:
        raise ValueError(
            f"No chunks produced from {corpus_dir} — is the corpus directory "
            f"present and holding .pdf/.md/.txt files?"
        )

    if console:
        console.print(
            f"[dim]indexing[/dim] [cyan]{name}[/cyan] "
            f"[dim]({len(documents)} docs → {len(chunks)} chunks)[/dim]"
        )

    # One batched embed call for the whole corpus (async hygiene), mirroring the
    # runner's batched query embed — never one HTTP request per chunk.
    texts = [chunk.text for chunk in chunks]
    vectors = await embedder.embed(texts)
    if hybrid:
        assert sparse_embedder is not None  # narrowed above; guards the type checker
        sparse_vectors = await sparse_embedder.embed(texts)
        await store.upsert(chunks, vectors, sparse_vectors)
    else:
        await store.upsert(chunks, vectors)
    return name
