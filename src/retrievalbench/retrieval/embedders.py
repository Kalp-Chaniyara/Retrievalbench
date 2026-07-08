import asyncio
from typing import Protocol

from fastembed import SparseTextEmbedding
from openai import AsyncOpenAI
from qdrant_client import models

from retrievalbench.config import EmbeddingConfig, SparseEmbeddingConfig


class Embedder(Protocol):
    name: str
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAITextEmbedderSmall:
    """
    Embeds texts using OpenAI text embedder small
    """

    name = "openAI_text_embedder_small"
    dim = 1536

    def __init__(self):
        self.client = AsyncOpenAI()

    async def embed(self, texts: list[str]) -> list[list[float]]:

        response = await self.client.embeddings.create(
            model="text-embedding-3-small", input=texts
        )

        return [embd.embedding for embd in response.data]


# Only one embedder today; the registry still makes it config-swappable and
# ready for a second entry. OpenAITextEmbedderSmall takes no constructor args
# (model is fixed internally), so the builder just instantiates it.
_EMBEDDERS: dict[str, type[Embedder]] = {
    "openai_small": OpenAITextEmbedderSmall,
}


def build_embedder(cfg: EmbeddingConfig) -> Embedder:
    cls = _EMBEDDERS[cfg.type]
    return cls()


class SparseEmbedder(Protocol):
    name: str

    # Two encodings on purpose: BM25 documents carry term frequencies, but a
    # query must NOT be weighted by them — query_embed emits the raw query terms
    # and Qdrant's IDF modifier supplies the corpus weighting at search time.
    async def embed(self, texts: list[str]) -> list[models.SparseVector]: ...

    async def embed_query(self, texts: list[str]) -> list[models.SparseVector]: ...


class BM25SparseEmbedder:
    """FastEmbed BM25 sparse encoder (client-side). Returns Qdrant SparseVectors
    directly — the only consumer is the store/retriever, so converting numpy →
    plain lists here keeps numpy from leaking through the rest of the pipeline.
    """

    name = "bm25"

    def __init__(self):
        # Local CPU model; downloads once on first use, then cached on disk.
        self.model = SparseTextEmbedding(model_name="Qdrant/bm25")

    async def embed(self, texts: list[str]) -> list[models.SparseVector]:
        # Sync/CPU-bound: run off the event loop instead of blocking it. This is
        # the correct async handling for a sync client — NOT `await` (there is
        # no I/O to await), and NOT a bare call (it would stall the loop).
        return await asyncio.to_thread(self._encode, texts, False)

    async def embed_query(self, texts: list[str]) -> list[models.SparseVector]:
        return await asyncio.to_thread(self._encode, texts, True)

    def _encode(self, texts: list[str], is_query: bool) -> list[models.SparseVector]:
        raw = self.model.query_embed(texts) if is_query else self.model.embed(texts)
        return [
            models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
            for e in raw
        ]


_SPARSE_EMBEDDERS: dict[str, type[SparseEmbedder]] = {
    "bm25": BM25SparseEmbedder,
}


def build_sparse_embedder(cfg: SparseEmbeddingConfig) -> SparseEmbedder:
    cls = _SPARSE_EMBEDDERS[cfg.type]
    return cls()
