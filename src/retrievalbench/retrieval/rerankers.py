import asyncio
from typing import Protocol

from rerankers import Reranker as _RerankersReranker

from retrievalbench.config import RerankerConfig
from retrievalbench.model import RetrievedChunk


class Reranker(Protocol):
    name: str

    async def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        # Re-score the retrieved candidates against the query and return the top
        # `top_k`. Takes RetrievedChunk (what the retriever emits), not Chunk, so
        # the runner passes retrieval output straight through.
        ...


class CrossEncoderReranker:
    """Cross-encoder reranker via the `rerankers` library (default
    `BAAI/bge-reranker-v2-m3`). Unlike the bi-encoder retriever — which scores a
    chunk by the distance between two SEPARATELY embedded vectors — a
    cross-encoder feeds (query, chunk) through the model TOGETHER, so it can
    attend across both. Slower and can't be pre-indexed, which is exactly why it
    reranks a shortlist (top_k_retrieve) instead of the whole corpus.
    """

    name = "cross_encoder"

    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3"):
        # Loads a local transformers model (downloads once, then cached). The lib
        # picks the cross-encoder backend from the model name/type.
        self.model = _RerankersReranker(model, model_type="cross-encoder")

    async def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        # Sync + CPU/GPU-bound (a forward pass per candidate): run off the event
        # loop like the sparse embedder, NOT a bare call that would stall it.
        return await asyncio.to_thread(self._rank, query, candidates, top_k)

    def _rank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        # doc_ids default to list index, so each result's doc_id maps back to the
        # candidate that produced it — no reliance on the lib accepting our string
        # chunk ids. The rerank score REPLACES the retrieval score so downstream
        # (ordering, viewer) sees the value that decided the final order.
        ranked = self.model.rank(query=query, docs=[c.text for c in candidates])
        return [
            candidates[result.doc_id].model_copy(update={"score": result.score})
            for result in ranked.top_k(top_k)
        ]


_RERANKERS: dict[str, type[Reranker]] = {
    "cross_encoder": CrossEncoderReranker,
}


def build_reranker(cfg: RerankerConfig) -> Reranker:
    cls = _RERANKERS[cfg.type]
    return cls(model=cfg.model)
