from typing import Protocol

from qdrant_client import AsyncQdrantClient, models

from retrievalbench.config import RetrieverConfig
from retrievalbench.model import RetrievedChunk


def _to_retrieved(points) -> list[RetrievedChunk]:
    """Qdrant scored points -> domain RetrievedChunks, via the payload every
    point carries (chunk_id is the fusion identity key)."""
    return [
        RetrievedChunk(
            score=point.score,
            chunk_id=point.payload["chunk_id"],
            document_id=point.payload["document_id"],
            metadata=point.payload["metadata"],
            text=point.payload["text"],
        )
        for point in points
    ]


class Retrieval(Protocol):
    name: str

    async def retrieve(
        self,
        dense_vector: list[float],
        limit: int,
        sparse_vector: models.SparseVector | None = None,
    ) -> list[RetrievedChunk]:
        # One method for both retrievers so the runner never branches on type.
        # Dense ignores sparse_vector; hybrid requires it.
        ...


class QdrantRetrieval:
    """
    Qdrant dense (semantic) retrieval against an unnamed-vector collection.
    """

    name = "Qdrant Retrieval"

    def __init__(self, collection: str, dim: int):
        self.client = AsyncQdrantClient(url="http://localhost:6333")
        self.collection = collection
        self.dim = dim

    async def retrieve(
        self,
        dense_vector: list[float],
        limit: int,
        sparse_vector: models.SparseVector | None = None,
    ) -> list[RetrievedChunk]:
        response = await self.client.query_points(
            collection_name=self.collection,
            query=dense_vector,
            limit=limit,
            with_payload=True,
        )
        return _to_retrieved(response.points)


class QdrantHybridRetrieval:
    """Dense + sparse over one hybrid collection, fused with Reciprocal Rank
    Fusion. Runs both searches, then combines them on rank (not raw score —
    cosine and BM25 live in incomparable spaces; only their ORDERING is
    comparable).
    """

    name = "Qdrant Hybrid Retrieval"

    def __init__(self, collection: str, dim: int, rrf_k: int = 60):
        self.client = AsyncQdrantClient(url="http://localhost:6333")
        self.collection = collection
        self.dim = dim
        self.rrf_k = rrf_k

    async def retrieve(
        self,
        dense_vector: list[float],
        limit: int,
        sparse_vector: models.SparseVector | None = None,
    ) -> list[RetrievedChunk]:
        if sparse_vector is None:
            raise ValueError("hybrid retrieval requires a sparse query vector")
        # Both sides pull `limit` candidates so fusion sees the full lists, then
        # returns the top `limit` — the runner slices to top_k_final after.
        dense = await self._dense_search(dense_vector, limit)
        sparse = await self._sparse_search(sparse_vector, limit)
        return self._fuse(dense, sparse, limit)

    async def _dense_search(
        self, dense_vector: list[float], limit: int
    ) -> list[RetrievedChunk]:
        response = await self.client.query_points(
            collection_name=self.collection,
            query=dense_vector,
            using="semantic",
            limit=limit,
            with_payload=True,
        )
        return _to_retrieved(response.points)

    async def _sparse_search(
        self, sparse_vector: models.SparseVector, limit: int
    ) -> list[RetrievedChunk]:
        response = await self.client.query_points(
            collection_name=self.collection,
            query=sparse_vector,
            using="text",
            limit=limit,
            with_payload=True,
        )
        return _to_retrieved(response.points)

    def _fuse(
        self,
        dense: list[RetrievedChunk],
        sparse: list[RetrievedChunk],
        limit: int,
    ) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion over the UNION of chunk_ids.

        Walk each ranked list independently and add 1/(k+rank) into a score dict
        keyed by chunk_id. A chunk in both lists gets two contributions and
        rises; a chunk in only one gets exactly that one term (nothing is added
        for the list it's absent from — no zero-rank, no skip). Return the top
        `limit` by fused score, carrying the RetrievedChunk (its `score` becomes
        the RRF score so downstream sees a meaningful ordering value).
        """
        scores: dict[str, float] = {}
        chunks: dict[str, RetrievedChunk] = {}
        for ranked in (dense, sparse):
            for rank, chunk in enumerate(ranked, start=1):
                scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (
                    self.rrf_k + rank
                )
                chunks.setdefault(chunk.chunk_id, chunk)

        ordered = sorted(scores, key=scores.__getitem__, reverse=True)
        return [
            chunks[chunk_id].model_copy(update={"score": scores[chunk_id]})
            for chunk_id in ordered[:limit]
        ]


# `collection` and `dim` are NOT user config: collection is the per-(chunking,
# embedding[, sparse]) cache key the runner computes, dim comes from the chosen
# embedder. So the builder takes them as runtime args alongside the config.
def build_retriever(cfg: RetrieverConfig, collection: str, dim: int) -> Retrieval:
    if cfg.type == "hybrid":
        return QdrantHybridRetrieval(collection=collection, dim=dim, rrf_k=cfg.rrf_k)
    return QdrantRetrieval(collection=collection, dim=dim)
