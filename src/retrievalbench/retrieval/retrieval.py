from typing import Protocol

from qdrant_client import AsyncQdrantClient

from retrievalbench.config import RetrieverConfig
from retrievalbench.model import RetrievedChunk


class Retrieval(Protocol):
    name: str

    async def dense_search(
        self, query_vectors: list[float], limit: int
    ) -> list[RetrievedChunk]: ...


class QdrantRetrieval:
    """
    Qdrant Vector Retrieval
    """

    name = "Qdrant Retrieval"

    def __init__(self, collection: str, dim: int):
        self.client = AsyncQdrantClient(url="http://localhost:6333")
        self.collection = collection
        self.dim = dim

    async def dense_search(
        self, query_vectors: list[float], limit: int
    ) -> list[RetrievedChunk]:
        response = await self.client.query_points(
            collection_name=self.collection,
            query=query_vectors,
            limit=limit,
            with_payload=True,
        )

        return [
            RetrievedChunk(
                score=retrieved_chunk.score,
                chunk_id=retrieved_chunk.payload["chunk_id"],
                document_id=retrieved_chunk.payload["document_id"],
                metadata=retrieved_chunk.payload["metadata"],
                text=retrieved_chunk.payload["text"],
            )
            for retrieved_chunk in response.points
        ]


# `collection` and `dim` are NOT user config: collection is the per-(chunking,
# embedding) cache key the runner computes (Step 5), dim comes from the chosen
# embedder. So the builder takes them as runtime args alongside the config.
_RETRIEVERS: dict[str, type[Retrieval]] = {
    "dense": QdrantRetrieval,
}


def build_retriever(cfg: RetrieverConfig, collection: str, dim: int) -> Retrieval:
    cls = _RETRIEVERS[cfg.type]
    return cls(collection=collection, dim=dim)
