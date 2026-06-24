from typing import Protocol

from qdrant_client import AsyncQdrantClient

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
            )
            for retrieved_chunk in response.points
        ]
