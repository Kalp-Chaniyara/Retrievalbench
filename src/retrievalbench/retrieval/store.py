import uuid
from typing import Protocol

from qdrant_client import AsyncQdrantClient, models

from retrievalbench.model import Chunk

NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


def to_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, chunk_id))


class VectorStore(Protocol):
    name: str

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...


class QdrantStore:
    """
    Qdrant Vector DB
    """

    name = "Qdrant Store"

    def __init__(self, collection: str, dim: int):
        self.client = AsyncQdrantClient(url="http://localhost:6333")
        self.collection = collection
        self.dim = dim

    async def exist_create_collection(self) -> None:
        exist = await self.client.collection_exists(self.collection)
        if not exist:
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.dim, distance=models.Distance.COSINE
                ),
            )

    async def is_populated(self) -> bool:
        """True iff the collection exists AND holds ≥1 point. The index-cache
        hit check: an existing-but-empty collection (e.g. a run that crashed
        mid-upsert) counts as a miss so we re-index instead of querying nothing.
        """
        if not await self.client.collection_exists(self.collection):
            return False
        result = await self.client.count(collection_name=self.collection)
        return result.count > 0

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        await self.exist_create_collection()

        points = [
            models.PointStruct(
                id=to_point_id(chunk.id),
                vector=vector,
                payload={
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "index": chunk.index,
                    "token_count": chunk.token_count,
                    "metadata": chunk.metadata,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        await self.client.upsert(collection_name=self.collection, points=points)
