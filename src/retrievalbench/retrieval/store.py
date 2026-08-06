import uuid
from typing import Protocol

from qdrant_client import AsyncQdrantClient, models

from retrievalbench.model import Chunk

NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


def to_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, chunk_id))


def _payload(chunk: Chunk) -> dict:
    """The metadata carried back through retrieval. Vectors only enable the
    similarity search; everything the pipeline needs downstream lives here."""
    return {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "text": chunk.text,
        "index": chunk.index,
        "token_count": chunk.token_count,
        "metadata": chunk.metadata,
    }


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
                payload=_payload(chunk),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        await self.client.upsert(collection_name=self.collection, points=points)


class QdrantHybridStore:
    """Same store, two vectors per point: a named dense vector ("semantic") and
    a named sparse vector ("text", BM25 with an IDF modifier). A hybrid run
    keys onto its OWN collection (see collection_name), never the dense one —
    the schemas differ, so they must never share a name.
    """

    name = "Qdrant Hybrid Store"

    def __init__(self, collection: str, dim: int):
        self.client = AsyncQdrantClient(url="http://localhost:6333")
        self.collection = collection
        self.dim = dim

    async def exist_create_collection(self) -> None:
        exist = await self.client.collection_exists(self.collection)
        if not exist:
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    "semantic": models.VectorParams(
                        size=self.dim, distance=models.Distance.COSINE
                    ),
                },
                sparse_vectors_config={
                    # Modifier.IDF: Qdrant applies the corpus-wide IDF term at
                    # query time, so BM25's global doc-frequency weighting works
                    # without us maintaining corpus stats by hand.
                    "text": models.SparseVectorParams(modifier=models.Modifier.IDF),
                },
            )

    async def is_populated(self) -> bool:
        """True iff the collection exists AND holds ≥1 point. Same cache-hit
        check as the dense store — an empty collection counts as a miss."""
        if not await self.client.collection_exists(self.collection):
            return False
        result = await self.client.count(collection_name=self.collection)
        return result.count > 0

    async def upsert(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        sparse_vectors: list[models.SparseVector],
    ) -> None:
        await self.exist_create_collection()

        points = [
            models.PointStruct(
                id=to_point_id(chunk.id),
                vector={"semantic": vector, "text": sparse},
                payload=_payload(chunk),
            )
            for chunk, vector, sparse in zip(
                chunks, vectors, sparse_vectors, strict=True
            )
        ]

        await self.client.upsert(collection_name=self.collection, points=points)
