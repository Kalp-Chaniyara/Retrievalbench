from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    source_path: str
    title: str | None
    text: str | None
    metadata: dict[str, str] = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str
    document_id: str
    text: str
    index: int  # order within the document
    token_count: int
    metadata: dict[str, str] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    score: float
    chunk_id: str
    document_id: str
    metadata: dict[str, str] = Field(default_factory=dict)
