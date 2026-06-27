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
    text: str
    document_id: str
    metadata: dict[str, str] = Field(default_factory=dict)


class GoldenItem(BaseModel):
    query: str
    expected_chunk_ids: list[str]
    expected_answer: str


class MetricScore(BaseModel):
    """One metric's score (0..1) plus the judge's human-readable reason."""

    score: float
    reason: str


class EvalScores(BaseModel):
    """All four RAG metrics for a single query."""

    faithfulness: MetricScore
    answer_relevancy: MetricScore
    context_precision: MetricScore
    context_recall: MetricScore
