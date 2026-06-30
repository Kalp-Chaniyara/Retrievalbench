from typing import Protocol

import tiktoken
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from retrievalbench.config import ChunkingConfig
from retrievalbench.model import Chunk, Document


# generate the chunk_id
def make_chunk_id(document_id: str, index: int) -> str:
    return f"{document_id}_{index:04d}"


class Chunker(Protocol):
    """
    Structural interface every chunker must statisfy
    """

    name: str

    def chunk(self, doc: Document): ...


class FixedSizeChunker:
    """
    Splits text into the fixed size chunks with proper overlapping of the words
    """

    name = "fixed_size_chunker"

    def __init__(self, size: int = 800, overlap: int = 150):
        self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        self.splitter = TokenTextSplitter(
            model_name="gpt-4o-mini", chunk_size=size, chunk_overlap=overlap
        )

    def chunk(self, doc: Document) -> list[Chunk]:
        row_chunks = self.splitter.split_text(doc.text)
        chunks = []
        for idx, row_chunk in enumerate(row_chunks):
            token_count = len(self.encoding.encode(row_chunk))

            chunks.append(
                Chunk(
                    id=make_chunk_id(doc.id, idx),
                    document_id=doc.id,
                    text=row_chunk,
                    index=idx,
                    token_count=token_count,
                )
            )

        return chunks


class RecursiveChunker:
    """
    Splits text by trying a hierarchy of separators in order
    (paragraph -> line -> sentence -> word), so chunks fall on natural
    boundaries instead of mid-sentence cuts. size/overlap are measured in
    tokens (via tiktoken) to stay comparable with FixedSizeChunker.
    """

    name = "recursive_chunker"

    def __init__(self, size: int = 800, overlap: int = 150):
        self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o-mini",
            chunk_size=size,
            chunk_overlap=overlap,
        )

    def chunk(self, doc: Document) -> list[Chunk]:
        row_chunks = self.splitter.split_text(doc.text)
        chunks = []
        for idx, row_chunk in enumerate(row_chunks):
            token_count = len(self.encoding.encode(row_chunk))

            chunks.append(
                Chunk(
                    id=make_chunk_id(doc.id, idx),
                    document_id=doc.id,
                    text=row_chunk,
                    index=idx,
                    token_count=token_count,
                )
            )

        return chunks


# Registry maps the YAML wire-name (cfg.type) -> class. Lives here, beside the
# classes (option B), so config.py never imports implementations. Keys match the
# Literal values in ChunkingConfig, so the lookup can't miss after validation.
_CHUNKERS: dict[str, type[Chunker]] = {
    "fixed": FixedSizeChunker,
    "recursive": RecursiveChunker,
}


def build_chunker(cfg: ChunkingConfig) -> Chunker:
    cls = _CHUNKERS[cfg.type]
    return cls(size=cfg.size, overlap=cfg.overlap)
