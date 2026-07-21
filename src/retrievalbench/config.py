from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChunkingConfig(BaseModel):
    # extra="forbid" -> a typo'd YAML key (e.g. "siez") errors here instead of
    # being silently ignored. We want config mistakes to fail loud, early.
    model_config = ConfigDict(extra="forbid")

    type: Literal["fixed", "recursive"]
    size: int = Field(default=800, gt=0)
    overlap: int = Field(default=150, ge=0)


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Only one embedder exists today; `type` is the swap point. Adding a second
    # embedder = add a Literal value here + a registry entry in embedders.py.
    type: Literal["openai_small"] = "openai_small"


class SparseEmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Only used by hybrid retrieval. BM25 is the one sparse encoder today; the
    # swap point mirrors EmbeddingConfig.
    type: Literal["bm25"] = "bm25"


class RetrieverConfig(BaseModel):
    # Named RetrieverConfig (not RetrievalConfig) to avoid clashing with the
    # top-level experiment model below. `collection`/`dim` are runtime-derived
    # (cache key + embedder.dim), NOT user config.
    model_config = ConfigDict(extra="forbid")

    type: Literal["dense", "hybrid"] = "dense"
    # RRF fusion constant (hybrid only). A real experiment knob, not a magic
    # number: higher k spreads weight further down the ranked lists. ~60 is the
    # conventional default.
    rrf_k: int = Field(default=60, gt=0)


class RerankerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Cross-encoder is the only reranker kind today; `type` is the swap point
    # (a hosted Cohere/Jina reranker would add a Literal value + registry entry).
    # `model` varies within that kind — the `rerankers` lib takes a model NAME,
    # so swapping BGE for another cross-encoder is a config change, not code.
    type: Literal["cross_encoder"] = "cross_encoder"
    model: str = "BAAI/bge-reranker-v2-m3"


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["openai"] = "openai"
    model: str = "gpt-4o-mini"
    # NOTE: temperature is intentionally NOT here. G4 hard rule: generation runs
    # at temperature=0 for reproducibility. It's not user-tunable by design.


class RetrievalConfig(BaseModel):
    """One experiment, fully declared in YAML. The source of truth the runner
    reads from — never raw YAML downstream."""

    model_config = ConfigDict(extra="forbid")

    name: str
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    retrieval: RetrieverConfig
    # Optional in the pipeline: None -> retrieval order is the final order (the
    # runner just slices top_k_final). Set -> retrieve top_k_retrieve, rerank,
    # keep top_k_final (the retrieve-50 -> rerank -> top-5 pattern).
    reranker: RerankerConfig | None = None
    generation: GenerationConfig
    # Dense-only configs leave this None; hybrid runs fill it in (below) so a
    # dense config's serialized form stays clean.
    sparse_embedding: SparseEmbeddingConfig | None = None
    top_k_retrieve: int = Field(default=50, gt=0)
    top_k_final: int = Field(default=5, gt=0)
    seed: int = 42

    @model_validator(mode="after")
    def _final_not_more_than_retrieved(self) -> "RetrievalConfig":
        if self.top_k_final > self.top_k_retrieve:
            raise ValueError(
                f"top_k_final ({self.top_k_final}) cannot exceed "
                f"top_k_retrieve ({self.top_k_retrieve})"
            )
        return self

    @model_validator(mode="after")
    def _hybrid_has_sparse(self) -> "RetrievalConfig":
        # Hybrid needs a sparse encoder; default it so YAML can just say
        # `retrieval: {type: hybrid}` without also declaring the encoder.
        if self.retrieval.type == "hybrid" and self.sparse_embedding is None:
            self.sparse_embedding = SparseEmbeddingConfig()
        return self


def load_config(path: str | Path) -> RetrievalConfig:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist")
    if not path.is_file():
        raise FileNotFoundError(f"File {path} is not a file")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # safe_load returns None for an empty file and non-dict for malformed YAML;
    # guard so the error names the cause instead of a cryptic **None splat.
    if not isinstance(data, dict):
        raise ValueError(
            f"Config {path} did not parse to a mapping (got {type(data).__name__})"
        )

    return RetrievalConfig(**data)
