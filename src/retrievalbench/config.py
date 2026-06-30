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


class RetrieverConfig(BaseModel):
    # Named RetrieverConfig (not RetrievalConfig) to avoid clashing with the
    # top-level experiment model below. Dense-only for now; `collection`/`dim`
    # are runtime-derived (cache key + embedder.dim), NOT user config.
    model_config = ConfigDict(extra="forbid")

    type: Literal["dense"] = "dense"


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
    generation: GenerationConfig
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
