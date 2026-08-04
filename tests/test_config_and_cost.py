"""Config validation + cost accounting.

Both guard against SILENT wrongness — the worst failure mode in a benchmark,
because the run still completes and produces plausible numbers you then compare
against other runs. A typo'd YAML key that falls back to a default, or a cost
that is always $0.00, corrupts every downstream conclusion without ever raising.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from retrievalbench.config import RetrievalConfig, load_config
from retrievalbench.generate import _PRICE_PER_1M, OpenAIGenerator, token_cost

BASE = {"name": "x", "embedding": {}, "retrieval": {}, "generation": {}}


def test_typo_in_a_yaml_key_is_rejected() -> None:
    """extra='forbid'. Without it, `siez: 200` is ignored, size silently falls
    back to its default, and the whole experiment measures the wrong thing."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RetrievalConfig(chunking={"type": "fixed", "siez": 200}, **BASE)


def test_unknown_top_level_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RetrievalConfig(chunking={"type": "fixed"}, nonsense=1, **BASE)


def test_top_k_final_cannot_exceed_top_k_retrieve() -> None:
    """You cannot rerank 20 items down from a shortlist of 5."""
    with pytest.raises(ValidationError, match="cannot exceed"):
        RetrievalConfig(
            chunking={"type": "fixed"}, top_k_retrieve=5, top_k_final=20, **BASE
        )


def test_hybrid_autofills_the_sparse_encoder() -> None:
    """Hybrid search returns nothing if chunks lack sparse vectors, so declaring
    `retrieval: {type: hybrid}` must not silently leave the encoder unset."""
    cfg = RetrievalConfig(
        chunking={"type": "fixed"},
        name="x",
        embedding={},
        retrieval={"type": "hybrid"},
        generation={},
    )
    assert cfg.sparse_embedding is not None
    assert cfg.sparse_embedding.type == "bm25"


def test_dense_config_leaves_sparse_unset() -> None:
    cfg = RetrievalConfig(chunking={"type": "fixed"}, **BASE)
    assert cfg.sparse_embedding is None


def test_shipped_configs_all_load() -> None:
    """Every committed YAML must parse — a broken config should fail CI, not a run."""
    paths = sorted(Path("configs").glob("*.yaml"))
    assert paths, "no configs found"
    for path in paths:
        cfg = load_config(path)
        assert cfg.name


def test_missing_config_file_raises_clearly() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("configs/does_not_exist.yaml")


def test_malformed_yaml_names_the_cause(tmp_path: Path) -> None:
    """safe_load returns None for an empty file; the guard must not **None-splat."""
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    with pytest.raises(ValueError, match="did not parse to a mapping"):
        load_config(empty)


# --- cost accounting -------------------------------------------------------


def test_token_cost_arithmetic() -> None:
    """gpt-4o-mini: $0.15/1M in, $0.60/1M out."""
    assert token_cost("gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)
    assert token_cost("gpt-4o-mini", 0, 1_000_000) == pytest.approx(0.60)
    assert token_cost("gpt-4o-mini", 1000, 500) == pytest.approx(
        (1000 * 0.15 + 500 * 0.60) / 1_000_000
    )


def test_zero_tokens_costs_zero() -> None:
    assert token_cost("gpt-4o-mini", 0, 0) == 0.0


def test_unpriced_generator_model_raises_at_construction() -> None:
    """The original bug was cost_usd defaulting to 0.0 forever, which made
    total_cost_usd identically $0.00 and every cost comparison meaningless.
    An unpriced model must fail loudly at construction, not silently cost $0."""
    with pytest.raises(ValueError, match="No price entry"):
        OpenAIGenerator(model="some-unreleased-model")


def test_every_priced_model_has_input_and_output_rates() -> None:
    for model, price in _PRICE_PER_1M.items():
        assert len(price) == 2, model
        assert all(rate > 0 for rate in price), model
