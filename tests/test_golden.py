"""Golden-set integrity + the snippet hit-test.

The integrity test here is the one that matters most: a golden item whose
snippet does not exist verbatim in the corpus is DEAD GROUND TRUTH. Because
`chunk_matches_snippets` is any-of, a typo'd snippet fails silently — the item
keeps passing on its other snippet while testing less than you think. That is
exactly what happened to `t2` ("standagrd" for "standard"), which quietly
downgraded a multi-hop check to a single-snippet one.
"""

from pathlib import Path

import pytest

from retrievalbench.golden import (
    GOLDEN_SET,
    GeneratedCandidate,
    _normalize,
    candidate_to_golden_item,
    chunk_matches_snippets,
    hit_chunk_ids,
    sample_chunks,
)

from .conftest import make_chunk, make_golden, make_retrieved


def test_every_golden_snippet_exists_verbatim_in_the_corpus(corpus_dir: Path) -> None:
    """Guards ground truth. A snippet that matches nothing is a silent hole."""
    corpus = " \n ".join(_normalize(p.read_text()) for p in corpus_dir.glob("*.md"))
    missing = [
        (item.id, snippet)
        for item in GOLDEN_SET
        for snippet in item.expected_snippets
        if _normalize(snippet) not in corpus
    ]
    assert not missing, f"golden snippets not found in the corpus: {missing}"


def test_golden_ids_are_unique() -> None:
    ids = [item.id for item in GOLDEN_SET]
    assert len(ids) == len(set(ids))


def test_golden_items_have_snippets_and_answers() -> None:
    for item in GOLDEN_SET:
        assert item.expected_snippets, f"{item.id} has no expected_snippets"
        assert item.expected_answer.strip(), f"{item.id} has no expected_answer"
        assert item.query.strip(), f"{item.id} has no query"


@pytest.mark.parametrize(
    ("chunk_text", "snippets", "expected"),
    [
        ("the cat sat on the mat", ["cat sat"], True),
        ("the cat sat on the mat", ["dog sat"], False),
        ("THE CAT   SAT\non the mat", ["cat sat"], True),  # normalised
        ("the cat sat", ["nope", "cat sat"], True),  # any-of
        ("the cat sat", [], False),  # no snippets -> no hit
    ],
)
def test_chunk_matches_snippets(
    chunk_text: str, snippets: list[str], expected: bool
) -> None:
    assert chunk_matches_snippets(chunk_text, snippets) is expected


def test_hit_chunk_ids_returns_only_matching_chunks() -> None:
    item = make_golden("key fact")
    retrieved = make_retrieved("noise", "contains the key fact", "more noise")
    assert hit_chunk_ids(retrieved, item) == {"doc_0001"}


def test_hit_chunk_ids_empty_means_retrieval_miss() -> None:
    """Empty set IS the F1 condition — the contract diagnostics depends on."""
    item = make_golden("never appears")
    assert hit_chunk_ids(make_retrieved("a", "b"), item) == set()


def test_sample_chunks_is_deterministic_for_a_seed() -> None:
    """G4 reproducibility: same corpus + n + seed -> same sample."""
    chunks = [make_chunk(f"text {i}", i) for i in range(20)]
    assert [c.id for c in sample_chunks(chunks, 5, seed=42)] == [
        c.id for c in sample_chunks(chunks, 5, seed=42)
    ]
    assert [c.id for c in sample_chunks(chunks, 5, seed=1)] != [
        c.id for c in sample_chunks(chunks, 5, seed=42)
    ]


def test_sample_chunks_caps_at_population_size() -> None:
    chunks = [make_chunk("a", 0), make_chunk("b", 1)]
    assert len(sample_chunks(chunks, 10, seed=42)) == 2


def test_generated_items_get_a_gen_prefixed_id() -> None:
    """Generated ids must never collide with the hand-written GOLDEN_SET ids."""
    item = candidate_to_golden_item(
        GeneratedCandidate(question="q", expected_answer="a", expected_snippets=["s"])
    )
    assert item.id.startswith("gen_")
    assert item.id not in {g.id for g in GOLDEN_SET}
