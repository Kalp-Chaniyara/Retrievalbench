import asyncio
import random
import re
import uuid

from openai import AsyncOpenAI
from pydantic import BaseModel

from retrievalbench.model import Chunk, GoldenItem, QueryType, RetrievedChunk

_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Collapse whitespace + lowercase. Source docs hard-wrap sentences, so a
    verbatim snippet spans newlines once chunked; normalizing both sides lets a
    clean single-line snippet still match the wrapped chunk text."""
    return _WHITESPACE.sub(" ", text).strip().lower()


def chunk_matches_snippets(chunk_text: str, expected_snippets: list[str]) -> bool:
    """True if a chunk is a retrieval hit for a golden item: it verbatim-contains
    (after whitespace/case normalization) ANY of the item's answer-bearing
    snippets. Snippets are config-independent source text, so this resolves to
    the right chunk(s) under whatever chunking produced them — no chunk id is
    hardcoded. Shared by the CLI display and (later) the F1 diagnostic."""
    haystack = _normalize(chunk_text)
    return any(_normalize(snippet) in haystack for snippet in expected_snippets)


def hit_chunk_ids(retrieved: list[RetrievedChunk], item: GoldenItem) -> set[str]:
    """Which of the retrieved chunk ids actually satisfy the golden item — the
    per-config resolution of 'the expected chunks' that a literal id list can't
    express. Empty set == retrieval miss (the F1 condition)."""
    return {
        chunk.chunk_id
        for chunk in retrieved
        if chunk_matches_snippets(chunk.text, item.expected_snippets)
    }


# --- LLM-based golden generator (Design §5.7) ---
#
# One sampled chunk -> one candidate (question, expected_answer, snippets),
# via a direct structured-output AsyncOpenAI call (no RAGAS/DeepEval testset
# generator: neither library's output has a verified verbatim-substring
# span, which is the one property the F1 gate actually depends on — see the
# design discussion in golden.py's history). A candidate only becomes
# reviewable if every snippet it returned is a real substring of the chunk
# it was generated from; anything else is discarded before a human ever
# sees it, so the CLI review step never has to second-guess the snippet.

DEFAULT_GENERATOR_MODEL = "gpt-4o-mini"

_GENERATOR_SYSTEM_PROMPT = (
    "You write one golden evaluation item for a RAG benchmark from a single "
    "source passage. Ask a specific question answerable ONLY from this "
    "passage (not general knowledge). Give a concise, correct answer. Then "
    "quote 1-2 short, distinctive phrases COPIED VERBATIM from the passage "
    "(exact characters, no paraphrasing, no ellipses) that together prove "
    "the answer. Keep each quote short enough that it couldn't plausibly "
    "appear in an unrelated passage."
)

# The query TYPE is an input, not something the model labels after the fact.
# Asking for "an exact-match query" and getting one makes the type true by
# construction; letting the model classify its own output would put noise into
# the per-type F1 breakdown, which is the finding this whole exercise exists
# to produce.
#
# exact_match is the load-bearing one: dense retrieval embeds meaning, so a
# rare literal token (a PEP number, a version string, an error code) has weak
# semantic signal and is exactly what a bi-encoder misses. Those are the
# queries expected to produce F1 — and the ones hybrid/BM25 should rescue.
_TYPE_GUIDANCE: dict[str, str] = {
    "exact_match": (
        "The question MUST hinge on a rare literal token that appears verbatim "
        "in the passage — an identifier, number, version string, error code, "
        "status field, date, or acronym. Use that exact token in the question. "
        "Do not paraphrase it. A reader must need that precise string to answer."
    ),
    "semantic": (
        "Ask a conceptual question about what the passage MEANS. Deliberately "
        "avoid reusing the passage's distinctive wording — paraphrase, so the "
        "question and the source share meaning but few literal tokens."
    ),
    "negation": (
        "Ask a question whose correct answer is a denial, a limitation, or an "
        "explicit absence stated in the passage (something rejected, deferred, "
        "withdrawn, not supported, or disallowed). The expected answer must "
        "say what is NOT the case."
    ),
    "multi_hop": (
        "Two passages are given. Ask ONE question that cannot be answered from "
        "either passage alone — it must require combining a fact from each. "
        "Quote at least one verbatim phrase from EACH passage."
    ),
}


class GeneratedCandidate(BaseModel):
    """Structured-output shape for one LLM-generated candidate, pre-review."""

    question: str
    expected_answer: str
    expected_snippets: list[str]


def _all_snippets_verbatim(chunk_text: str, snippets: list[str]) -> bool:
    """Generation-time validation: EVERY snippet must be a real substring of
    the exact chunk it was generated from (vs. `chunk_matches_snippets`'s
    retrieval-time ANY-of, which checks a snippet against an arbitrary
    retrieved chunk). A candidate that fails this was paraphrased, not
    quoted, and is discarded before it ever reaches review."""
    if not snippets:
        return False
    haystack = _normalize(chunk_text)
    return all(_normalize(snippet) in haystack for snippet in snippets)


def sample_chunks(chunks: list[Chunk], n: int, seed: int) -> list[Chunk]:
    """Seeded sample so `rbench gen-golden` is reproducible (G4) given the
    same corpus + n + seed, without regenerating from every chunk."""
    return random.Random(seed).sample(chunks, min(n, len(chunks)))


async def _generate_candidate(
    client: AsyncOpenAI, model: str, chunks: list[Chunk], query_type: QueryType
) -> GeneratedCandidate | None:
    # multi_hop gets two passages; every other type gets one. The verbatim
    # check below runs against the concatenation either way.
    passages = "\n\n---\n\n".join(
        f"Passage {i}:\n{c.text}" for i, c in enumerate(chunks, start=1)
    )
    response = await client.chat.completions.parse(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": f"{_GENERATOR_SYSTEM_PROMPT}\n\n{_TYPE_GUIDANCE[query_type]}",
            },
            {"role": "user", "content": passages},
        ],
        response_format=GeneratedCandidate,
    )
    candidate = response.choices[0].message.parsed
    source = "\n".join(c.text for c in chunks)
    if candidate is None or not _all_snippets_verbatim(
        source, candidate.expected_snippets
    ):
        return None
    return candidate


async def generate_candidates(
    chunks: list[Chunk],
    *,
    query_type: QueryType = "semantic",
    model: str = DEFAULT_GENERATOR_MODEL,
    n: int = 10,
    seed: int = 42,
) -> list[tuple[list[Chunk], GeneratedCandidate]]:
    """Sample chunks and generate `n` candidates of ONE query type, concurrently
    (independent LLM calls -> asyncio.gather, not a loop). Returns only the
    candidates that passed verbatim validation, paired with their source
    chunk(s) so the review step can show them.

    multi_hop draws 2 chunks per item because a question answerable from a
    single passage is not multi-hop by definition.
    """
    per_item = 2 if query_type == "multi_hop" else 1
    sampled = sample_chunks(chunks, n * per_item, seed)
    groups = [sampled[i : i + per_item] for i in range(0, len(sampled), per_item)]
    groups = [g for g in groups if len(g) == per_item]

    client = AsyncOpenAI()
    results = await asyncio.gather(
        *(_generate_candidate(client, model, group, query_type) for group in groups)
    )
    return [
        (group, candidate)
        for group, candidate in zip(groups, results, strict=True)
        if candidate is not None
    ]


def candidate_to_golden_item(
    candidate: GeneratedCandidate, query_type: QueryType = "semantic"
) -> GoldenItem:
    """A fresh id per generated item — 'gen_' prefixed so it never collides
    with the hand-written ids in GOLDEN_SET below.

    `query_type` is set by the HUMAN during review, not by the generator: the
    per-type F1 breakdown is the finding, so mislabelling it would corrupt the
    result the whole exercise is for.
    """
    return GoldenItem(
        id=f"gen_{uuid.uuid4().hex[:8]}",
        query=candidate.question,
        expected_answer=candidate.expected_answer,
        expected_snippets=candidate.expected_snippets,
        query_type=query_type,
    )


# Hand-written ground truth, scoped BY CORPUS. A dict rather than a flat list
# because expected_snippets must exist verbatim in a specific corpus — a flat
# list silently attaches coffee questions to a Python-PEP corpus and every
# query becomes a meaningless F1. Generated items live in GoldenStore (SQLite,
# also corpus-scoped) and are merged in at read time by cli._golden_set.
GOLDEN_SET: dict[str, list[GoldenItem]] = {
    "sample_data1": [
        GoldenItem(
            id="t1",  # temporal negation: after-cutoff
            query=(
                "I placed my order at 3 PM Pacific on a Wednesday. "
                "Will it be roasted that same day?"
            ),
            expected_snippets=["roasted on the next roast day"],
            expected_answer=(
                "No. Orders placed after the 10:00 AM Pacific cutoff are roasted "
                "on the next roast day, not the same day."
            ),
            query_type="negation",
        ),
        GoldenItem(
            id="t2",  # multi-hop: price (catalog) x 2 vs the $40 rule (shipping)
            query=(
                "If I buy two bags of Hambela, do I qualify for free standard shipping?"
            ),
            expected_snippets=[
                "$21 per 12-ounce",
                "Orders over $40 qualify for free standard shipping",
            ],
            expected_answer=(
                "Yes. Hambela is $21 per bag, so two bags total $42, which is over "
                "the $40 threshold that qualifies an order for free standard "
                "shipping."
            ),
            query_type="multi_hop",
        ),
        GoldenItem(
            id="t3",  # not-in-text geography: hallucination bait
            query="Can I have my coffee delivered to Toronto, Canada?",
            expected_snippets=[
                "Aurora currently does not ship outside the United States"
            ],
            expected_answer=(
                "No. Aurora ships within the United States only and does not "
                "currently ship internationally."
            ),
            query_type="negation",
        ),
    ],
    # techdocs items are LLM-generated + reviewed; they live in GoldenStore.
    "techdocs": [],
}
