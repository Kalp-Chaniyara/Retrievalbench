# Experiment log

Measured results. Kept out of the README so that stays a "how to use this"
document rather than a lab notebook.

Machine-readable regression baselines live beside this file in `baseline.json`
(written by `scripts/check_baseline.py --record`, read by the Tier-2 CI gate).

---

## Setup

| | |
|---|---|
| Corpus | `techdocs` — 733 Python PEPs (`.rst`), 15 MB |
| Golden set | 26 items, LLM-generated + verbatim-validated, typed by construction |
| Judge | `gpt-4o` · Generator | `gpt-4o-mini` · Embedder | `text-embedding-3-small` |
| Date | 2026-08-06 |

Golden set by type: 7 semantic · 7 negation · 6 exact_match · 6 multi_hop.
32 candidates were generated; **6 were rejected** by the verbatim-snippet
validator (19%), which is the validator doing its job — a paraphrased "quote"
is dead ground truth.

---

## Why the old corpus measured nothing

The original corpus was 6 documents. Retrieval asked for more chunks than
existed, so **every query retrieved the entire corpus**:

```
sample_data1 — 6 documents
  fixed_512           14 chunks   k=20   142.86%   <-- k >= N
  hybrid_reranked     13 chunks   k=50   384.62%   <-- k >= N
  recursive_800        6 chunks   k=20   333.33%   <-- k >= N

techdocs — 733 documents
  fixed_512        23999 chunks   k=20     0.08%
  hybrid_reranked  28438 chunks   k=50     0.18%
  recursive_800     5350 chunks   k=20     0.37%
```

`hit_chunk_ids` can never return empty when every chunk is always retrieved, so
**F1 was arithmetically impossible** — not unlikely. All 7 historical runs
measured nothing about retrieval. This is why F1 had fired 0 times.

Reproduce with `uv run python scripts/corpus_stats.py`.

---

## Results — 26 golden items, `techdocs`

| config | pass | F1 | F_GEN | latency | cost/run | faithfulness |
|---|---|---|---|---|---|---|
| `fixed_512` (dense) | 12/26 (46%) | **8** | 6 | 2054 ms | $0.0046 | 0.897 |
| `hybrid_reranked` | 15/26 (58%) | **3** | 8 | 4931 ms | $0.0050 | 0.856 |

### Per query type

`fixed_512`

| query type | n | PASS | F1 | F_GEN | F1 rate |
|---|---|---|---|---|---|
| exact_match | 6 | 3 | 2 | 1 | 33% |
| multi_hop | 6 | 2 | 3 | 1 | 50% |
| negation | 7 | 3 | 1 | 3 | 14% |
| semantic | 7 | 4 | 2 | 1 | 29% |

`hybrid_reranked`

| query type | n | PASS | F1 | F_GEN | F1 rate |
|---|---|---|---|---|---|
| exact_match | 6 | 5 | 1 | 0 | 17% |
| multi_hop | 6 | 2 | 1 | 3 | 17% |
| negation | 7 | 4 | 0 | 3 | 0% |
| semantic | 7 | 4 | 1 | 2 | 14% |

---

## What the numbers say

**F1 dropped 8 → 3 while F_GEN rose 6 → 8.** Five retrieval misses were fixed;
two of them became *generation* failures rather than passes. The failure moved
stage rather than disappearing.

That is the whole argument for stage attribution. An aggregate score reports
"46% → 58%, better". The diagnostics report says **retrieval is now largely
solved and the remaining problem is the generator** — F_GEN is 73% of failures
in the hybrid run. Those imply different next actions, and only one of them is
visible without the F1/F_GEN split.

**Per type:** F1 fell in every category. `multi_hop` was worst under dense
(50%) — one query embedding has to surface evidence from two documents, and a
single point in vector space pulls toward one topic. `negation` had the lowest
F1 (14%) but the highest F_GEN: retrieval finds the evidence, the generator
mishandles the denial.

**`exact_match` at 33% under dense was lower than predicted.** The hypothesis
was that rare literal tokens would dominate F1. They did not — `top_k=20` over
24k chunks is generous, and PEP prose repeats identifiers across documents, so
a rare token is less isolating than assumed.

---

## Caveats — read before quoting any of this

**The comparison is confounded.** `hybrid_reranked` differs from `fixed_512` in
**8 dimensions** (`chunking.type/size/overlap`, `retrieval.type`,
`reranker.type/model`, `top_k_retrieve` 20→50, plus the auto-attached sparse
encoder). The `top_k` change alone gives 2.5× the candidates and could account
for much of the F1 drop on its own.

So the defensible claim is **"this bundle reduces F1 from 8 to 3"** — *not*
"hybrid retrieval reduces F1". A causal claim needs a config identical to
baseline except `retrieval.type`. `rbench recommend` prints this caveat itself
via `config_diff`.

**n=26 → resolution is ~4 points.** One query flipping moves the pass rate by
3.8 points. The F1 gap (8 → 3) is well outside that; smaller differences in the
per-type tables are not.

**Hybrid is not dominant.** 2.4× the latency for +12 points, and slightly worse
faithfulness (0.897 → 0.856). Whether that trade is worth it is a budget
question, which is what `rbench recommend` exists to answer.

**Judge non-determinism.** Pass/fail comes from a GEval correctness trigger, so
re-running will not reproduce these numbers exactly. That is why the CI gate
compares against a baseline with a tolerance band rather than an absolute
threshold.

---

## Scale bugs found by growing the corpus

Three latent bugs surfaced only once the corpus was realistic. All were
invisible at 14 chunks:

| Bug | Symptom | Fix |
|---|---|---|
| `embed()` sent all texts in one request | `BadRequestError`: 5.1M tokens vs 300k cap | batch at 128 texts |
| `upsert()` sent all points in one request | `ReadError`: ~150 MB payload dropped | batch at 256 points |
| Judge calls fired concurrently | `RateLimitError` 429: `gpt-4o` capped at 30k TPM | serialize metric + diagnostic calls |

The third reversed an earlier design decision. Concurrency is correct when
latency-bound; this workload is **token-bound** (10s CPU across 18 min wall
clock — idle ~99% of the time waiting on TPM), so parallelism only reaches the
rate limit sooner. A bounded semaphore would have made it worse, which is why
Design §5.8's `[V3]` concurrency item was deliberately not implemented.

Known consequence: batched upserts make a *partial* index possible if a run is
killed mid-ingest, and `is_populated()` only checks `count > 0` — so an
incomplete index would be served as a cache hit. Not hit in practice; verify
point counts against `corpus_stats.py` if a run is interrupted.

---

## Reproducing

```bash
docker compose up -d
uv run python scripts/corpus_stats.py
uv run rbench run --config configs/baseline.yaml          # ~18 min (TPM-bound)
uv run rbench run --config configs/hybrid_reranked.yaml   # ~35 min (+ indexing)
uv run rbench report <run_id>
uv run rbench recommend
```

Indexing is cached per `(corpus, chunking, embedding[, sparse])`, so only the
first run per chunking config pays for embedding.
