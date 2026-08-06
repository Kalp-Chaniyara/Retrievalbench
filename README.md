# RetrievalBench

A local-first, config-driven harness for running, evaluating, and **diagnosing** retrieval (RAG) pipelines — built from scratch as a learning + portfolio project.

```
documents → chunk → embed → Qdrant → retrieve → rerank → generate → score → diagnose → recommend
```

It scores four RAG metrics per query, then does the thing most eval tools don't: **attributes every failed query to a stage** — was the evidence never retrieved, or was it retrieved and the generator still got it wrong?

> **Honest framing:** this is not trying to beat AutoRAG, RAGAS, or MLflow. Those are mature. This is a from-scratch retrieval-eval harness with failure diagnostics, built to understand retrieval deeply rather than to win a benchmark.

---

## The wedge: deterministic failure attribution

Aggregate scores tell you *that* a pipeline is bad. They don't tell you *which stage* to fix. RetrievalBench labels each failed query:

| Label | Meaning | What to change |
|---|---|---|
| **F1** — retrieval miss | None of the golden item's `expected_snippets` appear in the retrieved chunks. The evidence never reached the generator. | retrieval: hybrid, bigger `top_k`, different chunking. **No prompt change can fix this.** |
| **F_GEN** — generation failure | Evidence *was* retrieved, but the answer is still wrong. | the generator prompt or model — retrieval is fine |

Two properties make this trustworthy:

1. **The F1 test is deterministic** — a substring check (`chunk_matches_snippets`), no LLM. Free, reproducible, gateable in CI.
2. **Attribution only runs on queries a separate correctness trigger already marked failed.** A low faithfulness score is *not* a failure trigger — see [Metrics ≠ correctness](#metrics--correctness).

An LLM writes the human-readable note. It never decides the class.

*(The finer F2/F3 split — "ignored the evidence" vs "engaged and erred" — is a deliberate roadmap item. It needs a claim-provenance judge and is inherently non-deterministic, so it doesn't gate the shipped engine. See `RetrievalBench_Design.md` §5.10.1.)*

---

## Status

**Complete.** The engine runs end-to-end: ingest → retrieve (dense/hybrid) → rerank → generate → score → diagnose → recommend, gated by CI.

Measured results, the per-query-type failure breakdown, and the caveats that come with them live in **[`evals/RESULTS.md`](evals/RESULTS.md)** — kept out of this file so it stays a guide rather than a lab notebook.

---

## Setup

| Requirement | Why |
|---|---|
| **Python 3.13+** | pinned in `.python-version` |
| **[uv](https://docs.astral.sh/uv/)** | the only supported package manager — never pip/poetry/conda |
| **Docker** | runs Qdrant locally |
| **OpenAI API key** | embeddings, generation, and the LLM judge |

```bash
git clone git@github.com:Kalp-Chaniyara/Retrievalbench.git
cd Retrievalbench
uv sync
```

Create `.env` in the project root (gitignored — never commit it):

```bash
OPENAI_API_KEY=sk-...
DEEPEVAL_TELEMETRY_OPT_OUT=YES
DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE=90
```

Start Qdrant:

```bash
docker compose up -d
```

Sanity check: http://localhost:6333/dashboard

---

## Run it

Corpora are committed under `data/corpora/<corpus_id>/`, and indexing happens automatically on first run — no manual ingest step. The active corpus is `CORPUS_ID` in `cli.py`.

```bash
uv run rbench run
```

Chunking, embedding, and upserting into Qdrant are handled by the index cache, keyed on `(corpus, chunking, embedding[, sparse])`. Change the retriever and it reuses the vectors; change the chunk size and it re-indexes.

### Commands

| Command | What it does |
|---|---|
| `rbench run --config configs/baseline.yaml` | run one config over the golden set, score, diagnose, persist |
| `rbench report <run_id>` | per-query failure table (F1 / F_GEN) + notes + headline |
| `rbench compare <run_a> <run_b>` | metric deltas between two runs |
| `rbench recommend` | rank measured configs on quality/cost/latency, with caveats |
| `rbench gen-golden --n 8` | generate golden items per query type, review keep/edit/drop |

### `rbench recommend`

Ranks every measured config on quality/cost/latency and justifies the pick:

```
<corpus> — N config(s) · M golden items
     config      pass rate   F1  F_GEN  latency   cost/run
★ $  ...
✗    ...

★ recommended · $ cheapest (reference) · ✗ Pareto-dominated
```

Two rules it enforces, both load-bearing:

- **Quality is the correctness pass rate**, never the mean of the four metrics. A pipeline that answers *"I don't know"* to everything scores **faithfulness 1.0** (DeepEval returns a hardcoded `1` when the answer yields zero claims) and is the cheapest and fastest — it would win on three of four axes. Only ranking on pass rate rejects it.
- **Quality in percentage points, cost/latency as ratios.** 50% → 83% is `+33 points`, not "+66%".

It also reports **Pareto domination**, a **diminishing-returns** callout, a **confound warning** when the winner differs from the reference in several dimensions at once, and a **resolution caveat** when a gap is smaller than one golden query.

---

## How it fits together

| Stage | Code | Default |
|---|---|---|
| Load `.txt/.md/.rst/.pdf` → `Document` | `ingest/loader.py` | `document_id = sha256(bytes)[:16]` |
| Chunk | `ingest/chunkers.py` | `FixedSizeChunker` / `RecursiveChunker` (token-based) |
| Embed (batched) | `retrieval/embedders.py` | `text-embedding-3-small`, dim 1536 |
| Index (cached) | `ingest/index.py` | collection keyed on chunking + embedding |
| Store | `retrieval/store.py` | Qdrant; dense `semantic` + sparse `text` vectors |
| Retrieve | `retrieval/retrieval.py` | dense, or hybrid with **RRF fusion on rank** (`rrf_k=60`) |
| Rerank (optional) | `retrieval/rerankers.py` | `bge-reranker-v2-m3` cross-encoder |
| Generate | `generate.py` | `gpt-4o-mini`, `temperature=0`, answer-only-from-context |
| Score | `eval/metric.py` | 4 DeepEval metrics via one pooled client |
| Diagnose | `eval/diagnostics.py` | GEval correctness trigger → F1/F_GEN cascade → note |
| Recommend | `recommend.py` | pass-rate ranking + Pareto + budget |
| Persist | `storage.py` | SQLite (`RunStore`, `GoldenStore`) |

Domain types live in `model.py` — the single source of truth, reused everywhere.

### Model split (deliberate — don't collapse it)

- **Judge: `gpt-4o`.** The judge must be stronger than / a different family from the generator, or it grades its own family too leniently. `gpt-4o-mini` as judge also produced false contradictions *and* could hang DeepEval's faithfulness-verdicts call indefinitely.
- **Everything else: `gpt-4o-mini`** — the RAG generator (the system under test), the diagnostics note writer, the golden generator.

Never "save cost" by moving the judge to mini; never upgrade the generator to `gpt-4o` — the generator *is* what's being measured, so changing it breaks comparability with every saved run.

### Golden ground truth is snippets, not chunk ids

`GoldenItem.expected_snippets` holds **verbatim source text**, not chunk ids. A chunk id like `docid_0007` encodes *position*, which shifts the moment you change chunk size — so it can't be ground truth for a benchmark whose whole point is varying chunking. Snippets are config-stable: a chunk is a hit if it contains a snippet after whitespace/case normalization, resolved per config at eval time.

### Query types drive the finding

Each `GoldenItem` carries a `query_type` — `exact_match`, `semantic`,
`negation`, or `multi_hop` — and `rbench report` breaks F1/F_GEN down by it.
That per-type split is the point: "F1 fires on 50% of multi_hop and 14% of
negation" tells you which retrieval mode is failing, where a single aggregate
F1 rate averages the signal away.

The type is an **input** to generation, not a label the LLM assigns to its own
output — `gen-golden` asks for an exact-match query and gets one, so the type
is true by construction rather than inferred.

### Metrics ≠ correctness

The four metrics measure different things, and **none of them measures correctness**:

- **faithfulness** — is the answer grounded in the retrieved context (hallucination check)?
- **answer_relevancy** — does it address the question?
- **context_precision** — are relevant chunks ranked above the noise? *(never sees the answer)*
- **context_recall** — is everything the gold answer needs present? *(never sees the answer)*

A wrong-but-abstaining answer (`"I don't know."`) can score **1.000 on all four**. That's why the diagnostics engine uses a separate GEval correctness check against `expected_answer` as its failure trigger, and why `recommend` ranks on pass rate.

---

## CI

Two workflows, deliberately split because LLM-judge scores aren't reproducible — an absolute threshold would flake, and a flaky gate gets ignored.

| Workflow | Trigger | Contains | Cost |
|---|---|---|---|
| `ci.yml` | **every PR** | ruff, mypy, deterministic tests | free, ~40s |
| `eval.yml` | **`run-eval` label** or manual dispatch | tests first, then the golden set vs a baseline | $ |

Tier 1 gates the deterministic half: the F1 snippet hit-test, the F1→F_GEN cascade, the recommendation ranking policy, config validation, and cost arithmetic. It also verifies **every golden snippet exists verbatim in the corpus** — a typo'd snippet silently matches nothing and quietly weakens the F1 gate.

`ci.yml` additionally warns when a PR touches code that can change model output (`generate.py`, `retrieval/`, `eval/`, `chunkers.py`, `configs/`), suggesting the `run-eval` label — so nobody has to memorise which paths matter.

```bash
uv run pytest tests/          # no API calls, seconds
```

---

## Scope

**Shipped:** dense + hybrid (RRF) retrieval, cross-encoder reranking, the four
DeepEval metrics, F1/F_GEN diagnostics with per-query-type breakdown, the
recommendation engine, cost accounting, and a two-tier CI gate.

**Deliberately not built:** `QueryRewriteRetriever` / `MultiQueryRetriever`, the
F2/F3 sub-split of F_GEN, a web UI, and PyPI packaging. Bounded-semaphore
concurrency was measured and rejected — the eval loop is token-bound, not
latency-bound, so parallelism reaches the rate limit sooner without finishing
earlier (see [`evals/RESULTS.md`](evals/RESULTS.md)).

## Cost

Per query: 1 embedding call + 1 generation call (`gpt-4o-mini`) + 4 judge metrics + a correctness trigger (`gpt-4o`). The judge dominates. `cost_usd` tracks **pipeline cost only** — generation tokens — because judge spend is measurement overhead you don't pay in production.

---

## Troubleshooting

- **`TimeoutError: call timed out after Ns`** — a stalled OpenAI request. Fail fast: `export DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE=45`. Note DeepEval's requests are non-streaming, so a hang in `_receive_response_headers` means generation hasn't finished — not that the connection broke.
- **Connection refused on `localhost:6333`** — Qdrant isn't up: `docker compose up -d`.
- **All chunks show ✗** — your `expected_snippets` don't match the corpus text. Run `uv run pytest tests/test_golden.py` to find which.
- **`SSLCertVerificationError`** — a TLS-intercepting proxy/VPN. Disable it for the run, or `export SSL_CERT_FILE=/path/to/ca.pem`.
- **`rbench: command not found`** — use `uv run rbench ...`, or `uv sync` first.
- **Debugging a hang:** `ps aux | grep rbench` (0.0% CPU = blocked on I/O), `lsof -a -p <pid> -i -n -P` (the `-a` is required), `sudo uvx py-spy dump --pid <pid>`.

---

## Project layout

```
src/retrievalbench/
  model.py              # all Pydantic domain models (source of truth)
  config.py             # YAML -> RetrievalConfig, extra="forbid"
  cli.py                # rbench: run, report, compare, recommend, gen-golden
  runner.py             # per-query orchestration -> ExperimentRun
  recommend.py          # recommendation engine (pure, no I/O)
  generate.py           # OpenAIGenerator -> (answer, cost)
  golden.py             # snippet hit-test + LLM golden generator
  storage.py            # SQLite: RunStore, GoldenStore
  ingest/               # loader.py, chunkers.py, index.py
  retrieval/            # embedders.py, store.py, retrieval.py, rerankers.py
  eval/                 # metric.py (Scorer), diagnostics.py (F1/F_GEN)
tests/                  # deterministic tests, no API calls
scripts/check_baseline.py
data/corpora/<corpus_id>/    # committed corpora
evals/                  # RESULTS.md + baseline.json
configs/                # experiment YAMLs
.github/workflows/      # ci.yml (tier 1), eval.yml (tier 2)
```

Full spec: `RetrievalBench_Design.md` (architecture, data model, §5.10 diagnostics rules) and `RetrievalBench_Roadmap.md` (phase-by-phase build order).
