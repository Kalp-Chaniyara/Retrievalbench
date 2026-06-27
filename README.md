# RetrievalBench

A local-first, config-driven harness for running and **evaluating** retrieval (RAG) pipelines, built from scratch as a learning + portfolio project.

It wires one straight line end-to-end:

```
documents → chunk → embed → store (Qdrant) → retrieve top-k → generate answer → score (DeepEval)
```

and prints four RAG quality metrics — **faithfulness, answer relevancy, context precision, context recall** — per query and as means.

> **Status: Phase 0** (the thin slice). Lots is hardcoded on purpose: one embedder, one vector store, one generator, a hand-written golden set. The eventual differentiator — per-query failure attribution (F1 retrieval miss / F2 generation ignore / F3 generation error) — is **not built yet**. See `RetrievalBench_Design.md` and `RetrievalBench_Roadmap.md` for the full plan.

---

## What you need before you start

| Requirement | Why | Notes |
|---|---|---|
| **Python 3.13+** | runtime | `.python-version` pins 3.13 |
| **[uv](https://docs.astral.sh/uv/)** | the *only* supported package manager | never use pip/poetry/conda here |
| **Docker** | runs Qdrant (the vector store) locally | `docker compose` |
| **An OpenAI API key** | embeddings, generation, **and** the LLM judge all call OpenAI | this run costs real money — see [Cost](#a-note-on-cost) |

---

## Setup

### 1. Clone and install

```bash
git clone <your-fork-url> retrievalbench
cd retrievalbench
uv sync          # creates .venv and installs everything, including the `rbench` CLI
```

### 2. Add your API key

Create a `.env` file in the project root (it is gitignored — never commit it):

```bash
# .env
OPENAI_API_KEY=sk-...

# optional but recommended: silence DeepEval's telemetry phone-home
DEEPEVAL_TELEMETRY_OPT_OUT=YES
```

`AsyncOpenAI()` reads `OPENAI_API_KEY` from the environment automatically; the CLI loads `.env` via `load_dotenv()`.

### 3. Start Qdrant

```bash
docker compose up -d
```

This maps `./data` → the container's storage, so the vector DB persists to `data/` at the project root. Sanity check: open http://localhost:6333/dashboard.

---

## ⚠️ A fresh clone has no data — you must ingest your own

`data/` and `data/corpora/` are **gitignored**, so cloning gives you an empty corpus and an empty Qdrant. Before `rbench run` will work you must:

1. **Add documents.** Drop `.txt`, `.md`, or `.pdf` files into:
   ```
   data/corpora/sample_data1/
   ```

2. **Ingest them** (load → chunk → embed → upsert into Qdrant). The Phase-0 ingest is a throwaway script that is **not committed** (`src/scratch.py` is gitignored). Recreate it as `ingest.py` in the project root:

   ```python
   # ingest.py — load -> chunk -> embed -> store, then dump chunk ids
   import asyncio
   from pathlib import Path

   from dotenv import load_dotenv
   load_dotenv()

   from retrievalbench.ingest.chunkers import FixedSizeChunker
   from retrievalbench.ingest.loader import load_corpus
   from retrievalbench.retrieval.embedders import OpenAITextEmbedderSmall
   from retrievalbench.retrieval.store import QdrantStore

   CORPUS_DIR = "data/corpora/sample_data1"
   COLLECTION = "sample_data1"           # must match COLLECTION in cli.py
   DUMP_PATH = "data/corpora/sample_data1_chunks.md"
   EMBED_BATCH = 100

   async def main() -> None:
       docs = load_corpus(CORPUS_DIR)
       chunker = FixedSizeChunker(size=300, overlap=50)
       chunks = [c for d in docs for c in chunker.chunk(d)]
       print(f"loaded {len(docs)} docs -> {len(chunks)} chunks")

       embedder = OpenAITextEmbedderSmall()
       vectors: list[list[float]] = []
       for i in range(0, len(chunks), EMBED_BATCH):
           vectors.extend(await embedder.embed([c.text for c in chunks[i : i + EMBED_BATCH]]))

       store = QdrantStore(collection=COLLECTION, dim=embedder.dim)
       await store.upsert(chunks, vectors)
       print(f"upserted {len(chunks)} points into '{COLLECTION}'")

       # dump chunk_id + text so you can hand-write golden items
       lines = [f"## {c.id} (doc={c.document_id})\n{c.text.strip()}\n" for c in chunks]
       Path(DUMP_PATH).write_text("\n".join(lines), encoding="utf-8")
       print(f"wrote chunk dump -> {DUMP_PATH}")

   if __name__ == "__main__":
       asyncio.run(main())
   ```

   Run it:
   ```bash
   uv run python ingest.py
   ```

3. **Write your golden set.** `src/retrievalbench/golden.py` ships a hand-written `GOLDEN_SET` whose `expected_chunk_ids` are tied to the *original* corpus. A chunk id looks like `3a81559eaa45a5b2_0000` = `<document_id>_<index>`, where `document_id` is a hash of the **file's bytes** — so **your** ids will differ. Open the chunk dump from step 2, then rewrite each `GoldenItem` (`query`, `expected_chunk_ids`, `expected_answer`) against your own documents.

---

## Run it

```bash
uv run rbench run
```

You'll get a per-query panel — the query, the top-k retrieved chunks marked ✓/✗ against your golden `expected_chunk_ids`, the generated answer, and the four metric scores with the judge's reasoning — followed by a means table:

```
   Phase 0 — means over N queries
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ metric            ┃  mean ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ faithfulness      │ 1.000 │
│ answer_relevancy  │ 0.950 │
│ context_precision │ 1.000 │
│ context_recall    │ 1.000 │
└───────────────────┴───────┘
```

---

## How it fits together

| Stage | Code | Default |
|---|---|---|
| Load `.txt/.md/.pdf` → `Document` | `ingest/loader.py` | document_id = sha256(bytes)[:16] |
| Chunk → `Chunk` | `ingest/chunkers.py` | `FixedSizeChunker(size=300, overlap=50)` |
| Embed (batched) | `retrieval/embedders.py` | `text-embedding-3-small`, dim 1536 |
| Store | `retrieval/store.py` | Qdrant, cosine; point id = UUID5 of chunk id |
| Retrieve top-k | `retrieval/retrieval.py` | `query_points`, `TOP_K = 5` |
| Generate grounded answer | `generate.py` | `gpt-4o-mini`, `temperature=0`, "answer only from context" |
| Score | `eval/metric.py` | DeepEval 4 metrics, judged by `gpt-4o-mini`, run concurrently |
| Orchestrate + render | `cli.py` | `rbench run` |

Domain types (`Document`, `Chunk`, `RetrievedChunk`, `GoldenItem`, `EvalScores`, …) are Pydantic v2 models in `model.py` — the single source of truth.

**Run knobs** are constants at the top of `cli.py`:

```python
COLLECTION = "sample_data1"
JUDGE_MODEL = "gpt-4o-mini"
TOP_K = 5
```

### Understanding the metrics

- **faithfulness** — is the answer grounded in the retrieved context (no hallucination)?
- **answer_relevancy** — does the answer actually address the question?
- **context_precision** — a *ranking* metric: are relevant chunks ranked *above* irrelevant ones? It does **not** penalize fetching extra junk as long as the good chunk is near the top, so a perfect 1.0 with several "wasted" chunks below rank 1 is expected.
- **context_recall** — coverage: is everything the gold answer needs present somewhere in the retrieved chunks?

---

## A note on cost

Every `rbench run` makes, per query: 1 embedding call + 1 generation call + **4 LLM-judge metrics** (each spawning several sub-calls). With ~15 golden items that's well over a hundred OpenAI requests. It's cheap on `gpt-4o-mini` but not free — mind your usage.

---

## Troubleshooting

- **`SSLCertVerificationError: self-signed certificate in certificate chain`** — a TLS-intercepting proxy/VPN/antivirus is on your network. If it only hits PostHog telemetry, ignore it (set `DEEPEVAL_TELEMETRY_OPT_OUT=YES`). If it also stalls OpenAI calls (long hangs → `TimeoutError`/`RetryError`), disable the proxy/VPN for the run, or point Python at your corporate root CA: `export SSL_CERT_FILE=/path/to/ca.pem REQUESTS_CA_BUNDLE=/path/to/ca.pem`.
- **`TimeoutError: call timed out after Ns`** — network stall reaching OpenAI. Make it fail fast instead of hanging: `export DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE=30`.
- **Connection refused to `localhost:6333`** — Qdrant isn't up. `docker compose up -d`.
- **Empty / wrong results, all chunks ✗** — you haven't ingested, the collection name doesn't match `COLLECTION`, or your `golden.py` still has the original corpus's chunk ids.
- **`rbench: command not found`** — run inside the project venv via `uv run rbench run`, or `uv sync` first.

---

## Project layout

```
src/retrievalbench/
  model.py            # Pydantic domain models (source of truth)
  ingest/             # loader.py, chunkers.py
  retrieval/          # embedders.py, store.py, retrieval.py
  generate.py         # OpenAIGenerator
  golden.py           # hand-written GOLDEN_SET (rewrite for your corpus)
  eval/metric.py      # DeepEval metrics -> EvalScores
  cli.py              # `rbench run`
RetrievalBench_Design.md     # architecture & data model
RetrievalBench_Roadmap.md    # phase-by-phase build order
docker-compose.yaml          # Qdrant
```
