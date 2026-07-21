# RetrievalBench

A local-first, config-driven harness for running, evaluating, and **diagnosing** retrieval (RAG) pipelines. Primary purpose is learning retrieval + evaluation deeply by building it from scratch; secondary is a clean, explainable portfolio/OSS artifact.

**The differentiator ("the wedge"):** per-query failure attribution. Every failed query is labelled with one of three canonical RAG failure modes via a *deterministic rules engine* (an LLM only writes the human-readable note, never the classification):
- **F1 — retrieval miss:** none of the expected chunks were retrieved.
- **F2 — generation ignore:** the right chunk was retrieved but the answer didn't use it.
- **F3 — generation error:** the model used the chunk but still answered wrong.

This is NOT trying to beat AutoRAG / RAGAS / MLflow. Don't frame it as novel. Frame it as a from-scratch retrieval-eval harness with failure diagnostics.

## Full spec lives in two docs (read on demand, don't duplicate)
- `RetrievalBench_Design.md` — architecture, data model, per-component design, the F1/F2/F3 rules (§5.10), folder layout (§6), config schema (§7), phasing (§9). **Read this before implementing any component.**
- `RetrievalBench_Roadmap.md` — phase-by-phase build order with definitions of done.

## Current state (read before suggesting next steps)
Phase 2. Build direction is **back-to-front along the main path**.

Next, in order:
2. `rerankers.py`: `Reranker` using **`bge-reranker-v2-m3` via the `rerankers` library** (one API, swappable).
3. `eval/diagnostics.py`: **F1/F2/F3 rules engine** (Design §5.10) — deterministic labels + plain-language note per failed query, plus an aggregate summary.
4. `golden.py`: LLM-based golden generator (question / expected answer / `expected_snippets` — verbatim source spans, never chunk ids) + a CLI review step (keep/edit/drop).
5. `cli.py`: `rbench report run_id` prints the diagnosis ("62% of failures are F1 → try hybrid").

**We talk to `AsyncOpenAI` directly** via `chat.completions.create`. (An earlier plan to route it through a separate LLM-wrapper library has been dropped — ignore any reference to that.)

## Stack (decided — do not substitute)
- Python 3.11+, async-native.
- **uv** for everything — `uv add`, `uv sync`, `uv run`. Never pip/poetry/conda.
- **Pydantic v2** domain models are the source of truth (`src/retrievalbench/models.py`).
- **src/ layout.** Build backend is **Hatchling**; editable install via `uv sync`.
- Vector store: **Qdrant**, local via Docker.
- Eval: **RAGAS** (faithfulness, answer relevancy, context precision, context recall).
- Chunking: **Chonkie** preferred (or standalone `langchain-text-splitters`); not full LangChain.
- CLI: **Typer**. Later phases: FastAPI backend + React/Vite frontend.

## Conventions & hard rules (these are load-bearing — past bugs)
- **Imports are `from retrievalbench...`**, never `src.retrievalbench...`. The `src.` prefix breaks under src layout.
- **Folder only when ≥2 cohesive files belong together** (`ingest/`, `retrieval/`, `eval/`); flat single files otherwise (`generate.py`, `golden.py`, `storage.py`). Split a file into a folder when it grows a second file, not before.
- **Pydantic v2 / modern syntax only:** `X | None`, built-in generics (`list[str]`, `dict[str, str]`). Never `Optional`/`List` from `typing`.
- **API keys come from environment variables** via `pydantic-settings` `BaseSettings`. `AsyncOpenAI()` auto-reads `OPENAI_API_KEY` — do not pass the key explicitly. Never put secrets in YAML or domain models; never log or serialize them.
- **Class vs function rule:** use a class when a component has multiple methods sharing one config, needs construction-time validation, or must be a named domain type. **Config binds once at construction (`__init__`); input varies per call.** A constant like `name = "dense"` is a class attribute, not a constructor arg.
- **Async hygiene:** `await` every client call. Never use a sync client inside an `async` method. Extract the field you need (`.embedding`, `response.choices[0].message.content`) — never append the whole response object. **Batch** embedding calls; don't loop one HTTP request per text.
- **Reproducibility is non-negotiable (G4):** generation runs at `temperature=0`; seed everything seedable.
- **Golden ground truth is verbatim source *snippets*, not chunk ids** (`GoldenItem.expected_snippets`). A chunk's `docid_NNNN` id encodes *position*, which shifts with chunk size — so it can't be ground truth for a benchmark that varies chunking. Snippets are config-stable source text; a chunk is a retrieval **hit** if it contains a snippet after whitespace/case normalization, resolved per config via `golden.chunk_matches_snippets` / `hit_chunk_ids` (source docs hard-wrap, so normalize both sides). This is the **F1 (retrieval-miss) gate**; F2/F3 gate on it. Keep snippets short + distinctive so a chunk boundary can't split the match.
- **Metrics ≠ correctness — and LLM-judge scores are noisy.** DeepEval `faithfulness` measures grounding of the answer's claims against the *retrieved context* (a hallucination check), not correctness vs `expected_answer`; with few claims it's near-binary, so one bad judge verdict → 0.0. A weak judge (`gpt-4o-mini`, `cli.py:JUDGE_MODEL`) produces exactly these false contradictions. Trust the **aggregate mean across the golden set** (what `compare` reads), not any single-query number — and this is *why F1/F2/F3 classification stays a deterministic rules engine, the LLM only writing the note.*

## Qdrant specifics (already-hit gotchas)
- **Point IDs must be UUID or unsigned int** — not arbitrary strings. Use a deterministic **UUID5 derived from chunk content**, and keep the human-readable chunk id in the **payload**.
- Use explicit `url="http://localhost:6333"`, not a bare hostname.
- Guard `create_collection` with `collection_exists` for **idempotency**.
- `query_points` is a coroutine: `await` it into a variable, *then* read `.points`. Never `await client.query_points(...).points`.
- **Same embedder for chunks and query** — dense search compares vectors in one shared space; mixing embedders makes distances meaningless.
- Payload carries all metadata back through retrieval; vectors only enable the similarity search.

## Run Qdrant
```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
  qdrant/qdrant
```
The `-v` volume mount persists data to `qdrant_storage/` at the **project root** (alongside `src/` and `data/`, not inside `src/`). `qdrant_storage/`, `data/corpora/`, and `.env` are all gitignored.

## Build philosophy / scope guards
- **Thin slice first.** In Phase 0, hardcode aggressively. Get one straight line to a real score before adding any abstraction.
- **Don't build ahead of the main path:** no golden-set generator, diagnostics engine, recommendation engine, UI, or plugin registry until Phase 0 prints 4 numbers on real data.
- **No GraphRAG / semantic chunking / extra embedders mid-build** — explicitly later-tier.
- When adding the API/UI later: reuse the Pydantic models as FastAPI response schemas; never hand-write a parallel DTO.

## Working style
Be direct and correction-oriented: explicit "this is wrong because…" with reasoning, not reassurance. When a design choice is contested, state the tradeoff. This is a learning project — when something is a genuine design decision (class vs function, return type, prompt design), explain the options and the rule rather than silently picking; the reasoning is the point.
