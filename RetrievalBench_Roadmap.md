# RetrievalBench — Learning + Build Roadmap

> Companion to the Design doc. This is the *execution* plan: what to learn, what to build, in what order, with a done-check at every step.
> Method: **build while learning** (implementation-first), never learn-all-then-build.

**Total realistic budget:** ~6–8 weeks part-time (engine ~3.5 wks · UI ~2 wks · PyPI ~0.5 wk).
**Stop-anywhere-safe rule:** Phases 0–2 are the valuable core. Everything after is polish. If you stop after Phase 2 you still have a strong, defensible project.

---

## 0. How to use this roadmap

Each phase has two blocks:
- **LEARN** — time-boxed reading for a *mental model only*. Official docs first. The bar is the roadmap rule: *you've learned enough when you can say when to use and when NOT to use the thing.* Do not read past the timebox.
- **BUILD** — the concrete thing you implement right after, then **MEASURE**.

The micro-loop you repeat all project long:
```
read 30–60 min  →  build the one thing  →  run it / measure  →  write one sentence on what you learned  →  next
```
Never let LEARN exceed BUILD. If you've read for an hour and built nothing, stop reading and start typing.

---

## 1. Stack & toolchain (decided)

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | your primary; async-native |
| Package/build/publish | **uv** | single tool; fast; native Trusted Publishing |
| Models | Pydantic v2 | your existing pattern; reused as FastAPI schemas |
| Vector DB | Qdrant (Docker) | dense + sparse in one engine; "pick one, go deep" |
| Eval | RAGAS (+ DeepEval later) | standard RAG metrics |
| Backend API | **FastAPI** | async + Pydantic-native + free OpenAPI docs |
| Frontend | **React + Vite** | your MERN background; fast tooling; resume value |
| Charts | Recharts | simple, React-native |
| CLI | Typer | typed, minimal |

### The one decision to make consciously: how the UI ships in the pip package
- **(A) Bundle built React into the wheel.** Run `npm run build` → ship the static `dist/` as package data → FastAPI serves it via `StaticFiles` → `pip install retrievalbench` then `rbench ui` opens the bundled app. **Recommended.** Best UX; one install. Cost: release pipeline must run an npm build before `uv build`, and the wheel is larger.
- **(B) Python-only package, UI runs from source.** Simpler package, but pip users don't get the UI. Fine if the UI is "dev-only."
- **(C) Two separate packages.** Overkill for a solo local tool. Skip.

> **How to decide:** if the UI is part of the product story you want on your resume, choose **A** — a one-command installable tool *with* a UI is a stronger artifact than a library + a repo you have to clone. Accept the npm-build step in the release pipeline as the price.

---

## 2. Phase −1 — Foundations & setup · ~2–3 days

The *only* upfront learning worth doing before building.

**LEARN (≈2 hrs total):**
- uv: `uv init`, `uv add`, `uv run`, src layout, `pyproject.toml`. (30 min, official uv docs)
- Docker: just enough to `docker run` Qdrant and stop it. (20 min)
- Qdrant concepts: collection, point, vector, payload, search. (30 min, Qdrant quickstart)
- Pydantic v2: light refresher only — you know this. (15 min)

**BUILD:**
1. `uv init retrievalbench` with `src/` layout (per Design §6).
2. `pyproject.toml`: name, version `0.0.1`, Python pin, deps stubs.
3. Create the empty package skeleton (folders/files from Design §6, mostly empty).
4. `docker run` Qdrant locally; confirm the dashboard loads.
5. Drop in `models.py` from Design §4 (Pydantic models). They compile = your spine exists.
6. Git init, first commit, push to GitHub.

**DONE-CHECK:** `uv run python -c "import retrievalbench"` works; Qdrant is reachable; models import.

---

## 3. Phase 0 — Thin slice (one straight line to a score) · ~3–5 days

Goal: documents → a RAGAS number. Hardcode everything you can.

**LEARN (≈1.5 hrs):**
- Embeddings + cosine similarity — what a vector *is* and why similar text is close. (30 min)
- Chunking basics — why you split, what overlap does. (20 min)
- RAGAS quickstart + the 4 metrics — faithfulness, answer relevancy, context precision, context recall. (45 min, RAGAS docs)

**BUILD (back-to-front along the main path):**
1. `loaders.py`: load 3–5 PDFs/TXT → `Document`.
2. `chunkers.py`: `FixedSizeChunker(size, overlap)` → `list[Chunk]`.
3. `embedders.py`: one embedder (local `bge-small-en-v1.5` recommended — free iteration).
4. `store.py`: upsert vectors to Qdrant; `dense_search`.
5. `retrievers.py`: `DenseRetriever` (top-k).
6. `generate.py`: generator via **your own LLM wrapper** (chat.completions shape) — first dogfood.
7. **Hand-write 10–15 golden items** (`GoldenItem`) yourself.
8. `eval/metrics.py`: run RAGAS over the answers.
9. Minimal `cli.py`: `rbench run` that wires the above and prints the 4 means.

**MEASURE / DONE-CHECK:** `rbench run` prints 4 numbers on real data.
**Learning checkpoint:** you can explain, on a concrete example, what each RAGAS metric measured.

---

## 4. Phase 1 — Real MVP (configurable + persistent + comparable) · ~1 week

**LEARN (≈1.5 hrs):**
- Recursive chunking (paragraph→sentence→token). (20 min)
- Config-driven design — YAML → validated Pydantic config. (20 min)
- SQLite basics + storing JSON columns. (30 min)
- Reproducibility — seeds, caching, why determinism matters in eval. (20 min)

**BUILD:**
1. `chunkers.py`: add `RecursiveChunker`.
2. `config.py`: load YAML → `RetrievalConfig`; make chunker/embedder/retriever swappable by config.
3. `storage.py`: SQLite; persist `ExperimentRun` (+ results, evaluations).
4. `runner.py`: loop the golden set for one config → `ExperimentRun` → save.
5. **Index cache**: key the Qdrant collection on `(chunking, embedding)` so you don't re-embed across retrieval variants.
6. `cli.py`: `rbench compare run_a run_b` reads from storage, prints metric deltas.

**MEASURE / DONE-CHECK:** run fixed-512 vs recursive-800 from two YAMLs; read the delta from `compare`.
**Learning checkpoint:** you can state when fixed vs recursive chunking wins — from your own data.

---

## 5. Phase 2 — The wedge (hybrid + rerank + diagnostics) · ~1.5 weeks

This is the highest-value phase. Don't skip the diagnostics — it's the differentiator.

**LEARN (≈2 hrs):**
- BM25, sparse vectors, Reciprocal Rank Fusion. (45 min)
- Bi-encoder vs cross-encoder reranking; the retrieve-50→rerank→top-5 pattern. (30 min)
- The 3 RAG failure modes (F1 retrieval-miss / F2 generation-ignore / F3 generation-error). (20 min)
- Golden-set generation pitfalls (leakage, too-easy questions). (20 min)

**BUILD:**
1. `store.py` + `retrievers.py`: `HybridRetriever` (Qdrant sparse + dense, RRF fusion).
2. `rerankers.py`: `Reranker` using **`bge-reranker-v2-m3` via the `rerankers` library** (one API, swappable).
3. `eval/diagnostics.py`: **F1/F2/F3 rules engine** (Design §5.10) — deterministic labels + plain-language note per failed query, plus an aggregate summary.
4. `golden.py`: LLM-based golden generator (question / expected answer / `expected_chunk_ids`) + a CLI review step (keep/edit/drop).
5. `cli.py`: `rbench report run_id` prints the diagnosis ("62% of failures are F1 → try hybrid").

**MEASURE / DONE-CHECK:** the report attributes failures by mode, and switching dense→hybrid measurably reduces F1.
**Learning checkpoint:** you can debug a RAG system by failure mode, not vibes. **This is the senior skill.**

---

## 6. Phase 3 — Polish + product thinking · ~1 week

**LEARN (≈1 hr):** Pareto/tradeoff reasoning; query-rewrite & multi-query retrieval; DeepEval CI-style assertions.

**BUILD:**
1. `recommend.py`: recommendation engine — best config within cost/latency budget, with diminishing-returns callouts.
2. `retrievers.py`: `QueryRewriteRetriever`, `MultiQueryRetriever`.
3. DeepEval **CI gate** on your own repo: GitHub Action runs the golden set on each PR, fails on metric regression.

**DONE-CHECK:** `rbench recommend` outputs a justified pipeline choice with cost/latency tradeoffs.
**Learning checkpoint:** you can answer "which RAG should I build for this corpus and why."

> Engine is now complete. If you stop here, you have the full learning value + a strong CLI project.

---

## 7. Phase 4 — The UI (FastAPI + React/Vite) · ~1.5–2 weeks

Build the API **first** (exposing data the engine already produces), then the React app on top. Read-only screens before any "trigger a run" action.

**LEARN (≈2.5 hrs — light, given MERN):**
- FastAPI: path operations, Pydantic response models, CORS, `StaticFiles`. (1 hr, FastAPI docs)
- Vite project setup + dev proxy to the API. (30 min)
- React data fetching (TanStack Query or plain fetch+state). (30 min refresher)
- Recharts basics for the leaderboard/metric charts. (30 min)

**BUILD — API layer (`src/retrievalbench/api/`):**
1. FastAPI app that **reuses your Pydantic models as response schemas** (no new DTOs).
2. Endpoints, read-only first:
   - `GET /runs` — list runs (leaderboard data)
   - `GET /runs/{id}` — full run + aggregate
   - `GET /runs/{id}/diagnostics` — per-query F1/F2/F3 + notes
   - `GET /runs/{id}/chunks` — chunk viewer data
   - `GET /runs/{id}/retrieval/{query_id}` — retrieved vs reranked chunks
3. Then one **write** endpoint: `POST /runs` to trigger a run from a config (background task).

**BUILD — React/Vite app (`frontend/`):**
4. **Leaderboard** — all runs, sortable by metric/cost/latency.
5. **Run comparison** — A vs B side by side.
6. **Chunk viewer** — document split + overlap visualized. (most illuminating screen)
7. **Retrieval viewer** — query → retrieved chunks vs reranked chunks side by side.
8. **Diagnosis screen** — failure breakdown + plain-language notes.

**DONE-CHECK:** open `localhost`, pick a run, see its diagnostics and the chunk/retrieval viewers in the browser.

> **Scope guard:** build only the screens above. No auth, no multi-user, no settings pages. Read-only viewers are the demo-able, learning-rich part; the trigger-a-run button is a bonus.

---

## 8. Phase 5 — Package & publish to PyPI · ~3–5 days

**LEARN (≈1.5 hrs):**
- `pyproject.toml` metadata, wheel vs sdist, entry points (the `rbench` command). (40 min)
- Semantic versioning; TestPyPI vs PyPI. (20 min)
- Trusted Publishing / OIDC from GitHub Actions (no tokens). (30 min, uv + PyPA docs)

**BUILD:**
1. Finalize `pyproject.toml`: metadata, classifiers, `[project.scripts] rbench = "retrievalbench.cli:app"`, dependency pins.
2. **Ship the UI (option A):** add an `npm run build` step; include the built `frontend/dist/` as package data; have `rbench ui` serve it via FastAPI `StaticFiles`.
3. Write `README.md` (frame it honestly per Design §10.5 — "a from-scratch retrieval-eval harness with failure diagnostics," not "beats AutoRAG") and add a `LICENSE`.
4. Add a **smoke test** that imports the package and runs `rbench --help` against the built wheel.
5. GitHub Actions release workflow (tag-triggered) using **uv + Trusted Publishing**:
   - `permissions: id-token: write`, `environment: pypi`
   - steps: checkout → setup-uv → `npm build` (for the bundled UI) → `uv build` → smoke-test the wheel → `uv publish`
6. Register the trusted publisher on **TestPyPI** first; publish there; install into a **fresh** environment and verify `rbench --help` + `rbench ui` work.
7. Register the trusted publisher on real **PyPI**; tag `v0.1.0`; cut a GitHub Release → workflow publishes.

**DONE-CHECK:** in a clean venv, `pip install retrievalbench` → `rbench --help` works → `rbench ui` serves the bundled UI.

---

## 9. Timeline at a glance

| Phase | Focus | Effort | Cumulative |
|---|---|---|---|
| −1 | Foundations & setup | 2–3 days | ~0.5 wk |
| 0 | Thin slice → first score | 3–5 days | ~1 wk |
| 1 | Configurable + persistent MVP | ~1 wk | ~2 wks |
| 2 | **Hybrid + rerank + diagnostics (the wedge)** | ~1.5 wks | ~3.5 wks |
| 3 | Recommendation + CI | ~1 wk | ~4.5 wks |
| 4 | UI (FastAPI + React/Vite) | ~1.5–2 wks | ~6.5 wks |
| 5 | PyPI release | 3–5 days | ~7 wks |

**Engine value lives in Phases 0–2.** UI (4) and PyPI (5) are distribution/polish — high resume value, low *new-learning* value. Order is deliberately stop-anywhere-safe.

---

## 10. Guardrails / anti-patterns

1. **Don't learn-all-first.** If you've read for an hour without building, stop and type.
2. **Don't build the UI before the engine produces real data.** The UI is a view over `ExperimentRun`s that already exist.
3. **Don't build the API speculatively.** Add an endpoint only when a screen needs it.
4. **Reuse Pydantic models as FastAPI response models.** Never hand-write a parallel schema.
5. **Don't publish to real PyPI until a clean-env install from TestPyPI works.** Test server first, always.
6. **Don't add GraphRAG / semantic chunking / more embedders mid-build.** They're Later-tier; they'll eat your timeline and teach little beyond what hybrid+rerank already taught you.
7. **Timebox every phase.** A finished Phase 2 beats a half-built Phase 4. Shipped > perfect.
8. **One sentence per learning checkpoint.** Write down what you learned each loop — it's what you'll repeat in interviews.

---

## 11. What to read, by topic (official-docs-first)

- **uv** — docs.astral.sh/uv (packaging + GitHub Actions guides)
- **Qdrant** — qdrant.tech docs (quickstart, hybrid/sparse, Query API)
- **RAGAS** — docs.ragas.io (metrics + testset generation)
- **rerankers lib** — github.com/AnswerDotAI/rerankers (read the model card for bge-reranker-v2-m3)
- **FastAPI** — fastapi.tiangolo.com (response models, StaticFiles, background tasks)
- **Vite** — vitejs.dev (guide + env/proxy)
- **PyPI Trusted Publishing** — docs.pypi.org + PyPA publishing guide

Read each *just-in-time* for the phase that needs it, not in advance.
