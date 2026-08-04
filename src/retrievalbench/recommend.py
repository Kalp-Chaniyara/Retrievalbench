"""Recommendation engine (Design §5.11) — compare measured runs and recommend a
config across quality / cost / latency.

Read-only: it never executes a run. It can only choose among configs that have
actually been measured, so `rbench run` each config first.

**Quality is the correctness PASS RATE, not the mean of the four metrics.** The
metrics don't measure correctness: faithfulness scores a claim-free "I don't
know" as 1.0 (deepeval returns 1 when there are zero verdicts), and
context_precision/recall never even see the answer. Ranking on their mean would
crown a pipeline that abstains on every query. The pass rate comes from the
diagnostics correctness trigger (answer vs expected_answer), so refusals count
as failures — which is the whole reason the wedge exists.
"""

from retrievalbench.config import RetrievalConfig
from retrievalbench.model import (
    Budget,
    ConfigCandidate,
    ExperimentRun,
    FailureMode,
    Recommendation,
)

# Diminishing-returns rule, stated so it can be argued with rather than hidden:
# a win that buys little quality for a lot more money/time gets called out.
DIMINISHING_QUALITY_POINTS = 5.0
DIMINISHING_RATIO = 1.5

# Config fields that differ without changing the pipeline's behaviour, so they
# never count as a "dimension" when explaining what two configs differ by.
_DIFF_IGNORED = {"name", "seed"}


def _flatten(prefix: str, value: object) -> dict[str, object]:
    if isinstance(value, dict):
        out: dict[str, object] = {}
        for key, sub in value.items():
            if not prefix and key in _DIFF_IGNORED:
                continue
            out.update(_flatten(f"{prefix}.{key}" if prefix else str(key), sub))
        return out
    return {prefix: value}


def config_diff(a: RetrievalConfig, b: RetrievalConfig) -> list[str]:
    """Dotted names of every field two configs differ on. Load-bearing for
    honesty: if the winner differs in 5 dimensions at once you CANNOT attribute
    its gain to any single one, and the report has to say so."""
    flat_a = _flatten("", a.model_dump())
    flat_b = _flatten("", b.model_dump())
    return sorted(
        key
        for key in flat_a.keys() | flat_b.keys()
        if flat_a.get(key) != flat_b.get(key)
    )


def _candidate(run: ExperimentRun) -> ConfigCandidate:
    f1 = sum(1 for e in run.evaluations if e.failure_mode is FailureMode.RETRIEVAL_MISS)
    f_gen = sum(
        1 for e in run.evaluations if e.failure_mode is FailureMode.GENERATION_FAILURE
    )
    passed = sum(1 for e in run.evaluations if e.failure_mode is FailureMode.NONE)
    return ConfigCandidate(
        run_id=run.id,
        config_name=run.config.name,
        config=run.config,
        total_queries=len(run.evaluations),
        passed=passed,
        f1_count=f1,
        f_gen_count=f_gen,
        mean_latency_ms=run.aggregate.get("mean_latency_ms", 0.0),
        cost_per_run_usd=run.aggregate.get("total_cost_usd", 0.0),
        faithfulness=run.aggregate.get("faithfulness", 0.0),
    )


def latest_per_config(runs: list[ExperimentRun]) -> list[ExperimentRun]:
    """One run per config — the most recent. Re-running the same config should
    refine its number, not give it extra entries in the ranking."""
    newest: dict[str, ExperimentRun] = {}
    for run in runs:
        current = newest.get(run.config.name)
        if current is None or run.created_at > current.created_at:
            newest[run.config.name] = run
    return sorted(newest.values(), key=lambda r: r.config.name)


def _dominated_names(candidates: list[ConfigCandidate]) -> list[str]:
    """Pareto: X is dominated if some Y is at least as good on ALL of
    (quality up, cost down, latency down) and strictly better on one."""
    dominated: list[str] = []
    for x in candidates:
        for y in candidates:
            if y.run_id == x.run_id:
                continue
            no_worse = (
                y.pass_rate >= x.pass_rate
                and y.cost_per_run_usd <= x.cost_per_run_usd
                and y.mean_latency_ms <= x.mean_latency_ms
            )
            strictly_better = (
                y.pass_rate > x.pass_rate
                or y.cost_per_run_usd < x.cost_per_run_usd
                or y.mean_latency_ms < x.mean_latency_ms
            )
            if no_worse and strictly_better:
                dominated.append(x.config_name)
                break
    return dominated


def _ratio(new: float, old: float) -> float | None:
    """None when the baseline is 0 — a ratio against zero is undefined, and
    printing 'inf×' or silently showing 1.0 would both be lies."""
    return new / old if old > 0 else None


def _next_step(winner: ConfigCandidate) -> list[str]:
    """Turn the winner's remaining failure mix into an actionable next move —
    this is the wedge feeding the recommendation."""
    notes: list[str] = []
    if winner.failed == 0:
        notes.append("No failures left on this golden set — grow it to keep testing.")
        return notes
    if winner.f1_count:
        notes.append(
            f"{winner.f1_count} F1 (retrieval miss) remain: evidence never reached "
            f"the generator. Try hybrid retrieval, a larger top_k_retrieve, or "
            f"different chunking — prompt/model changes cannot fix these."
        )
    if winner.f_gen_count:
        notes.append(
            f"{winner.f_gen_count} F_GEN remain: evidence WAS retrieved but the "
            f"answer was still wrong. Look at the generator prompt/model, "
            f"not retrieval."
        )
    return notes


class ComparabilityError(ValueError):
    """Runs that can't be honestly compared (different corpus or golden set)."""


def recommend(
    runs: list[ExperimentRun], budget: Budget | None = None
) -> Recommendation:
    """Rank measured configs and justify a choice. Pure function over saved runs
    — no I/O, no API calls — so it is unit-testable without executing a pipeline.
    """
    budget = budget or Budget()
    if not runs:
        raise ComparabilityError("No saved runs — run `rbench run` first.")

    corpora = {r.corpus_id for r in runs}
    if len(corpora) > 1:
        raise ComparabilityError(
            f"Runs span multiple corpora ({sorted(corpora)}); scores are not "
            f"comparable across corpora. Filter to one with --corpus-id."
        )

    chosen = latest_per_config(runs)
    sizes = {len(r.evaluations) for r in chosen}
    if len(sizes) > 1:
        raise ComparabilityError(
            f"Runs were scored on different golden-set sizes ({sorted(sizes)}). "
            f"Pass rates are not comparable — re-run every config on the same "
            f"golden set."
        )

    candidates = [_candidate(r) for r in chosen]
    feasible = [c for c in candidates if budget.allows(c)]
    infeasible = [c for c in candidates if not budget.allows(c)]

    # Rank: pass rate first (the trustworthy signal), faithfulness only to break
    # ties, then prefer the cheaper config when quality is genuinely equal.
    feasible.sort(key=lambda c: (-c.pass_rate, -c.faithfulness, c.cost_per_run_usd))

    rec = Recommendation(
        corpus_id=next(iter(corpora)),
        golden_set_size=next(iter(sizes)),
        candidates=feasible,
        infeasible=infeasible,
        dominated=_dominated_names(feasible),
    )

    if not feasible:
        rec.notes.append(
            "No config fits the budget. Loosen --max-latency-ms/--max-cost-usd, "
            "or see the over-budget table for the closest options."
        )
        return rec

    winner = feasible[0]
    # Reference = cheapest feasible: the "do nothing / spend least" option the
    # extra cost of the winner has to justify itself against.
    reference = min(feasible, key=lambda c: (c.cost_per_run_usd, c.mean_latency_ms))
    rec.winner = winner
    rec.reference = reference

    if len(feasible) == 1:
        rec.notes.append(
            "Only one config has been measured for this corpus — nothing to "
            "compare against. Run another config, then re-run `rbench recommend`."
        )
        rec.notes.extend(_next_step(winner))
        return rec

    if winner.run_id != reference.run_id:
        rec.quality_gain_points = (winner.pass_rate - reference.pass_rate) * 100
        rec.cost_ratio = _ratio(winner.cost_per_run_usd, reference.cost_per_run_usd)
        rec.latency_ratio = _ratio(winner.mean_latency_ms, reference.mean_latency_ms)
        pricier = (rec.cost_ratio or 1.0) > DIMINISHING_RATIO or (
            rec.latency_ratio or 1.0
        ) > DIMINISHING_RATIO
        rec.diminishing_returns = (
            rec.quality_gain_points < DIMINISHING_QUALITY_POINTS and pricier
        )
        rec.confounded_dimensions = config_diff(winner.config, reference.config)

    rec.notes.extend(_next_step(winner))
    return rec
