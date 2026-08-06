"""Tier-2 regression check: compare the newest saved run against a baseline.

Why a baseline with a TOLERANCE rather than an absolute threshold:
LLM-judge scores are not reproducible. The same code scored twice can give
faithfulness 0.91 then 0.87. `assert faithfulness >= 0.90` would therefore go
red at random, and a flaky gate gets ignored — which is worse than no gate. So
we record what the pipeline scored when we last accepted it, and fail only on a
drop LARGER than the noise band.

Usage:
    python scripts/check_baseline.py --config-name fixed_512            # check
    python scripts/check_baseline.py --config-name fixed_512 --record   # accept
"""

import argparse
import json
import sys
from pathlib import Path

from retrievalbench.storage import RunStore

BASELINE_PATH = Path("evals/baseline.json")

# Per-metric tolerance. Quality metrics get a wide band because the judge is
# noisy; pass rate is coarse (1 query on a small golden set is a big jump), so
# it is compared with a points band derived from the golden-set size instead.
TOLERANCES = {
    "faithfulness": 0.15,
    "answer_relevancy": 0.15,
    "context_precision": 0.15,
    "context_recall": 0.15,
}


def latest_run(store: RunStore, config_name: str):
    for run_id, name, _ in store.list_runs():  # already ordered newest-first
        if name == config_name:
            return store.get_run(run_id)
    return None


def summarize(run) -> dict:
    passed = sum(1 for e in run.evaluations if e.failure_mode.value == "none")
    total = len(run.evaluations)
    return {
        "config_name": run.config.name,
        "golden_set_size": total,
        "pass_rate": passed / total if total else 0.0,
        "faithfulness": run.aggregate.get("faithfulness", 0.0),
        "answer_relevancy": run.aggregate.get("answer_relevancy", 0.0),
        "context_precision": run.aggregate.get("context_precision", 0.0),
        "context_recall": run.aggregate.get("context_recall", 0.0),
        "total_cost_usd": run.aggregate.get("total_cost_usd", 0.0),
        "mean_latency_ms": run.aggregate.get("mean_latency_ms", 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument(
        "--record",
        action="store_true",
        help="Overwrite the baseline with the current run instead of checking.",
    )
    args = parser.parse_args()

    run = latest_run(RunStore(), args.config_name)
    if run is None:
        print(f"::error::no saved run found for config {args.config_name!r}")
        return 1

    current = summarize(run)
    print(f"current: {json.dumps(current, indent=2)}")

    # A run that cost nothing means cost accounting silently broke (the exact
    # bug that made total_cost_usd identically $0.00 for months).
    if current["total_cost_usd"] <= 0:
        print("::error::total_cost_usd is 0 — cost accounting is broken")
        return 1

    baselines = json.loads(BASELINE_PATH.read_text()) if BASELINE_PATH.exists() else {}

    if args.record:
        baselines[args.config_name] = current
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(baselines, indent=2) + "\n")
        print(f"recorded baseline for {args.config_name} -> {BASELINE_PATH}")
        return 0

    baseline = baselines.get(args.config_name)
    if baseline is None:
        print(
            f"::warning::no baseline for {args.config_name!r} yet. "
            f"Re-run with --record to accept the current numbers."
        )
        return 0

    if baseline["golden_set_size"] != current["golden_set_size"]:
        print(
            f"::warning::golden set changed "
            f"({baseline['golden_set_size']} -> {current['golden_set_size']}); "
            f"scores are not comparable. Re-record the baseline."
        )
        return 0

    failures = []
    for metric, tolerance in TOLERANCES.items():
        drop = baseline[metric] - current[metric]
        if drop > tolerance:
            failures.append(
                f"{metric}: {baseline[metric]:.3f} -> {current[metric]:.3f} "
                f"(drop {drop:.3f} > tolerance {tolerance})"
            )

    # Pass rate: one query is 100/n points, so anything smaller is unresolvable.
    # Only fail when MORE than one query regressed.
    resolution = 1.0 / current["golden_set_size"]
    pass_drop = baseline["pass_rate"] - current["pass_rate"]
    if pass_drop > resolution * 1.5:
        failures.append(
            f"pass_rate: {baseline['pass_rate']:.0%} -> {current['pass_rate']:.0%} "
            f"({pass_drop * 100:.0f} points; more than one query regressed)"
        )

    if failures:
        for line in failures:
            print(f"::error::REGRESSION {line}")
        return 1

    print("no regression beyond tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
