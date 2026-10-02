"""Scalar-vs-Pareto comparison for the "Scalar vs Pareto" page (pure: pandas and numpy only).

Pairs every task's scalar experiment with its suffixed arms ("AND" with "AND Pareto"), reduces
each run to the numbers that say how the search went, and decides a per-arm verdict against the
scalar arm. A run counts as a success when its best solution clears the target on every partial
fitness: the end condition evolution itself uses.
"""
import json
import math
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

from .pareto import constraint_violation, hypervolume, non_dominated_sort
from .stats import mann_whitney_u

SCALAR_ARM = "scalar"
FEASIBILITY_FLOOR = 0.1
ALPHA = 0.05


def pair_experiments(names: list[str]) -> dict[str, dict[str, str]]:
    """{task: {"scalar": name, <suffix>: name, ...}} for every experiment that has at least one
    suffixed sibling. The longest matching prefix wins, so "Memory Trace Pareto" pairs with
    "Memory Trace", not with some shorter experiment name."""
    tasks: dict[str, dict[str, str]] = {}
    for name in names:
        bases = [base for base in names if name.startswith(base + " ")]
        if bases:
            base = max(bases, key=len)
            tasks.setdefault(base, {SCALAR_ARM: base})[name[len(base) + 1:]] = name
    return tasks


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _last_jsonl_record(path: Path) -> dict:
    """The last record of a JSON-lines file without reading the whole file: objectives.jsonl
    holds every individual of every generation and can run to megabytes."""
    with path.open("rb") as f:
        f.seek(0, 2)
        position = f.tell()
        chunk = b""
        while position > 0:
            step = min(1 << 16, position)
            position -= step
            f.seek(position)
            chunk = f.read(step) + chunk
            lines = [line for line in chunk.splitlines() if line.strip()]
            if len(lines) > 1 or position == 0:
                return json.loads(lines[-1])
    raise ValueError(f"{path} is empty")


def _clears_target(partials: list[float], target: float) -> bool:
    return bool(partials) and all(p > target for p in partials)


def run_summary(run_dir: Path, target: float) -> dict | None:
    """How one run went, or None when it lacks overview.jsonl or objectives.jsonl."""
    run_dir = Path(run_dir)
    overview_path, objectives_path = run_dir / "overview.jsonl", run_dir / "objectives.jsonl"
    if not overview_path.exists() or not objectives_path.exists():
        return None
    overview = _read_jsonl(overview_path)
    if not overview:
        return None
    success_generation = next(
        (g["generation"] for g in overview if _clears_target(g["bestSolution"]["partialFitness"], target)), None)
    final_best = overview[-1]["bestSolution"]

    individuals = _last_jsonl_record(objectives_path)["individuals"]
    partials = np.array([i["partialFitness"] for i in individuals], dtype=float)
    objectives = np.clip(np.array([i["objectives"] for i in individuals], dtype=float), 0.0, 1.0)
    feasible = constraint_violation(partials, FEASIBILITY_FLOOR) == 0
    feasible_hv = 0.0
    if feasible.any():
        ranks = non_dominated_sort(objectives[feasible])
        feasible_hv = hypervolume(objectives[feasible][ranks == 0])[0]
    return {
        "success": success_generation is not None,
        "generations_to_success": success_generation,
        "generations": len(overview),
        "final_best_fitness": float(final_best["fitness"]),
        "final_best_min_partial": float(min(final_best["partialFitness"])),
        "best_generalist": float(partials.min(axis=1).max()),
        "feasible_hv": float(feasible_hv),
        "active_species": statistics.mean(g["numberOfActiveSpecies"] for g in overview),
    }


def best_trajectory(run_dir: Path) -> pd.DataFrame:
    """Per generation: the best solution's weighted fitness and its lowest partial."""
    overview = _read_jsonl(Path(run_dir) / "overview.jsonl")
    return pd.DataFrame({
        "generation": [g["generation"] for g in overview],
        "best_fitness": [g["bestSolution"]["fitness"] for g in overview],
        "best_min_partial": [min(g["bestSolution"]["partialFitness"]) for g in overview],
    })


def trajectory_band(trajectories: list[pd.DataFrame], value: str) -> pd.DataFrame:
    """Median and interquartile range of `value` across runs, per generation. A run that ended
    early (it met the target) keeps its last value, since that is where its search stopped."""
    if not trajectories:
        return pd.DataFrame(columns=["generation", "median", "q1", "q3"])
    last = max(int(t["generation"].max()) for t in trajectories)
    columns = [t.set_index("generation")[value].reindex(range(last + 1)).ffill() for t in trajectories]
    stacked = pd.concat(columns, axis=1)
    return pd.DataFrame({
        "generation": range(last + 1),
        "median": stacked.median(axis=1).to_numpy(),
        "q1": stacked.quantile(0.25, axis=1).to_numpy(),
        "q3": stacked.quantile(0.75, axis=1).to_numpy(),
    })


def fisher_exact(successes_a: int, runs_a: int, successes_b: int, runs_b: int) -> float:
    """Two-sided Fisher's exact test on a 2x2 success table: the probability, with the margins
    fixed, of a table at least as unlikely as the observed one."""
    total_successes, total = successes_a + successes_b, runs_a + runs_b

    def probability(k: int) -> float:
        return math.comb(runs_a, k) * math.comb(runs_b, total_successes - k) / math.comb(total, total_successes)

    observed = probability(successes_a)
    lowest, highest = max(0, total_successes - runs_b), min(runs_a, total_successes)
    return min(1.0, sum(p for k in range(lowest, highest + 1) if (p := probability(k)) <= observed * (1 + 1e-9)))


def _significant_direction(arm: list[float], scalar: list[float]) -> int:
    """+1 / -1 when the arm is significantly higher / lower than scalar (Mann-Whitney U), else 0."""
    if len(arm) < 2 or len(scalar) < 2:
        return 0
    _, p = mann_whitney_u(arm, scalar)
    if math.isnan(p) or p >= ALPHA:
        return 0
    return 1 if statistics.median(arm) > statistics.median(scalar) else -1


def verdict(scalar_runs: list[dict], arm_runs: list[dict]) -> str:
    """"better", "worse", "mixed" or "comparable" for an arm against the scalar arm, at ALPHA:
    first on success rate (Fisher's exact test), then on the best generalist and the final best
    fitness (Mann-Whitney U)."""
    arm_successes = sum(r["success"] for r in arm_runs)
    scalar_successes = sum(r["success"] for r in scalar_runs)
    if fisher_exact(arm_successes, len(arm_runs), scalar_successes, len(scalar_runs)) < ALPHA:
        return "better" if arm_successes / len(arm_runs) > scalar_successes / len(scalar_runs) else "worse"
    directions = [
        _significant_direction([r[key] for r in arm_runs], [r[key] for r in scalar_runs])
        for key in ("best_generalist", "final_best_fitness")
    ]
    if 1 in directions and -1 in directions:
        return "mixed"
    if 1 in directions:
        return "better"
    if -1 in directions:
        return "worse"
    return "comparable"


def _median(values) -> float:
    values = [v for v in values if v is not None]
    return float(statistics.median(values)) if values else math.nan


def arm_table(runs_by_arm: dict[str, list[dict]]) -> pd.DataFrame:
    """One row per arm (scalar first) with the medians the page reports and, for every other arm,
    its verdict against scalar."""
    scalar = runs_by_arm.get(SCALAR_ARM, [])
    ordered = sorted(runs_by_arm, key=lambda arm: (arm != SCALAR_ARM, arm))
    rows = []
    for arm in ordered:
        runs = runs_by_arm[arm]
        rows.append({
            "arm": arm,
            "runs": len(runs),
            "success rate": sum(r["success"] for r in runs) / len(runs),
            "median generations to success": _median([r["generations_to_success"] for r in runs if r["success"]]),
            "final best fitness": _median([r["final_best_fitness"] for r in runs]),
            "best's lowest partial": _median([r["final_best_min_partial"] for r in runs]),
            "best generalist": _median([r["best_generalist"] for r in runs]),
            "feasible hypervolume": _median([r["feasible_hv"] for r in runs]),
            "verdict": "" if arm == SCALAR_ARM or not scalar else verdict(scalar, runs),
        })
    return pd.DataFrame(rows)
