"""Pure multi-objective helpers for the Pareto page (numpy and pandas, no Streamlit).

Mirrors include/neat/pareto.h so that ranks computed here for a run's recorded objectives
reproduce the ranks the run itself wrote to objectives.jsonl: the same constrained-domination,
the same fallback when epsilon-dominance is cyclic, and the same crowding-distance rules. Every
objective is maximised and expected in [0, 1]. Leaf module: no imports from other viz/ modules.
"""
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

# A front-0 member with some raw partial below this is a degenerate specialist: always-on,
# always-off or do-nothing controllers that are perfect on some scenarios and fail others.
SPECIALIST_THRESHOLD = 0.1
HV_MC_SAMPLES = 100_000
HV_MC_SEED = 0
_HV_MC_CHUNK = 20_000


def dominates(a, b, epsilon: float = 0.0) -> bool:
    """a dominates b (maximisation) iff a_i >= b_i - eps everywhere and a_j > b_j + eps somewhere."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return bool(np.all(a >= b - epsilon) and np.any(a > b + epsilon))


def constraint_violation(partials, floor: float) -> np.ndarray:
    """Per row, the total shortfall of the raw partials below `floor`; all zeros when floor <= 0."""
    partials = np.asarray(partials, dtype=float)
    if floor <= 0:
        return np.zeros(len(partials))
    return np.clip(floor - partials, 0.0, None).sum(axis=1)


def dominance_matrix(objectives, epsilon: float = 0.0, violations=None) -> np.ndarray:
    """D[i, j] is True when point i constrained-dominates point j (Deb 2002): feasible beats
    infeasible, the smaller violation wins between infeasible points, and epsilon-dominance
    decides between feasible ones. With no violations every point is feasible."""
    objectives = np.asarray(objectives, dtype=float)
    if len(objectives) == 0:
        return np.zeros((0, 0), dtype=bool)
    no_worse = (objectives[:, None, :] >= objectives[None, :, :] - epsilon).all(axis=2)
    better = (objectives[:, None, :] > objectives[None, :, :] + epsilon).any(axis=2)
    pareto = no_worse & better
    if violations is None:
        return pareto
    violations = np.asarray(violations, dtype=float)
    feasible = violations <= 0
    row_feasible, column_feasible = feasible[:, None], feasible[None, :]
    both_infeasible = ~row_feasible & ~column_feasible
    return (
        (row_feasible & column_feasible & pareto)
        | (row_feasible & ~column_feasible)
        | (both_infeasible & (violations[:, None] < violations[None, :]))
    )


def non_dominated_sort(objectives, epsilon: float = 0.0, violations=None) -> np.ndarray:
    """NSGA-II fast non-dominated sort. Returns one front index per point (0 = non-dominated).

    When epsilon-dominance is cyclic and no remaining point is undominated, the remaining points
    dominated by the fewest remaining points form the next front, exactly as the C++ sort does.
    """
    dominance = dominance_matrix(objectives, epsilon, violations)
    count = len(dominance)
    ranks = np.full(count, -1, dtype=int)
    remaining = np.ones(count, dtype=bool)
    dominated_by = dominance.sum(axis=0)
    rank = 0
    while remaining.any():
        front = remaining & (dominated_by == 0)
        if not front.any():
            fewest = dominated_by[remaining].min()
            front = remaining & (dominated_by == fewest)
        ranks[front] = rank
        remaining &= ~front
        dominated_by = dominated_by - dominance[front].sum(axis=0)
        rank += 1
    return ranks


def _front_crowding(objectives: np.ndarray) -> np.ndarray:
    size, objective_count = objectives.shape
    if size <= 2:
        return np.full(size, math.inf)
    distances = np.zeros(size)
    for objective in range(objective_count):
        order = np.argsort(objectives[:, objective], kind="stable")
        values = objectives[order, objective]
        value_range = values[-1] - values[0]
        if value_range <= 0:
            continue
        distances[order[0]] = math.inf
        distances[order[-1]] = math.inf
        distances[order[1:-1]] += (values[2:] - values[:-2]) / value_range
    return distances


def crowding_distance(objectives, ranks) -> np.ndarray:
    """NSGA-II crowding distance of each point within its own front. Boundary points are +inf,
    an objective with zero range contributes nothing, and a front of <= 2 points is all +inf."""
    objectives = np.asarray(objectives, dtype=float)
    ranks = np.asarray(ranks)
    distances = np.zeros(len(objectives))
    for rank in np.unique(ranks):
        members = np.flatnonzero(ranks == rank)
        distances[members] = _front_crowding(objectives[members])
    return distances


def group_objectives(partials, groups, weights=None) -> np.ndarray:
    """One column per group: the mean of its partials, weighted by `weights` renormalised within
    the group when given (a zero-weight group falls back to the plain mean). No groups: the
    partials unchanged."""
    partials = np.asarray(partials, dtype=float)
    if not groups:
        return partials.copy()
    columns = []
    for group in groups:
        members = partials[:, group]
        group_weights = None if weights is None else np.asarray(weights, dtype=float)[group]
        if group_weights is None or group_weights.sum() <= 0:
            columns.append(members.mean(axis=1))
        else:
            columns.append(members @ group_weights / group_weights.sum())
    return np.column_stack(columns)


def _non_dominated_points(points: np.ndarray) -> np.ndarray:
    return points[non_dominated_sort(points) == 0]


def _hypervolume_2d(points: np.ndarray) -> float:
    order = np.argsort(-points[:, 0], kind="stable")
    xs = points[order, 0]
    running_max_y = np.maximum.accumulate(points[order, 1])
    next_xs = np.append(xs[1:], 0.0)
    return float(np.sum((xs - next_xs) * running_max_y))


def _hypervolume_3d(points: np.ndarray) -> float:
    levels = np.unique(points[:, 2])[::-1]
    total = 0.0
    for index, level in enumerate(levels):
        next_level = levels[index + 1] if index + 1 < len(levels) else 0.0
        total += _hypervolume_2d(points[points[:, 2] >= level][:, :2]) * (level - next_level)
    return total


def _hypervolume_monte_carlo(points: np.ndarray, samples: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    covered = 0
    for start in range(0, samples, _HV_MC_CHUNK):
        draws = rng.random((min(_HV_MC_CHUNK, samples - start), points.shape[1]))
        inside = np.zeros(len(draws), dtype=bool)
        for point in points:
            inside |= (draws <= point).all(axis=1)
        covered += int(inside.sum())
    return covered / samples


def hypervolume(points, samples: int = HV_MC_SAMPLES, seed: int = HV_MC_SEED) -> tuple[float, bool]:
    """Volume of [0,1]^m dominated by `points`, reference point 0. Returns (value, exact): exact
    for m <= 3 (2-D sweep, 3-D slicing), otherwise a seeded Monte-Carlo estimate."""
    points = np.clip(np.asarray(points, dtype=float), 0.0, 1.0)
    if points.size == 0:
        return 0.0, True
    points = _non_dominated_points(points)
    objective_count = points.shape[1]
    if objective_count == 1:
        return float(points.max()), True
    if objective_count == 2:
        return _hypervolume_2d(points), True
    if objective_count == 3:
        return _hypervolume_3d(points), True
    return _hypervolume_monte_carlo(points, samples, seed), False


def parse_groups(text: str) -> list[list[int]]:
    """Parse "0,2 | 1,3,4" into [[0, 2], [1, 3, 4]] (0-based partial indices, as in the
    config's objectiveGroups). Blank text means no grouping. Raises ValueError otherwise."""
    if not text.strip():
        return []
    groups = []
    for chunk in text.split("|"):
        tokens = [token.strip() for token in chunk.split(",") if token.strip()]
        if not tokens:
            raise ValueError("every group needs at least one index")
        try:
            groups.append([int(token) for token in tokens])
        except ValueError as error:
            raise ValueError(f"not an index list: '{chunk.strip()}'") from error
    return groups


def format_groups(groups) -> str:
    """Inverse of parse_groups."""
    return " | ".join(",".join(str(index) for index in group) for group in groups)


def validate_partition(groups, partial_count: int) -> str | None:
    """None when `groups` uses every index in [0, partial_count) exactly once with no empty
    group (or is empty); otherwise a short reason, matching validateObjectiveGroups()."""
    if not groups:
        return None
    if any(len(group) == 0 for group in groups):
        return "a group is empty"
    uses = [0] * partial_count
    for group in groups:
        for index in group:
            if index < 0 or index >= partial_count:
                return f"index {index} is out of range for {partial_count} partials"
            uses[index] += 1
    for index, used in enumerate(uses):
        if used != 1:
            return f"index {index} is used {used} times"
    return None


def staircase_2d(xy) -> np.ndarray:
    """The points of `xy` that are non-dominated in these two dimensions, sorted by x."""
    xy = np.asarray(xy, dtype=float)
    if len(xy) == 0:
        return xy.reshape(0, 2)
    front = np.unique(_non_dominated_points(xy), axis=0)
    return front[np.argsort(front[:, 0], kind="stable")]


def specialist_share(partials, threshold: float = SPECIALIST_THRESHOLD) -> float:
    """Share of rows with some raw partial below `threshold`; NaN for no rows."""
    partials = np.asarray(partials, dtype=float)
    if len(partials) == 0:
        return math.nan
    return float((partials.min(axis=1) < threshold).mean())


@dataclass
class ObjectiveSpace:
    """The objective matrix the page ranks, its column labels, and where the values came from."""

    values: np.ndarray
    labels: list[str]
    source: str


def select_objectives(partials, groups, recorded_objectives, recorded_groups) -> ObjectiveSpace:
    """The objective matrix for `groups`. No groups: the raw partials. The run's own recorded
    grouping: the objectives the run wrote (weighted by its fitness weights). Any other grouping:
    plain means, since old runs do not record their weights."""
    partials = np.asarray(partials, dtype=float)
    if not groups:
        return ObjectiveSpace(partials, [f"p{i + 1}" for i in range(partials.shape[1])], "raw partials")
    labels = [f"g{k + 1} ({'+'.join(f'p{i + 1}' for i in group)})" for k, group in enumerate(groups)]
    if recorded_objectives is not None and recorded_groups and [list(g) for g in recorded_groups] == groups:
        return ObjectiveSpace(np.asarray(recorded_objectives, dtype=float), labels, "recorded")
    return ObjectiveSpace(group_objectives(partials, groups), labels, "plain mean")


def summarize_generation(objectives, partials, fitness, epsilon: float, floor: float) -> dict:
    """Headline Pareto metrics for one generation's population: ranks and crowding, front-0 size,
    number of fronts, feasible share, specialist share of front 0, hypervolume of the feasible
    front-0 points, and the rank of the scalar (weighted-sum) best individual."""
    objectives = np.clip(np.asarray(objectives, dtype=float), 0.0, 1.0)
    partials = np.asarray(partials, dtype=float)
    fitness = np.asarray(fitness, dtype=float)
    violations = constraint_violation(partials, floor)
    ranks = non_dominated_sort(objectives, epsilon, violations)
    front0 = ranks == 0
    feasible_front0 = front0 & (violations <= 0)
    hv, hv_exact = hypervolume(objectives[feasible_front0])
    return {
        "ranks": ranks,
        "crowding": crowding_distance(objectives, ranks),
        "violations": violations,
        "front0_size": int(front0.sum()),
        "front0_share": float(front0.mean()) if len(ranks) else math.nan,
        "n_fronts": int(ranks.max()) + 1 if len(ranks) else 0,
        "feasible_share": float((violations <= 0).mean()) if len(ranks) else math.nan,
        "specialist_share": specialist_share(partials[front0]),
        "feasible_front0_size": int(feasible_front0.sum()),
        "hypervolume": hv,
        "hypervolume_exact": hv_exact,
        "scalar_best_rank": int(ranks[int(np.argmax(fitness))]) if len(ranks) else -1,
    }


def numbered_columns(df, prefix: str) -> list[str]:
    """Columns named <prefix><n> (p1..pN partials, o1..oK recorded objectives), sorted by n."""
    numbered = [(int(c[len(prefix):]), c) for c in df.columns if c.startswith(prefix) and c[len(prefix):].isdigit()]
    return [c for _, c in sorted(numbered)]


def space_for_rows(rows, groups, recorded_groups) -> ObjectiveSpace:
    """select_objectives() for a slice of the population frame (columns p1..pN, and o1..oK
    when the run recorded objectives)."""
    recorded_columns = numbered_columns(rows, "o")
    recorded = rows[recorded_columns].to_numpy() if recorded_columns else None
    return select_objectives(rows[numbered_columns(rows, "p")].to_numpy(), groups, recorded, recorded_groups)


def generation_metrics(individuals, generations, groups, recorded_groups, epsilon: float, floor: float):
    """One row per generation (1-based gen_display) of the page's headline metrics, as
    percentages where they are shares. The feasible share is left out while the floor is off."""
    rows = []
    for generation in generations:
        members = individuals[individuals["generation"] == generation]
        if members.empty:
            continue
        space = space_for_rows(members, groups, recorded_groups)
        partials = members[numbered_columns(members, "p")].to_numpy()
        summary = summarize_generation(space.values, partials, members["fitness"].to_numpy(), epsilon, floor)
        row = {
            "gen_display": int(generation) + 1,
            "hypervolume": summary["hypervolume"],
            "front 0 (%)": 100 * summary["front0_share"],
            "fronts": summary["n_fronts"],
            "specialists in front 0 (%)": 100 * summary["specialist_share"],
        }
        if floor > 0:
            row["feasible (%)"] = 100 * summary["feasible_share"]
        rows.append(row)
    return pd.DataFrame(rows)


def correlation_long(values, labels: list[str]):
    """Pearson correlation between objective columns as (row, column, correlation) rows. An
    objective with no variance has no defined correlation and is left out entirely."""
    values = np.asarray(values, dtype=float)
    varying = [k for k in range(values.shape[1]) if np.ptp(values[:, k]) > 0]
    if len(varying) == 0:
        return pd.DataFrame(columns=["row", "column", "correlation"])
    matrix = np.atleast_2d(np.corrcoef(values[:, varying].T))
    rows = [
        {"row": labels[i], "column": labels[j], "correlation": float(matrix[a, b])}
        for a, i in enumerate(varying)
        for b, j in enumerate(varying)
    ]
    return pd.DataFrame(rows)
