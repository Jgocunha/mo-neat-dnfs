import math

import numpy as np
import pytest

from viz.pareto import (
    constraint_violation,
    correlation_long,
    crowding_distance,
    dominates,
    generation_metrics,
    group_objectives,
    hypervolume,
    non_dominated_sort,
    numbered_columns,
    parse_groups,
    select_objectives,
    specialist_share,
    staircase_2d,
    summarize_generation,
    validate_partition,
)

INF = math.inf


def test_dominates_without_epsilon():
    assert dominates([1, 1], [0, 0])
    assert dominates([1, 0.5], [1, 0])
    assert not dominates([1, 0], [0, 1])
    assert not dominates([0.5, 0.5], [0.5, 0.5])


def test_dominates_with_epsilon():
    assert dominates([0.55, 0.5], [0.5, 0.5])
    assert not dominates([0.55, 0.5], [0.5, 0.5], epsilon=0.1)
    assert dominates([0.7, 0.45], [0.5, 0.5], epsilon=0.1)
    assert not dominates([0.9, 0.35], [0.5, 0.5], epsilon=0.1)


def test_non_dominated_sort_on_a_two_dimensional_staircase():
    points = [
        [1.0, 0.0],  # 0: front 0
        [0.0, 1.0],  # 1: front 0
        [0.5, 0.5],  # 2: front 0
        [0.4, 0.4],  # 3: front 1
        [0.0, 0.0],  # 4: front 2
        [0.9, 0.0],  # 5: front 1
        [0.3, 0.6],  # 6: front 0
    ]
    assert non_dominated_sort(points).tolist() == [0, 0, 0, 1, 2, 1, 0]


def test_non_dominated_sort_puts_duplicates_in_one_front():
    assert non_dominated_sort([[0.5, 0.5]] * 3).tolist() == [0, 0, 0]


def test_non_dominated_sort_of_nothing_is_empty():
    assert non_dominated_sort(np.zeros((0, 2))).tolist() == []


def test_non_dominated_sort_with_epsilon_merges_near_ties():
    points = [[0.55, 0.5], [0.5, 0.5]]
    assert non_dominated_sort(points).tolist() == [0, 1]
    assert non_dominated_sort(points, epsilon=0.1).tolist() == [0, 0]


def test_non_dominated_sort_ranks_feasible_ahead_of_infeasible():
    points = [[0.1, 0.1], [1.0, 1.0], [0.2, 0.0], [0.9, 0.9]]
    violations = [0.0, 0.1, 0.0, 0.05]
    assert non_dominated_sort(points, violations=violations).tolist() == [0, 2, 0, 1]


def test_non_dominated_sort_orders_near_equal_violations_by_objectives():
    # The C++ case in tests/test_pareto.cpp: violations 0.1 and 0.1002 tie within 0.01.
    objectives = [[0.2, 0.2], [0.9, 0.9], [0.5, 0.5]]
    violations = [0.1, 0.1002, 0.3]
    assert non_dominated_sort(objectives, 0.0, violations).tolist() == [0, 1, 2]
    assert non_dominated_sort(objectives, 0.0, violations, violation_epsilon=0.01).tolist() == [1, 0, 2]


def test_violation_tie_band_never_lets_an_infeasible_point_beat_a_feasible_one():
    objectives = [[0.1, 0.1], [1.0, 1.0]]
    violations = [0.0, 0.001]
    assert non_dominated_sort(objectives, 0.0, violations, violation_epsilon=0.01).tolist() == [0, 1]


def test_non_dominated_sort_assigns_every_point_when_epsilon_dominance_is_cyclic():
    # Same cycle as the C++ test: with epsilon 0.1, a > b > c > a.
    points = [[0.25, 0.18, 0.10], [0.10, 0.25, 0.18], [0.18, 0.10, 0.25]]
    assert non_dominated_sort(points, epsilon=0.1).tolist() == [0, 0, 0]


def test_crowding_distance_boundaries_are_infinite_and_interior_is_hand_computed():
    points = [[0.0, 1.0], [0.25, 0.75], [0.5, 0.5], [1.0, 0.0]]
    ranks = np.zeros(4, dtype=int)
    distances = crowding_distance(points, ranks)
    assert distances[0] == INF and distances[3] == INF
    assert distances[1] == pytest.approx(1.0)
    assert distances[2] == pytest.approx(1.5)


def test_crowding_distance_is_computed_per_front_and_ignores_zero_range():
    points = [[0.0, 0.5], [0.5, 0.5], [1.0, 0.5], [0.1, 0.1]]
    ranks = np.array([0, 0, 0, 1])
    distances = crowding_distance(points, ranks)
    assert distances.tolist()[:3] == [INF, pytest.approx(1.0), INF]
    assert distances[3] == INF  # a front of one point is all boundary


def test_group_objectives_plain_and_weighted_means():
    partials = np.array([[0.2, 0.4, 0.6, 0.8]])
    groups = [[0, 2], [1, 3]]
    assert group_objectives(partials, groups).tolist() == [[pytest.approx(0.4), pytest.approx(0.6)]]
    weighted = group_objectives(partials, groups, weights=[0.1, 0.2, 0.3, 0.4])
    assert weighted[0, 0] == pytest.approx((0.1 * 0.2 + 0.3 * 0.6) / 0.4)
    assert weighted[0, 1] == pytest.approx((0.2 * 0.4 + 0.4 * 0.8) / 0.6)
    assert group_objectives(partials, []).tolist() == partials.tolist()


def test_constraint_violation():
    partials = np.array([[0.1, 0.5, 0.25], [0.0, 0.2, 1.0]])
    assert constraint_violation(partials, 0.0).tolist() == [0.0, 0.0]
    assert constraint_violation(partials, 0.3).tolist() == [pytest.approx(0.25), pytest.approx(0.4)]


def test_hypervolume_of_a_single_point_is_the_product_of_its_coordinates():
    for point in ([0.5], [0.5, 0.4], [0.5, 0.4, 0.3]):
        value, exact = hypervolume([point])
        assert exact
        assert value == pytest.approx(math.prod(point))


def test_hypervolume_of_a_two_dimensional_staircase():
    value, exact = hypervolume([[1.0, 0.2], [0.5, 0.6], [0.2, 1.0]])
    assert exact
    assert value == pytest.approx(0.5 * 0.2 + 0.3 * 0.6 + 0.2 * 1.0)


def test_hypervolume_in_three_dimensions_subtracts_the_overlap():
    value, exact = hypervolume([[1.0, 1.0, 0.5], [0.5, 0.5, 1.0]])
    assert exact
    assert value == pytest.approx(0.5 + 0.25 - 0.125)


def test_hypervolume_ignores_dominated_points_and_handles_empty():
    assert hypervolume([[0.5, 0.5], [0.2, 0.2]])[0] == pytest.approx(0.25)
    assert hypervolume(np.zeros((0, 3))) == (0.0, True)


def test_hypervolume_above_three_objectives_is_a_seeded_monte_carlo_estimate():
    value, exact = hypervolume([[0.5, 0.5, 0.5, 0.5]])
    assert not exact
    assert value == pytest.approx(0.0625, abs=0.005)
    assert hypervolume([[0.5, 0.5, 0.5, 0.5]])[0] == value  # fixed seed: reproducible


def test_parse_groups():
    assert parse_groups("0,2 | 1,3,4 | 5,6,7") == [[0, 2], [1, 3, 4], [5, 6, 7]]
    assert parse_groups("  ") == []
    with pytest.raises(ValueError):
        parse_groups("0,a | 1")
    with pytest.raises(ValueError):
        parse_groups("0,1 || 2")


def test_validate_partition():
    assert validate_partition([[0, 2], [1, 3]], 4) is None
    assert validate_partition([], 4) is None
    assert "used 2 times" in validate_partition([[0, 1], [1, 2, 3]], 4)
    assert "used 0 times" in validate_partition([[0, 1], [3]], 4)
    assert "out of range" in validate_partition([[0, 1], [2, 3, 4]], 4)
    assert "empty" in validate_partition([[0, 1, 2, 3], []], 4)


def test_staircase_2d_keeps_the_projected_front_sorted_by_x():
    xy = [[0.5, 0.6], [0.2, 1.0], [0.4, 0.4], [1.0, 0.2], [1.0, 0.1]]
    assert staircase_2d(xy).tolist() == [[0.2, 1.0], [0.5, 0.6], [1.0, 0.2]]


def test_specialist_share():
    partials = np.array([[0.05, 0.9], [0.5, 0.5], [0.9, 0.0]])
    assert specialist_share(partials) == pytest.approx(2 / 3)
    assert math.isnan(specialist_share(np.zeros((0, 2))))


def test_select_objectives_prefers_recorded_values_only_for_the_recorded_grouping():
    partials = np.array([[0.2, 0.4, 0.6, 0.8]])
    recorded = np.array([[0.5, 0.7]])  # weighted by the run, so not the plain mean
    same = select_objectives(partials, [[0, 2], [1, 3]], recorded, [[0, 2], [1, 3]])
    assert same.values.tolist() == recorded.tolist()
    assert same.source == "recorded"
    assert same.labels == ["g1 (p1+p3)", "g2 (p2+p4)"]

    other = select_objectives(partials, [[0, 1], [2, 3]], recorded, [[0, 2], [1, 3]])
    assert other.values.tolist() == [[pytest.approx(0.3), pytest.approx(0.7)]]
    assert other.source == "plain mean"

    raw = select_objectives(partials, [], recorded, [[0, 2], [1, 3]])
    assert raw.values.tolist() == partials.tolist()
    assert raw.labels == ["p1", "p2", "p3", "p4"]
    assert raw.source == "raw partials"


def test_summarize_generation():
    partials = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.4, 0.4]])
    fitness = np.array([0.5, 0.5, 0.5, 0.4])
    summary = summarize_generation(partials, partials, fitness, epsilon=0.0, floor=0.1)
    assert summary["front0_size"] == 1  # only (0.5, 0.5) is feasible and undominated
    assert summary["feasible_front0_size"] == 1
    assert summary["n_fronts"] == 3  # the two specialists tie on violation 0.1
    assert summary["feasible_share"] == pytest.approx(0.5)
    assert summary["specialist_share"] == pytest.approx(0.0)
    assert summary["hypervolume"] == pytest.approx(0.25)
    assert summary["hypervolume_exact"]
    assert summary["scalar_best_rank"] == 2  # argmax picks index 0, an infeasible specialist


def _population_frame():
    import pandas as pd

    return pd.DataFrame(
        {
            "generation": [0, 0, 0, 1, 1],
            "id": [1, 2, 3, 4, 5],
            "species": [0, 0, 1, 1, 1],
            "fitness": [0.5, 0.5, 0.3, 0.8, 0.2],
            "p1": [1.0, 0.0, 0.3, 0.8, 0.2],
            "p2": [0.0, 1.0, 0.0, 0.8, 0.2],
        }
    )


def test_numbered_columns_sorts_numerically():
    import pandas as pd

    df = pd.DataFrame(columns=["p10", "p2", "o1", "p1", "fitness"])
    assert numbered_columns(df, "p") == ["p1", "p2", "p10"]
    assert numbered_columns(df, "o") == ["o1"]


def test_generation_metrics_summarises_every_requested_generation():
    metrics = generation_metrics(_population_frame(), (0, 1), groups=[], recorded_groups=None, epsilon=0.0, floor=0.0)
    assert metrics["gen_display"].tolist() == [1, 2]
    assert metrics["front 0 (%)"].tolist() == [pytest.approx(200 / 3), pytest.approx(50.0)]
    assert metrics["fronts"].tolist() == [2, 2]
    assert metrics["specialists in front 0 (%)"].tolist() == [pytest.approx(100.0), pytest.approx(0.0)]
    assert metrics["hypervolume"].tolist() == [pytest.approx(0.0), pytest.approx(0.64)]
    assert "feasible (%)" not in metrics.columns  # floor off: nothing to report


def test_correlation_long_drops_objectives_without_variance():
    values = np.array([[0.1, 0.9, 0.5], [0.2, 0.8, 0.5], [0.3, 0.7, 0.5]])
    corr = correlation_long(values, ["a", "b", "c"])
    pairs = {(r, c): v for r, c, v in corr.itertuples(index=False)}
    assert pairs[("a", "b")] == pytest.approx(-1.0)
    assert pairs[("a", "a")] == pytest.approx(1.0)
    assert not any("c" in key for key in pairs)
