import json

import pytest

from viz.moo_compare import (
    arm_table,
    best_trajectory,
    fisher_exact,
    pair_experiments,
    run_summary,
    trajectory_band,
    verdict,
)


def _write_run(run_dir, best_partials_per_generation, final_population_partials):
    run_dir.mkdir(parents=True)
    overview = [
        {"generation": g, "bestSolution": {"fitness": sum(p) / len(p), "partialFitness": p}, "numberOfActiveSpecies": 2}
        for g, p in enumerate(best_partials_per_generation)
    ]
    (run_dir / "overview.jsonl").write_text("\n".join(json.dumps(line) for line in overview) + "\n")
    individuals = [
        {"id": i, "partialFitness": p, "objectives": [sum(p) / len(p)] * 2, "fitness": sum(p) / len(p)}
        for i, p in enumerate(final_population_partials)
    ]
    records = [{"generation": len(overview) - 1, "objectiveGroups": [], "individuals": individuals}]
    (run_dir / "objectives.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_pair_experiments_groups_each_task_with_its_suffixed_arms():
    names = ["AND", "AND Pareto", "XOR", "XOR Pareto", "Memory Trace", "Memory Trace Pareto NoFloor", "Counting"]

    pairs = pair_experiments(names)

    assert pairs == {
        "AND": {"scalar": "AND", "Pareto": "AND Pareto"},
        "XOR": {"scalar": "XOR", "Pareto": "XOR Pareto"},
        "Memory Trace": {"scalar": "Memory Trace", "Pareto NoFloor": "Memory Trace Pareto NoFloor"},
    }


def test_run_summary_reads_success_best_and_generalist(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir, [[0.5, 0.9], [0.96, 0.97], [0.97, 0.98]], [[0.2, 0.9], [0.6, 0.7], [0.96, 0.97]])

    summary = run_summary(run_dir, target=0.95)

    assert summary["success"] is True
    assert summary["generations_to_success"] == 1
    assert summary["generations"] == 3
    assert summary["final_best_min_partial"] == pytest.approx(0.97)
    assert summary["best_generalist"] == pytest.approx(0.96)


def test_run_summary_of_a_run_that_never_clears_every_partial(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir, [[0.99, 0.0], [0.99, 0.5]], [[0.99, 0.5]])

    summary = run_summary(run_dir, target=0.95)

    assert summary["success"] is False
    assert summary["generations_to_success"] is None


def test_run_summary_is_none_without_the_json_files(tmp_path):
    (tmp_path / "run").mkdir()
    assert run_summary(tmp_path / "run", target=0.95) is None


def test_best_trajectory_has_one_row_per_generation(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir, [[0.5, 0.9], [0.6, 0.8]], [[0.6, 0.8]])

    trajectory = best_trajectory(run_dir)

    assert trajectory["generation"].tolist() == [0, 1]
    assert trajectory["best_min_partial"].tolist() == pytest.approx([0.5, 0.6])


def test_trajectory_band_carries_a_finished_run_forward():
    import pandas as pd

    short = pd.DataFrame({"generation": [0, 1], "value": [0.2, 0.9]})
    long = pd.DataFrame({"generation": [0, 1, 2, 3], "value": [0.1, 0.3, 0.5, 0.7]})

    band = trajectory_band([short, long], "value")

    assert band["generation"].tolist() == [0, 1, 2, 3]
    # The short run stopped at 0.9 and keeps it.
    assert band["median"].tolist() == pytest.approx([0.15, 0.6, 0.7, 0.8])


def test_fisher_exact_two_sided():
    assert fisher_exact(5, 5, 0, 5) == pytest.approx(2 / 252)
    assert fisher_exact(3, 5, 3, 5) == pytest.approx(1.0)


def _runs(successes, generalists, fitnesses):
    return [
        {"success": s, "best_generalist": g, "final_best_fitness": f}
        for s, g, f in zip(successes, generalists, fitnesses)
    ]


def test_verdict_prefers_a_significantly_higher_success_rate():
    scalar = _runs([False] * 10, [0.1] * 10, [0.7] * 10)
    pareto = _runs([True] * 10, [0.96] * 10, [0.97] * 10)
    assert verdict(scalar, pareto) == "better"
    assert verdict(pareto, scalar) == "worse"


def test_verdict_falls_back_to_the_generalist_and_fitness():
    scalar = _runs([False] * 10, [0.10 + 0.01 * i for i in range(10)], [0.70 + 0.001 * i for i in range(10)])
    pareto = _runs([False] * 10, [0.50 + 0.01 * i for i in range(10)], [0.70 + 0.001 * i for i in range(10)])
    assert verdict(scalar, pareto) == "better"


def test_verdict_is_comparable_without_a_significant_difference():
    scalar = _runs([True, False] * 5, [0.5 + 0.01 * i for i in range(10)], [0.8 + 0.01 * i for i in range(10)])
    pareto = _runs([False, True] * 5, [0.5 + 0.01 * (9 - i) for i in range(10)], [0.8 + 0.01 * (9 - i) for i in range(10)])
    assert verdict(scalar, pareto) == "comparable"


def test_arm_table_has_a_row_per_arm_and_a_verdict_for_each_non_scalar_arm():
    scalar = [{"success": False, "generations_to_success": None, "generations": 10, "final_best_fitness": 0.7,
               "final_best_min_partial": 0.0, "best_generalist": 0.1, "feasible_hv": 0.0, "active_species": 3.0}] * 3
    pareto = [{"success": True, "generations_to_success": 4, "generations": 5, "final_best_fitness": 0.97,
               "final_best_min_partial": 0.96, "best_generalist": 0.96, "feasible_hv": 0.9, "active_species": 2.0}] * 3

    table = arm_table({"scalar": scalar, "Pareto": pareto})

    assert table["arm"].tolist() == ["scalar", "Pareto"]
    assert table["success rate"].tolist() == [0.0, 1.0]
    assert table.loc[1, "median generations to success"] == 4
    assert table.loc[0, "verdict"] == ""
    assert table.loc[1, "verdict"] in {"better", "comparable"}
