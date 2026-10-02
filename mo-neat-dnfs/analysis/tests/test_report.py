import pandas as pd

from viz.parsing import RecordedObjectives
from viz.report import pareto_markdown_lines


def _population(generations=(0, 1)):
    rows = []
    for generation in generations:
        rows += [
            {"generation": generation, "id": 1, "species": 0, "fitness": 0.5, "p1": 1.0, "p2": 0.0},
            {"generation": generation, "id": 2, "species": 0, "fitness": 0.6, "p1": 0.6, "p2": 0.6},
            {"generation": generation, "id": 3, "species": 1, "fitness": 0.3, "p1": 0.3, "p2": 0.3},
        ]
    return pd.DataFrame(rows)


def test_pareto_markdown_lines_summarise_the_final_generation_of_an_older_run():
    text = "\n".join(pareto_markdown_lines(_population(), None))
    assert "## Pareto (final generation 2)" in text
    assert "statistics/" in text
    assert "- Front 0: 2 of 3 (66.7%)" in text
    assert "- Fronts: 2" in text
    assert "- Hypervolume (feasible front 0): 0.3600" in text
    assert "- Specialists in front 0: 50.0%" in text
    assert "- Scalar-best individual: front 0" in text
    assert "Feasible" not in text


def test_pareto_markdown_lines_use_the_runs_recorded_settings():
    individuals = _population((0,)).assign(o1=[0.5, 0.6, 0.3], rank=[0, 0, 1], crowding=[float("inf")] * 3, violation=0.0)
    recorded = RecordedObjectives(
        {"mode": "pareto", "epsilon": 0.0, "feasibility_floor": 0.1, "objective_groups": [[0, 1]]},
        individuals,
        pd.DataFrame({"generation": [0], "size": [4], "accepted": [1]}),
    )
    text = "\n".join(pareto_markdown_lines(individuals, recorded))
    assert "objectives.jsonl" in text and "pareto selection" in text
    assert "- Objectives (m = 1): g1 (p1+p2)" in text
    assert "- Feasible: 66.7%" in text
    assert "- Archive: 4" in text


def test_pareto_markdown_lines_are_empty_without_objective_data():
    assert pareto_markdown_lines(pd.DataFrame(), None) == []
