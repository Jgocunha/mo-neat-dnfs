import json

import altair as alt
import pandas as pd

from viz.plots import (
    chart_all_runs_overlay,
    chart_architecture_complexity_scatter,
    chart_best_mutation_timeline,
    chart_convergence_generations_histogram,
    chart_cross_experiment_boxplot,
    chart_first_crossing_per_component,
    chart_front0_scatter_matrix,
    chart_front_evolution,
    chart_front_union,
    chart_genome_topology_curves,
    chart_hypervolume_by_run,
    chart_innovation_growth,
    chart_kernel_usage_time,
    chart_lineage_fitness,
    chart_mutation_categories,
    chart_mutation_effectiveness,
    chart_mutations_per_generation,
    chart_objective_correlation,
    chart_parallel_coordinates,
    chart_partial_component_failure_rates,
    chart_partial_component_heatmap,
    chart_pareto_metrics,
    chart_pareto_scatter,
    chart_population_distribution,
    chart_run_duration_histogram,
    chart_seconds_per_generation_histogram,
    chart_species_champion_trajectory,
    chart_species_counts,
    chart_species_lifespans,
    chart_species_membership,
    chart_topology_frequency_heatmap,
    chart_topology_trajectory,
    chart_total_fitness,
)


def _has_field(chart_dict, field_name) -> bool:
    """Search a vega-lite spec dict (possibly nested under layer/hconcat/vconcat) for any
    encoding channel referencing `field_name`."""
    if "encoding" in chart_dict:
        for channel in chart_dict["encoding"].values():
            if isinstance(channel, dict) and channel.get("field") == field_name:
                return True
            if isinstance(channel, list):
                for c in channel:
                    if isinstance(c, dict) and c.get("field") == field_name:
                        return True
    for key in ("layer", "hconcat", "vconcat"):
        if key in chart_dict:
            for sub in chart_dict[key]:
                if _has_field(sub, field_name):
                    return True
    return False


def test_chart_total_fitness_returns_layered_chart_with_expected_fields():
    df = pd.DataFrame(
        {
            "generation": [0, 1, 2, 3],
            "avg_fitness": [0.1, 0.2, 0.3, 0.4],
            "best_fitness": [0.2, 0.3, 0.5, 0.6],
        }
    )
    chart = chart_total_fitness(df, target_fitness=0.5, success_generations=[4], success_label="met")
    assert isinstance(chart, (alt.LayerChart, alt.Chart))
    spec = chart.to_dict()
    assert _has_field(spec, "gen_display")
    assert _has_field(spec, "fitness")


def test_chart_total_fitness_without_success_generations_still_builds():
    df = pd.DataFrame({"generation": [0, 1], "avg_fitness": [0.1, 0.2], "best_fitness": [0.2, 0.3]})
    chart = chart_total_fitness(df, target_fitness=0.5)
    assert chart.to_dict() is not None


def test_chart_topology_trajectory_returns_expected_fields():
    traj_df = pd.DataFrame(
        {
            "generation": [0, 1, 2, 3],
            "hidden_fields": [0, 0, 1, 1],
            "enabled_connections": [3, 3, 3, 4],
        }
    )
    chart = chart_topology_trajectory(traj_df)
    spec = chart.to_dict()
    assert _has_field(spec, "gen_display")
    assert _has_field(spec, "count")


def test_chart_all_runs_overlay_handles_empty_input():
    chart = chart_all_runs_overlay([], target_fitness=0.9)
    assert chart.to_dict() is not None


def test_chart_all_runs_overlay_returns_expected_fields():
    all_run_metrics = [
        {"run_dir": "run1", "fitness_values": [0.1, 0.2, 0.3], "success": True},
        {"run_dir": "run2", "fitness_values": [0.05, 0.1], "success": False},
    ]
    chart = chart_all_runs_overlay(all_run_metrics, target_fitness=0.25)
    spec = chart.to_dict()
    assert _has_field(spec, "generation")
    assert _has_field(spec, "fitness")


def test_chart_species_membership_handles_empty_input():
    chart = chart_species_membership({})
    assert chart.to_dict() is not None


def test_chart_species_membership_returns_expected_fields():
    species_meta = {
        1: {"max_members": 10, "members_by_gen": {0: 5, 1: 6}},
        2: {"max_members": 3, "members_by_gen": {0: 2, 1: 2}},
    }
    chart = chart_species_membership(species_meta, top_n=1)
    spec = chart.to_dict()
    assert _has_field(spec, "generation")
    assert _has_field(spec, "members")


def test_chart_species_counts_returns_expected_fields():
    df = pd.DataFrame({"generation": [0, 1, 2], "num_species": [2, 3, 3], "num_active_species": [2, 2, 3]})
    chart = chart_species_counts(df)
    spec = chart.to_dict()
    assert _has_field(spec, "gen_display")
    assert _has_field(spec, "count")


def test_chart_innovation_growth_builds():
    df = pd.DataFrame({"generation": [0, 1], "innovation_number": [1, 3]})
    assert chart_innovation_growth(df).to_dict() is not None


def test_chart_genome_topology_curves_returns_expected_fields():
    df = pd.DataFrame(
        {
            "generation": [0, 1],
            "avg_genome_size": [3, 4],
            "avg_field_genes": [1, 1],
            "avg_conn_genes": [2, 3],
        }
    )
    chart = chart_genome_topology_curves(df)
    spec = chart.to_dict()
    assert _has_field(spec, "genes")


def test_chart_lineage_fitness_marks_structural_change():
    lineage_df = pd.DataFrame({"generation": [0, 1, 2], "fitness": [0.1, 0.2, 0.3], "hidden_fields": [0, 1, 1]})
    chart = chart_lineage_fitness(lineage_df)
    spec = chart.to_dict()
    assert _has_field(spec, "fitness")
    assert "layer" in spec


def test_chart_population_distribution_builds():
    dist_df = pd.DataFrame({"generation": [0, 0, 1, 1], "fitness": [0.1, 0.2, 0.3, 0.4]})
    chart = chart_population_distribution(dist_df, "fitness", [0, 1])
    assert chart.to_dict() is not None


def test_chart_species_lifespans_returns_expected_fields():
    species_meta = {
        1: {"first_gen": 0, "last_gen": 3, "max_members": 5, "members_by_gen": {0: 5}},
        2: {"first_gen": 1, "last_gen": 2, "max_members": 2, "members_by_gen": {1: 2}},
    }
    chart = chart_species_lifespans(species_meta)
    spec = chart.to_dict()
    assert _has_field(spec, "start")
    assert _has_field(spec, "end")


def test_chart_species_lifespans_handles_empty_input():
    assert chart_species_lifespans({}).to_dict() is not None


def test_chart_species_champion_trajectory_builds():
    traj_df = pd.DataFrame({"generation": [0, 1, 2], "fitness": [0.1, 0.2, 0.3]})
    assert chart_species_champion_trajectory(traj_df, species_id=1).to_dict() is not None


def test_chart_kernel_usage_time_returns_none_for_empty_input():
    assert chart_kernel_usage_time(pd.DataFrame(), kind="field") is None


def test_chart_kernel_usage_time_returns_expected_fields():
    df_usage = pd.DataFrame(
        {"generation": [0, 1], "field_gaussian_pct": [50, 60], "field_mexican_pct": [50, 40]}
    )
    chart = chart_kernel_usage_time(df_usage, kind="field")
    spec = chart.to_dict()
    assert _has_field(spec, "pct")


def test_chart_mutations_per_generation_builds():
    mut_events = pd.DataFrame({"generation": [0, 0, 1], "mutation_raw": ["a", "b", "a"]})
    assert chart_mutations_per_generation(mut_events).to_dict() is not None


def test_chart_mutation_effectiveness_returns_none_for_empty_input():
    assert chart_mutation_effectiveness(pd.DataFrame()) is None


def test_chart_mutation_effectiveness_returns_expected_fields():
    mut_events = pd.DataFrame(
        {"mutation_raw": ["add_field", "add_conn", "add_field"], "fitness": [0.1, 0.3, 0.2]}
    )
    chart = chart_mutation_effectiveness(mut_events)
    spec = chart.to_dict()
    assert _has_field(spec, "delta_vs_global")


def test_chart_mutation_categories_builds():
    mut_events = pd.DataFrame({"category": ["struct", "struct", "param"], "mutation_raw": ["a", "b", "c"]})
    assert chart_mutation_categories(mut_events).to_dict() is not None


def test_chart_best_mutation_timeline_builds():
    per_gen_df = pd.DataFrame({"generation": [0, 1], "mutation": ["a", "b"], "delta_vs_gen": [0.1, -0.05]})
    assert chart_best_mutation_timeline(per_gen_df).to_dict() is not None


def test_chart_run_duration_histogram_builds():
    assert chart_run_duration_histogram(pd.Series([1.0, 2.0, 1.5])).to_dict() is not None


def test_chart_seconds_per_generation_histogram_builds():
    assert chart_seconds_per_generation_histogram(pd.Series([1.0, 2.0])).to_dict() is not None


def test_chart_convergence_generations_histogram_builds():
    assert chart_convergence_generations_histogram([5, 6, 7, 10]).to_dict() is not None


def test_chart_architecture_complexity_scatter_returns_expected_fields():
    chart = chart_architecture_complexity_scatter([1, 2], [3, 4], ["run1", "run2"])
    spec = chart.to_dict()
    assert _has_field(spec, "hidden_fields")
    assert _has_field(spec, "enabled_connections")


def test_chart_cross_experiment_boxplot_builds():
    chart = chart_cross_experiment_boxplot({"exp1": [5, 6, 7], "exp2": [3, 4]})
    spec = chart.to_dict()
    assert _has_field(spec, "generations")


def test_chart_first_crossing_per_component_handles_never_reached():
    crossing = {1: 5, 2: None, 3: 8}
    chart = chart_first_crossing_per_component(crossing)
    spec = chart.to_dict()
    assert _has_field(spec, "height")


def test_chart_first_crossing_per_component_handles_empty_input():
    assert chart_first_crossing_per_component({}).to_dict() is not None


def test_chart_partial_component_failure_rates_builds():
    rates_df = pd.DataFrame({"component": [1, 2, 3], "failure_rate": [0.1, 0.5, 0.9]})
    chart = chart_partial_component_failure_rates(rates_df)
    spec = chart.to_dict()
    assert _has_field(spec, "failure_pct")


def test_chart_partial_component_failure_rates_handles_empty_input():
    assert chart_partial_component_failure_rates(pd.DataFrame()).to_dict() is not None


def test_chart_partial_component_heatmap_returns_expected_fields():
    best_df = pd.DataFrame({"generation": [0, 1, 2], "best_p1": [0.1, 0.2, 0.3], "best_p2": [0.5, 0.6, 0.4]})
    chart = chart_partial_component_heatmap(best_df, {1: 0.25, 2: 0.5})
    spec = chart.to_dict()
    assert _has_field(spec, "value")


def test_chart_partial_component_heatmap_handles_empty_input():
    assert chart_partial_component_heatmap(pd.DataFrame(), {}).to_dict() is not None


def test_chart_topology_frequency_heatmap_returns_expected_fields():
    freq_df = pd.DataFrame({"hidden_fields": [0, 1], "enabled_connections": [3, 4], "count": [5, 2]})
    chart = chart_topology_frequency_heatmap(freq_df)
    spec = chart.to_dict()
    assert _has_field(spec, "hidden_fields")


def test_chart_topology_frequency_heatmap_handles_empty_input():
    assert chart_topology_frequency_heatmap(pd.DataFrame()).to_dict() is not None


def _pareto_population():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "species": [0, 0, 1, 2],
            "fitness": [0.5, 0.6, 0.4, 0.3],
            "x": [0.2, 0.9, 0.5, 0.1],
            "y": [0.9, 0.2, 0.4, 0.1],
            "rank": [0, 0, 1, 5],
            "feasible": [True, True, False, True],
        }
    )


def test_chart_pareto_scatter_colours_by_rank_class_and_labels_the_projection():
    staircase = pd.DataFrame({"x": [0.2, 0.9], "y": [0.9, 0.2]})
    chart = chart_pareto_scatter(_pareto_population(), staircase, "p1", "p2", color_by="rank", mode="light")
    spec = chart.to_dict()
    assert _has_field(spec, "color_class")
    assert "2-D projection" in json.dumps(spec)


def test_chart_pareto_scatter_by_species_folds_the_tail_into_other():
    population = _pareto_population()
    population["species"] = [0, 1, 2, 3]
    chart = chart_pareto_scatter(population, pd.DataFrame({"x": [], "y": []}), "p1", "p2", color_by="species", mode="dark")
    spec = json.dumps(chart.to_dict())
    assert "other species" in spec


def test_chart_front0_scatter_matrix_repeats_every_objective():
    front0 = pd.DataFrame({"g1": [0.1, 0.5], "g2": [0.9, 0.4], "g3": [0.3, 0.3], "id": [1, 2]})
    spec = chart_front0_scatter_matrix(front0, ["g1", "g2", "g3"]).to_dict()
    assert spec["repeat"]["row"] == ["g1", "g2", "g3"]


def test_chart_parallel_coordinates_highlights_the_scalar_best():
    long_df = pd.DataFrame(
        {"id": [1, 1, 2, 2], "axis": ["p1", "p2", "p1", "p2"], "value": [0.1, 0.9, 0.8, 0.2], "scalar_best": [False, False, True, True]}
    )
    spec = chart_parallel_coordinates(long_df, ["p1", "p2"], mode="light").to_dict()
    assert _has_field(spec, "axis")
    assert "scalar best" in json.dumps(spec)


def test_chart_front_evolution_uses_a_sequential_generation_scale():
    df = pd.DataFrame({"gen_display": [1, 1, 5], "x": [0.2, 0.6, 0.7], "y": [0.8, 0.3, 0.6]})
    spec = chart_front_evolution(df, df, "p1", "p2", mode="dark").to_dict()
    assert _has_field(spec, "gen_display")
    assert "blues" in json.dumps(spec)


def test_chart_pareto_metrics_is_small_multiples_with_independent_scales():
    df = pd.DataFrame({"gen_display": [1, 2, 1, 2], "metric": ["hypervolume", "hypervolume", "fronts", "fronts"], "value": [0.1, 0.2, 5, 7]})
    spec = chart_pareto_metrics(df).to_dict()
    assert spec["resolve"]["scale"]["y"] == "independent"


def test_chart_objective_correlation_is_diverging_over_minus_one_to_one():
    corr = pd.DataFrame({"row": ["p1", "p2"], "column": ["p2", "p1"], "correlation": [-0.5, -0.5]})
    spec = json.dumps(chart_objective_correlation(corr, ["p1", "p2"], mode="light").to_dict())
    assert "[-1, 0, 1]" in spec.replace("-1.0", "-1").replace("1.0", "1").replace("0.0", "0")


def test_chart_hypervolume_by_run_and_front_union_build():
    hv = pd.DataFrame({"run": ["a", "b"], "hypervolume": [0.3, 0.5]})
    assert _has_field(chart_hypervolume_by_run(hv, estimate=False).to_dict(), "hypervolume")
    union = pd.DataFrame({"run": ["a", "b"], "x": [0.1, 0.9], "y": [0.9, 0.1]})
    spec = chart_front_union(union, union, "p1", "p2", mode="light").to_dict()
    assert _has_field(spec, "run")


def test_chart_cross_experiment_boxplot_accepts_its_own_labels():
    spec = json.dumps(chart_cross_experiment_boxplot({"a": [0.1, 0.2]}, title="Final hypervolume", y_title="hypervolume").to_dict())
    assert "Final hypervolume" in spec
    default = json.dumps(chart_cross_experiment_boxplot({"a": [3, 4]}).to_dict())
    assert "generations to threshold" in default
