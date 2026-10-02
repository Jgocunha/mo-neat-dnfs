import math
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import networkx as nx

from . import theme
from .theme import CATEGORICAL_CYCLE, COLOR_AVG, COLOR_BEST, COLOR_FAILURE, COLOR_STRUCTURAL_CHANGE, COLOR_SUCCESS, COLOR_TARGET
from .theme import DIVERGING_MIDPOINT, DIVERGING_NEGATIVE, DIVERGING_POSITIVE, NEUTRAL_STRONG, NEUTRAL_WEAK, SPECIES_SLOTS, TEXT_INK

theme.apply_plot_style()
theme.register_altair_theme()

# All-runs overlay can exceed Altair's default 5000-row safety limit for larger experiments
# (e.g. 40 runs x 200 generations = 8000 rows) -- disabled deliberately rather than downsampling,
# since downsampling would defeat the chart's purpose (seeing every individual run's curve).
alt.data_transformers.disable_max_rows()


def show_fig(fig):
    """Render a figure and close it, so it doesn't leak into Matplotlib's global
    figure registry. plot_topology_graph (the one remaining matplotlib plot) creates a new
    figure via plt.subplots() on every call, and nothing ever closed them, so figures
    accumulated on every rerun (e.g. every slider drag)."""
    st.pyplot(fig)
    plt.close(fig)


def plot_topology_graph(g, pos, field_nodes, kernel_nodes):
    """
    Draw the interaction graph with a clean left→right layout.

    Fields and kernels are shown with different shapes. Kept as matplotlib/networkx: Altair has
    no graph-layout mark, so this is the one chart that doesn't have a chart_* twin.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    if len(g.nodes) == 0:
        ax.text(0.5, 0.5, "No interaction topology available", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig

    if not pos:
        pos = nx.spring_layout(g, seed=42)

    # Edges
    nx.draw_networkx_edges(g, pos, ax=ax, arrows=True, arrowstyle="->")

    # Field nodes
    nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=[n for n in field_nodes if n in g.nodes],
        node_shape="o",
        ax=ax,
    )

    # Kernel nodes
    nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=[n for n in kernel_nodes if n in g.nodes],
        node_shape="s",
        ax=ax,
    )

    # Labels: just use node names (nf 1, gk 1-3, etc.)
    nx.draw_networkx_labels(g, pos, font_size=8, ax=ax)

    ax.axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Interactive Altair charts. This is the dashboard's one plotting language --
# every chart below reads colors/heights from viz/theme.py so nothing visually
# drifts from any other chart. The networkx topology graph above is the sole
# exception (no Altair graph-layout mark exists).
# ---------------------------------------------------------------------------


def chart_total_fitness(
    df: pd.DataFrame,
    target_fitness: float,
    success_generations: list[int] | None = None,
    success_label: str = "All partial targets met",
):
    """Main fitness chart. Hover tooltips show the exact generation/fitness value.

    NOTE: `df["generation"]` is 0-based (as stored in per_generation_overview.txt).
    We display generations as 1-based everywhere in the UI/plots.
    """
    plot_df = df.assign(gen_display=df["generation"] + 1)
    long_df = plot_df.melt(
        id_vars=["gen_display"],
        value_vars=["avg_fitness", "best_fitness"],
        var_name="series",
        value_name="fitness",
    )
    series_labels = {"avg_fitness": "avg. fitness", "best_fitness": "best fitness"}
    long_df["series_label"] = long_df["series"].map(series_labels)

    line = (
        alt.Chart(long_df)
        .mark_line()
        .encode(
            x=alt.X("gen_display:Q", title="generation"),
            y=alt.Y("fitness:Q", title="fitness"),
            color=alt.Color(
                "series_label:N",
                title=None,
                scale=alt.Scale(domain=["avg. fitness", "best fitness"], range=[COLOR_AVG, COLOR_BEST]),
            ),
            tooltip=[
                alt.Tooltip("gen_display:Q", title="generation"),
                alt.Tooltip("series_label:N", title="series"),
                alt.Tooltip("fitness:Q", format=".4f"),
            ],
        )
    )

    target_rule = (
        alt.Chart(pd.DataFrame({"target": [target_fitness]}))
        .mark_rule(strokeDash=[4, 4], color=COLOR_TARGET)
        .encode(y="target:Q", tooltip=[alt.Tooltip("target:Q", title="target", format=".4f")])
    )

    layers = [line, target_rule]

    if success_generations:
        # groupby/max (not set_index) so a run whose overview logged the same generation twice
        # (e.g. an interrupted/restarted run) still yields one scalar per generation, not a Series.
        best_by_gen = plot_df.groupby("gen_display")["best_fitness"].max()
        xs = [g for g in success_generations if g in best_by_gen.index]
        if xs:
            points_df = pd.DataFrame({"gen_display": xs, "fitness": [float(best_by_gen.loc[g]) for g in xs]})
            points = (
                alt.Chart(points_df)
                .mark_point(shape="diamond", size=90, filled=True, color=COLOR_SUCCESS)
                .encode(
                    x="gen_display:Q",
                    y="fitness:Q",
                    tooltip=[
                        alt.Tooltip("gen_display:Q", title="generation"),
                        alt.Tooltip("fitness:Q", title=success_label, format=".4f"),
                    ],
                )
            )
            layers.append(points)

    return alt.layer(*layers).properties(title="Total fitness", height=theme.CHART_HEIGHT).interactive()


def chart_partial_fitness_grid(partial_df: pd.DataFrame, partial_targets: dict):
    """Grid of small interactive per-component fitness charts. Draws directly into st.columns
    (not a pure chart-returning helper) and handles the empty-state itself, matching the shape
    every other render_*() caller expects."""
    if partial_df is None or partial_df.empty:
        st.info("No partial fitness statistics found (statistics/generation_X.txt missing?).")
        return

    num_partial = 0
    for col in partial_df.columns:
        if col.startswith("best_p"):
            idx = int(col.replace("best_p", ""))
            num_partial = max(num_partial, idx)

    if num_partial == 0:
        st.info("No partial fitness columns could be parsed.")
        return

    st.markdown("#### Partial fitness components")

    per_row = 4
    rows = math.ceil(num_partial / per_row)
    gen = partial_df["generation"] + 1

    comp_index = 1
    for _ in range(rows):
        cols = st.columns(per_row)
        for col in cols:
            if comp_index > num_partial:
                break

            best_col = f"best_p{comp_index}"
            avg_col = f"avg_p{comp_index}"
            if best_col not in partial_df.columns:
                comp_index += 1
                continue

            target = partial_targets.get(comp_index)

            with col:
                series = {"best": partial_df[best_col]}
                if avg_col in partial_df.columns:
                    series["avg."] = partial_df[avg_col]
                cell_df = pd.DataFrame({"gen_display": gen, **series}).melt(
                    id_vars=["gen_display"], var_name="series", value_name="fitness"
                )
                line = (
                    alt.Chart(cell_df)
                    .mark_line()
                    .encode(
                        x=alt.X("gen_display:Q", title="generation"),
                        y=alt.Y("fitness:Q", title="fitness"),
                        color=alt.Color("series:N", title=None),
                        tooltip=[
                            alt.Tooltip("gen_display:Q", title="generation"),
                            alt.Tooltip("series:N"),
                            alt.Tooltip("fitness:Q", format=".4f"),
                        ],
                    )
                )
                layers = [line]
                if target is not None:
                    target_rule = (
                        alt.Chart(pd.DataFrame({"target": [float(target)]}))
                        .mark_rule(strokeDash=[4, 4], color=COLOR_TARGET)
                        .encode(y="target:Q")
                    )
                    layers.append(target_rule)

                    reached = partial_df[partial_df[best_col] >= target]
                    if not reached.empty:
                        row = reached.iloc[0]
                        reached_df = pd.DataFrame({"gen_display": [row["generation"] + 1], "fitness": [row[best_col]]})
                        layers.append(
                            alt.Chart(reached_df)
                            .mark_point(shape="circle", size=70, filled=True, color=COLOR_SUCCESS)
                            .encode(x="gen_display:Q", y="fitness:Q", tooltip=[alt.Tooltip("gen_display:Q", title="target reached at gen")])
                        )

                chart = alt.layer(*layers).properties(title=f"partial fitness {comp_index}", height=180)
                st.altair_chart(chart, width="stretch")

            comp_index += 1


def chart_all_runs_overlay(all_run_metrics: list, target_fitness: float):
    """Every run's best-fitness-vs-generation curve overlaid, colored by success/failure, with
    a median line and IQR band across runs, and click-to-isolate via the legend. Runs of
    different lengths are aligned by generation index (position 0, 1, 2, ... from each run's own
    start) and padded with NaN past their own length, so the band narrows to only the
    still-running runs at the tail."""
    if not all_run_metrics:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line().properties(title="All-runs fitness overlay (no data)", height=theme.CHART_HEIGHT)

    max_len = max(len(m["fitness_values"]) for m in all_run_metrics)
    matrix = np.full((len(all_run_metrics), max_len), np.nan)

    run_rows = []
    for i, m in enumerate(all_run_metrics):
        vals = m["fitness_values"]
        matrix[i, : len(vals)] = vals
        is_success = bool(m.get("success"))
        run_label = "successful run" if is_success else "unsuccessful run"
        for gen0, v in enumerate(vals):
            run_rows.append({"run": m.get("run_dir", i), "generation": gen0 + 1, "fitness": v, "outcome": run_label})
    runs_df = pd.DataFrame(run_rows)

    xs_all = np.arange(1, max_len + 1)
    median = np.nanmedian(matrix, axis=0)
    q25 = np.nanpercentile(matrix, 25, axis=0)
    q75 = np.nanpercentile(matrix, 75, axis=0)
    band_df = pd.DataFrame({"generation": xs_all, "median": median, "q25": q25, "q75": q75})

    legend_selection = alt.selection_point(fields=["outcome"], bind="legend")

    run_lines = (
        alt.Chart(runs_df)
        .mark_line(opacity=0.25)
        .encode(
            x=alt.X("generation:Q", title="generation"),
            y=alt.Y("fitness:Q", title="best fitness"),
            color=alt.Color(
                "outcome:N",
                title=None,
                scale=alt.Scale(domain=["successful run", "unsuccessful run"], range=[COLOR_SUCCESS, COLOR_FAILURE]),
            ),
            detail="run:N",
            opacity=alt.condition(legend_selection, alt.value(0.25), alt.value(0.03)),
            tooltip=[
                alt.Tooltip("run:N"),
                alt.Tooltip("generation:Q"),
                alt.Tooltip("fitness:Q", format=".4f"),
                alt.Tooltip("outcome:N"),
            ],
        )
        .add_params(legend_selection)
    )

    band = alt.Chart(band_df).mark_area(opacity=0.15, color="black").encode(
        x="generation:Q", y="q25:Q", y2="q75:Q"
    )
    median_line = alt.Chart(band_df).mark_line(color="black", strokeWidth=2).encode(
        x="generation:Q", y=alt.Y("median:Q"), tooltip=[alt.Tooltip("generation:Q"), alt.Tooltip("median:Q", format=".4f")]
    )
    target_rule = (
        alt.Chart(pd.DataFrame({"target": [target_fitness]}))
        .mark_rule(strokeDash=[4, 4], color=COLOR_TARGET)
        .encode(y="target:Q")
    )

    n_success = sum(1 for m in all_run_metrics if m.get("success"))
    title = f"All-runs fitness overlay ({len(all_run_metrics)} runs, {n_success} successful)"
    return alt.layer(band, run_lines, median_line, target_rule).properties(title=title, height=theme.CHART_HEIGHT_TALL).interactive()


def chart_topology_trajectory(traj_df: pd.DataFrame):
    """Best-solution hidden-field/enabled-connection counts across generations, with a marker on
    every generation where the hidden-field count changed from the previous one (a structural
    change on the winning lineage)."""
    plot_df = traj_df.assign(gen_display=traj_df["generation"] + 1)
    long_df = plot_df.melt(
        id_vars=["gen_display"],
        value_vars=["hidden_fields", "enabled_connections"],
        var_name="series",
        value_name="count",
    )
    series_labels = {"hidden_fields": "hidden fields", "enabled_connections": "enabled connections"}
    long_df["series_label"] = long_df["series"].map(series_labels)

    line = (
        alt.Chart(long_df)
        .mark_line()
        .encode(
            x=alt.X("gen_display:Q", title="generation"),
            y=alt.Y("count:Q", title="count"),
            color=alt.Color(
                "series_label:N",
                title=None,
                scale=alt.Scale(
                    domain=["hidden fields", "enabled connections"],
                    range=[CATEGORICAL_CYCLE[1], CATEGORICAL_CYCLE[0]],
                ),
            ),
            tooltip=[
                alt.Tooltip("gen_display:Q", title="generation"),
                alt.Tooltip("series_label:N", title="series"),
                alt.Tooltip("count:Q"),
            ],
        )
    )

    layers = [line]
    changed = plot_df["hidden_fields"].diff().fillna(0) != 0
    if changed.any():
        changed_df = plot_df.loc[changed, ["gen_display", "hidden_fields"]]
        points = (
            alt.Chart(changed_df)
            .mark_point(shape="circle", size=80, filled=True, color=COLOR_STRUCTURAL_CHANGE)
            .encode(
                x="gen_display:Q",
                y="hidden_fields:Q",
                tooltip=[alt.Tooltip("gen_display:Q", title="generation"), alt.Tooltip("hidden_fields:Q", title="hidden fields")],
            )
        )
        layers.append(points)

    return alt.layer(*layers).properties(title="Best-solution topology over time", height=theme.CHART_HEIGHT).interactive()


def chart_species_membership(species_meta: dict, top_n: int = 15):
    """Stacked-area membership of the top_n species (by peak membership), with the rest of the
    population collapsed into a single 'other' band. Legend entries can be clicked to isolate a
    single species' band."""
    all_gens = sorted({g for m in species_meta.values() for g in m["members_by_gen"]})
    if not species_meta or not all_gens:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_area().properties(title="Species membership over time (no data)", height=theme.CHART_HEIGHT)

    ranked = sorted(species_meta.items(), key=lambda kv: kv[1]["max_members"], reverse=True)
    top_ids = [sid for sid, _ in ranked[:top_n]]
    other_ids = [sid for sid, _ in ranked[top_n:]]

    rows = []
    for g in all_gens:
        for sid in top_ids:
            count = species_meta[sid]["members_by_gen"].get(g, 0)
            rows.append({"generation": g + 1, "species": str(sid), "members": count})
        if other_ids:
            other_count = sum(species_meta[sid]["members_by_gen"].get(g, 0) for sid in other_ids)
            rows.append({"generation": g + 1, "species": "other", "members": other_count})

    long_df = pd.DataFrame(rows)
    legend_selection = alt.selection_point(fields=["species"], bind="legend")

    chart = (
        alt.Chart(long_df)
        .mark_area()
        .encode(
            x=alt.X("generation:Q", title="generation"),
            y=alt.Y("members:Q", title="members", stack="zero"),
            color=alt.Color("species:N", title="species", scale=alt.Scale(range=CATEGORICAL_CYCLE)),
            opacity=alt.condition(legend_selection, alt.value(0.85), alt.value(0.08)),
            tooltip=[alt.Tooltip("generation:Q"), alt.Tooltip("species:N"), alt.Tooltip("members:Q")],
        )
        .add_params(legend_selection)
        .properties(title=f"Species membership over time (top {len(top_ids)} of {len(species_meta)} species)", height=theme.CHART_HEIGHT_TALL)
    )
    return chart


def chart_topology_frequency_heatmap(freq_df: pd.DataFrame):
    """Topology-frequency map: hidden-field count x enabled-connection count, marker
    size/annotation = number of successful runs converging to that exact topology. A sized
    scatter reads better than a dense grid heatmap at the run counts this tool typically has on
    disk (tens) -- most cells would otherwise be empty."""
    if freq_df is None or freq_df.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_circle().properties(title="Topology-frequency map (no successful runs)", height=theme.CHART_HEIGHT)

    points = (
        alt.Chart(freq_df)
        .mark_circle(opacity=0.55, color=CATEGORICAL_CYCLE[0], stroke="black", strokeWidth=1)
        .encode(
            x=alt.X("enabled_connections:Q", title="enabled connections"),
            y=alt.Y("hidden_fields:Q", title="hidden fields"),
            size=alt.Size("count:Q", legend=None, scale=alt.Scale(range=[200, 2000])),
            tooltip=[alt.Tooltip("hidden_fields:Q"), alt.Tooltip("enabled_connections:Q"), alt.Tooltip("count:Q")],
        )
    )
    text = points.mark_text(color="black", fontSize=9).encode(text="count:Q")
    total = int(freq_df["count"].sum())
    return alt.layer(points, text).properties(title=f"Topology-frequency map ({total} successful runs)", height=theme.CHART_HEIGHT_TALL)


def chart_partial_component_heatmap(best_df: pd.DataFrame, partial_targets: dict):
    """Heatmap of (best_p_i - target_i) per generation, one row per component with a target
    set, diverging around 0 (purple = above target, orange = below). Encoding the value relative
    to its own target directly in color avoids needing a separate per-row threshold overlay,
    which a raw-value heatmap with a different target per row could not show cleanly."""
    comps = sorted(int(c) for c in partial_targets)
    if not comps or best_df is None or best_df.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_rect().properties(title="Partial-component heatmap (no data)", height=theme.CHART_HEIGHT)

    plot_df = best_df.assign(gen_display=best_df["generation"] + 1)
    rows = []
    for c in comps:
        col = f"best_p{c}"
        target = float(partial_targets[c])
        for _, row in plot_df.iterrows():
            value = float(row[col]) - target if col in plot_df.columns else float("nan")
            rows.append({"component": f"p{c}", "gen_display": row["gen_display"], "value": value})
    long_df = pd.DataFrame(rows).dropna(subset=["value"])

    vmax = float(long_df["value"].abs().max()) if not long_df.empty else 1.0
    vmax = vmax if vmax > 0 else 1.0

    chart = (
        alt.Chart(long_df)
        .mark_rect()
        .encode(
            x=alt.X("gen_display:O", title="generation", axis=alt.Axis(labelAngle=0, labelOverlap=True)),
            y=alt.Y("component:N", title=None, sort=[f"p{c}" for c in comps]),
            color=alt.Color("value:Q", title="best - target", scale=alt.Scale(scheme="purpleorange", domainMid=0, domain=[-vmax, vmax])),
            tooltip=[alt.Tooltip("component:N"), alt.Tooltip("gen_display:Q", title="generation"), alt.Tooltip("value:Q", format="+.4f")],
        )
        .properties(title="Partial-component value vs. target (purple = above, orange = below)", height=max(theme.CHART_HEIGHT, 40 * len(comps)))
    )
    return chart


def chart_first_crossing_per_component(crossing: dict):
    comps = sorted(crossing.keys())
    if not comps:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_bar().properties(title="First crossing per component (no data)", height=theme.CHART_HEIGHT)

    values = [crossing[c] for c in comps]
    reached = [v for v in values if v is not None]
    never_height = (max(reached) * 1.15 + 1) if reached else 1

    df = pd.DataFrame(
        {
            "component": [f"p{c}" for c in comps],
            "height": [v if v is not None else never_height for v in values],
            "reached": [v is not None for v in values],
            "label": [str(v) if v is not None else "never" for v in values],
        }
    )

    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("component:N", title=None, sort=[f"p{c}" for c in comps]),
            y=alt.Y("height:Q", title="first generation crossing target"),
            color=alt.Color(
                "reached:N",
                title=None,
                scale=alt.Scale(domain=[True, False], range=[COLOR_SUCCESS, COLOR_FAILURE]),
                legend=None,
            ),
            tooltip=[alt.Tooltip("component:N"), alt.Tooltip("label:N", title="first crossing")],
        )
    )
    labels = (
        alt.Chart(df[~df["reached"]])
        .mark_text(align="center", baseline="line-bottom", angle=270, dy=-4)
        .encode(x="component:N", y="height:Q", text="label:N")
    )
    return alt.layer(bars, labels).properties(title="First-crossing generation per partial-fitness component", height=theme.CHART_HEIGHT)


def chart_partial_component_failure_rates(rates_df: pd.DataFrame):
    if rates_df is None or rates_df.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_bar().properties(title="Partial-component failure rates (no data)", height=theme.CHART_HEIGHT)

    plot_df = rates_df.assign(
        component_label=rates_df["component"].apply(lambda c: f"p{c}"),
        failure_pct=rates_df["failure_rate"] * 100,
    )
    chart = (
        alt.Chart(plot_df)
        .mark_bar(color=CATEGORICAL_CYCLE[0])
        .encode(
            x=alt.X("component_label:N", title=None, sort=plot_df["component_label"].tolist()),
            y=alt.Y("failure_pct:Q", title="% of runs failing this component", scale=alt.Scale(domain=[0, 100])),
            tooltip=[alt.Tooltip("component_label:N", title="component"), alt.Tooltip("failure_pct:Q", format=".1f")],
        )
        .properties(title="Which components most often gate success", height=theme.CHART_HEIGHT)
    )
    return chart


def chart_species_counts(df: pd.DataFrame):
    plot_df = df.assign(gen_display=df["generation"] + 1)
    long_df = plot_df.melt(id_vars=["gen_display"], value_vars=["num_species", "num_active_species"], var_name="series", value_name="count")
    labels = {"num_species": "species", "num_active_species": "active species"}
    long_df["series_label"] = long_df["series"].map(labels)
    chart = (
        alt.Chart(long_df)
        .mark_line()
        .encode(
            x=alt.X("gen_display:Q", title="generation"),
            y=alt.Y("count:Q", title="count"),
            color=alt.Color("series_label:N", title=None, scale=alt.Scale(domain=list(labels.values()), range=CATEGORICAL_CYCLE[:2])),
            tooltip=[alt.Tooltip("gen_display:Q", title="generation"), alt.Tooltip("series_label:N", title="series"), alt.Tooltip("count:Q")],
        )
        .properties(title="Species count evolution", height=theme.CHART_HEIGHT)
    )
    return chart


def chart_innovation_growth(df: pd.DataFrame):
    plot_df = df.assign(gen_display=df["generation"] + 1)
    chart = (
        alt.Chart(plot_df)
        .mark_line(color=CATEGORICAL_CYCLE[0])
        .encode(
            x=alt.X("gen_display:Q", title="generation"),
            y=alt.Y("innovation_number:Q", title="innovation number"),
            tooltip=[alt.Tooltip("gen_display:Q", title="generation"), alt.Tooltip("innovation_number:Q")],
        )
        .properties(title="Innovation numbers growth", height=theme.CHART_HEIGHT)
    )
    return chart


def chart_genome_topology_curves(df: pd.DataFrame):
    plot_df = df.assign(gen_display=df["generation"] + 1)
    long_df = plot_df.melt(
        id_vars=["gen_display"],
        value_vars=["avg_genome_size", "avg_field_genes", "avg_conn_genes"],
        var_name="series",
        value_name="genes",
    )
    labels = {"avg_genome_size": "avg genome size", "avg_field_genes": "avg field genes", "avg_conn_genes": "avg connection genes"}
    long_df["series_label"] = long_df["series"].map(labels)
    chart = (
        alt.Chart(long_df)
        .mark_line()
        .encode(
            x=alt.X("gen_display:Q", title="generation"),
            y=alt.Y("genes:Q", title="genes"),
            color=alt.Color("series_label:N", title=None, scale=alt.Scale(domain=list(labels.values()), range=CATEGORICAL_CYCLE[:3])),
            tooltip=[alt.Tooltip("gen_display:Q", title="generation"), alt.Tooltip("series_label:N", title="series"), alt.Tooltip("genes:Q")],
        )
        .properties(title="Genome topology", height=theme.CHART_HEIGHT)
    )
    return chart


def chart_lineage_fitness(lineage_df: pd.DataFrame):
    """Fitness of a traced lineage (one ancestor per generation it spans), with markers where
    the lineage's own hidden-field count changed from its previous ancestor."""
    plot_df = lineage_df.assign(gen_display=lineage_df["generation"] + 1)
    line = (
        alt.Chart(plot_df)
        .mark_line(point=True, color=CATEGORICAL_CYCLE[0])
        .encode(
            x=alt.X("gen_display:Q", title="generation"),
            y=alt.Y("fitness:Q", title="fitness"),
            tooltip=[alt.Tooltip("gen_display:Q", title="generation"), alt.Tooltip("fitness:Q", format=".4f")],
        )
    )
    layers = [line]
    changed = plot_df["hidden_fields"].diff().fillna(0) != 0
    if changed.any():
        changed_df = plot_df.loc[changed, ["gen_display", "fitness"]]
        points = (
            alt.Chart(changed_df)
            .mark_point(shape="circle", size=80, filled=True, color=COLOR_STRUCTURAL_CHANGE)
            .encode(
                x="gen_display:Q",
                y="fitness:Q",
                tooltip=[alt.Tooltip("gen_display:Q", title="generation"), alt.Tooltip("fitness:Q", title="hidden-field count changed", format=".4f")],
            )
        )
        layers.append(points)
    return alt.layer(*layers).properties(title="Traced lineage fitness", height=theme.CHART_HEIGHT)


def chart_population_distribution(dist_df: pd.DataFrame, value_col: str, sampled_generations: list):
    """Box plot of `value_col` (fitness / genome_size / age) across the population, one box
    per generation in `sampled_generations` (a subset of dist_df["generation"] values, chosen
    by the caller for readability -- 1000 individuals x 200 generations would otherwise be
    unreadable as a single chart)."""
    order = [g + 1 for g in sampled_generations]
    rows = []
    for g in sampled_generations:
        vals = dist_df.loc[dist_df["generation"] == g, value_col]
        for v in vals:
            rows.append({"gen_display": g + 1, value_col: v})
    plot_df = pd.DataFrame(rows)

    chart = (
        alt.Chart(plot_df)
        .mark_boxplot(color=CATEGORICAL_CYCLE[0], outliers=False)
        .encode(
            x=alt.X("gen_display:O", title="generation", sort=order, axis=alt.Axis(labelAngle=-90 if len(order) > 20 else 0)),
            y=alt.Y(f"{value_col}:Q", title=value_col.replace("_", " ")),
        )
        .properties(title=f"Population {value_col.replace('_', ' ')} distribution", height=theme.CHART_HEIGHT_TALL)
    )
    return chart


def chart_species_lifespans(species_meta: dict, top_n: int = 30):
    """One horizontal bar per species spanning [first_gen, last_gen], sorted by first-seen
    generation. Capped to top_n species (earliest first) since a run can log thousands."""
    ranked = sorted(species_meta.items(), key=lambda kv: kv[1]["first_gen"])[:top_n]
    if not ranked:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_bar().properties(title="Species lifespans (no data)", height=theme.CHART_HEIGHT)

    rows = [{"species": str(sid), "start": m["first_gen"] + 1, "end": m["last_gen"] + 2} for sid, m in ranked]
    df = pd.DataFrame(rows)
    order = [r["species"] for r in rows]

    chart = (
        alt.Chart(df)
        .mark_bar(color=CATEGORICAL_CYCLE[0])
        .encode(
            y=alt.Y("species:N", title="species id", sort=order),
            x=alt.X("start:Q", title="generation"),
            x2="end:Q",
            tooltip=[alt.Tooltip("species:N"), alt.Tooltip("start:Q", title="first gen"), alt.Tooltip("end:Q", title="last gen (exclusive)")],
        )
        .properties(
            title=f"Species lifespans (first {len(ranked)} of {len(species_meta)} species, by first-seen generation)",
            height=max(theme.CHART_HEIGHT, 18 * len(ranked)),
        )
    )
    return chart


def chart_species_champion_trajectory(traj_df: pd.DataFrame, species_id: int):
    plot_df = traj_df.assign(gen_display=traj_df["generation"] + 1)
    chart = (
        alt.Chart(plot_df)
        .mark_line(point=True, color=CATEGORICAL_CYCLE[0])
        .encode(
            x=alt.X("gen_display:Q", title="generation"),
            y=alt.Y("fitness:Q", title="champion fitness"),
            tooltip=[alt.Tooltip("gen_display:Q", title="generation"), alt.Tooltip("fitness:Q", format=".4f")],
        )
        .properties(title=f"Species {species_id} champion fitness over time", height=theme.CHART_HEIGHT)
    )
    return chart


def chart_kernel_usage_time(df_usage: pd.DataFrame, kind: str):
    """
    kind: 'field' or 'inter'
    """
    if df_usage is None or df_usage.empty:
        return None

    if kind == "field":
        y1 = "field_gaussian_pct"
        y2 = "field_mexican_pct"
        title = "Field kernel usage across population"
        ylabel = "% of field kernels"
    else:
        y1 = "inter_gaussian_pct"
        y2 = "inter_mexican_pct"
        title = "Interaction kernel usage across population"
        ylabel = "% of interaction kernels"

    plot_df = df_usage.assign(gen_display=df_usage["generation"] + 1)
    long_df = plot_df.melt(id_vars=["gen_display"], value_vars=[y1, y2], var_name="series", value_name="pct")
    labels = {y1: "Gaussian", y2: "Mexican-hat"}
    long_df["series_label"] = long_df["series"].map(labels)

    chart = (
        alt.Chart(long_df)
        .mark_line()
        .encode(
            x=alt.X("gen_display:Q", title="generation"),
            y=alt.Y("pct:Q", title=ylabel, scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("series_label:N", title=None, scale=alt.Scale(domain=["Gaussian", "Mexican-hat"], range=CATEGORICAL_CYCLE[:2])),
            tooltip=[alt.Tooltip("gen_display:Q", title="generation"), alt.Tooltip("series_label:N", title="series"), alt.Tooltip("pct:Q", format=".1f")],
        )
        .properties(title=title, height=theme.CHART_HEIGHT)
    )
    return chart


def chart_mutations_per_generation(mut_events: pd.DataFrame):
    """Line chart: how many mutations occurred each generation."""
    per_gen = (
        mut_events.groupby("generation")["mutation_raw"]
        .count()
        .reset_index(name="num_mutations")
    )
    per_gen["gen_display"] = per_gen["generation"] + 1
    chart = (
        alt.Chart(per_gen)
        .mark_line(color=CATEGORICAL_CYCLE[0])
        .encode(
            x=alt.X("gen_display:Q", title="generation"),
            y=alt.Y("num_mutations:Q", title="number of mutations"),
            tooltip=[alt.Tooltip("gen_display:Q", title="generation"), alt.Tooltip("num_mutations:Q")],
        )
        .properties(title="Mutation activity per generation", height=theme.CHART_HEIGHT)
    )
    return chart


def chart_mutation_effectiveness(mut_events: pd.DataFrame, top_n: int = 10):
    """
    Horizontal bar chart of the top mutations with the highest
    positive Δ mean fitness vs the global mean.
    """
    if mut_events.empty:
        return None

    summary = (
        mut_events.groupby("mutation_raw")["fitness"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "occurrences", "mean": "mean_fitness"})
    )

    overall_mean = mut_events["fitness"].mean()
    summary["delta_vs_global"] = summary["mean_fitness"] - overall_mean

    # Top N beneficial
    top = (
        summary.sort_values("delta_vs_global", ascending=False)
        .head(top_n)
        .sort_values("delta_vs_global")  # so bars go from worst -> best
    )

    bars = (
        alt.Chart(top)
        .mark_bar(color=CATEGORICAL_CYCLE[0])
        .encode(
            x=alt.X("delta_vs_global:Q", title="Δ mean fitness vs global", axis=alt.Axis(format=".4f")),
            y=alt.Y("mutation_raw:N", title=None, sort=top["mutation_raw"].tolist(), axis=alt.Axis(labelLimit=260)),
            tooltip=[alt.Tooltip("mutation_raw:N", title="mutation"), alt.Tooltip("occurrences:Q"), alt.Tooltip("delta_vs_global:Q", format=".4f")],
        )
    )
    zero_rule = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
    return alt.layer(bars, zero_rule).properties(title=f"Most beneficial mutations (top {len(top)})", height=theme.CHART_HEIGHT_TALL)


def chart_mutation_categories(mut_events: pd.DataFrame):
    """Horizontal bar chart: how many mutation events per category. Horizontal (not vertical
    bars with rotated labels) because category names can be long enough to overlap badly on a
    shared x-axis; a category axis has as much label room as the chart is tall."""
    per_cat = (
        mut_events.groupby("category")["mutation_raw"]
        .count()
        .reset_index(name="num_events")
        .sort_values("num_events", ascending=False)
    )
    chart = (
        alt.Chart(per_cat)
        .mark_bar(color=CATEGORICAL_CYCLE[0])
        .encode(
            y=alt.Y("category:N", title=None, sort="-x", axis=alt.Axis(labelLimit=260)),
            x=alt.X("num_events:Q", title="mutation events"),
            tooltip=[alt.Tooltip("category:N"), alt.Tooltip("num_events:Q")],
        )
        .properties(title="Mutation events by category", height=max(theme.CHART_HEIGHT, 22 * len(per_cat)))
    )
    return chart


def chart_best_mutation_timeline(per_gen_df: pd.DataFrame):
    """Line chart: per-generation most beneficial mutation (delta vs gen mean)."""
    plot_df = per_gen_df.assign(gen_display=per_gen_df["generation"] + 1)
    zero_rule = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
    line = (
        alt.Chart(plot_df)
        .mark_line(point=True, color=CATEGORICAL_CYCLE[0])
        .encode(
            x=alt.X("gen_display:Q", title="generation"),
            y=alt.Y("delta_vs_gen:Q", title="Δ best mutation vs gen mean", axis=alt.Axis(format=".4f")),
            tooltip=[
                alt.Tooltip("gen_display:Q", title="generation"),
                alt.Tooltip("mutation:N", title="mutation"),
                alt.Tooltip("delta_vs_gen:Q", title="Δ vs gen mean", format=".4f"),
            ],
        )
    )
    return alt.layer(zero_rule, line).properties(title="Per-generation most beneficial mutation (approx.)", height=theme.CHART_HEIGHT)


def chart_run_duration_histogram(durations: pd.Series):
    df = pd.DataFrame({"duration_hours": durations.dropna()})
    chart = (
        alt.Chart(df)
        .mark_bar(color=CATEGORICAL_CYCLE[0])
        .encode(
            x=alt.X("duration_hours:Q", bin=alt.Bin(maxbins=10), title="duration (hours)"),
            y=alt.Y("count():Q", title="runs"),
            tooltip=[alt.Tooltip("count():Q", title="runs")],
        )
        .properties(title="Run duration", height=theme.CHART_HEIGHT)
    )
    return chart


def chart_seconds_per_generation_histogram(values: pd.Series):
    df = pd.DataFrame({"seconds_per_generation": values.dropna()})
    chart = (
        alt.Chart(df)
        .mark_bar(color=CATEGORICAL_CYCLE[0])
        .encode(
            x=alt.X("seconds_per_generation:Q", bin=alt.Bin(maxbins=10), title="sec / generation"),
            y=alt.Y("count():Q", title="runs"),
            tooltip=[alt.Tooltip("count():Q", title="runs")],
        )
        .properties(title="Time per generation", height=theme.CHART_HEIGHT)
    )
    return chart


def chart_convergence_generations_histogram(generations_to_threshold: list):
    df = pd.DataFrame({"generations": generations_to_threshold})
    max_bins = max(1, min(10, len(generations_to_threshold)))
    chart = (
        alt.Chart(df)
        .mark_bar(color=CATEGORICAL_CYCLE[0])
        .encode(
            x=alt.X("generations:Q", bin=alt.Bin(maxbins=max_bins), title="generations to success"),
            y=alt.Y("count():Q", title="runs"),
            tooltip=[alt.Tooltip("count():Q", title="runs")],
        )
        .properties(title="How many generations runs need to meet all partial targets", height=theme.CHART_HEIGHT)
    )
    return chart


def chart_architecture_complexity_scatter(hidden: list, connections: list, labels: list):
    df = pd.DataFrame({"hidden_fields": hidden, "enabled_connections": connections, "run": labels})
    points = (
        alt.Chart(df)
        .mark_circle(size=80, color=CATEGORICAL_CYCLE[0])
        .encode(
            x=alt.X("hidden_fields:Q", title="hidden fields"),
            y=alt.Y("enabled_connections:Q", title="enabled connections"),
            tooltip=[alt.Tooltip("run:N"), alt.Tooltip("hidden_fields:Q"), alt.Tooltip("enabled_connections:Q")],
        )
    )
    text = points.mark_text(align="left", dx=6, dy=-4, fontSize=9).encode(text="run:N")
    return alt.layer(points, text).properties(title="Architecture complexity (successful runs)", height=theme.CHART_HEIGHT_TALL)


def chart_cross_experiment_boxplot(
    plottable: dict,
    title: str = "Convergence speed across experiments",
    y_title: str = "generations to threshold",
):
    """plottable: {experiment_name: [value, ...]} -- generations to threshold by default; the
    Pareto comparison passes final hypervolumes with its own title."""
    rows = [{"experiment": name, "generations": v} for name, vals in plottable.items() for v in vals]
    df = pd.DataFrame(rows)
    order = list(plottable.keys())
    chart = (
        alt.Chart(df)
        .mark_boxplot(color=CATEGORICAL_CYCLE[0], outliers=False)
        .encode(
            x=alt.X("experiment:N", title=None, sort=order),
            y=alt.Y("generations:Q", title=y_title),
        )
        .properties(title=title, height=theme.CHART_HEIGHT_TALL)
    )
    return chart


# ---------------------------------------------------------------------------------------------
# Pareto page. Every chart takes `mode` ("light"/"dark", from theme.theme_type()) because its
# neutrals are per mode; the charts themselves stay pure so tests can build them.
# ---------------------------------------------------------------------------------------------

_RANK_CLASSES = ["front 0", "fronts 1-3", "front 4+"]
_OTHER_SPECIES = "other species"
_PROJECTION_LABEL = "front 0, 2-D projection"
# Objectives are maximised and bounded by 1, so 1 stays the axis ceiling; the floor follows the
# data, since late-run populations crowd into the top few percent of the range.
_OBJECTIVE_SCALE = alt.Scale(zero=False, domainMax=1)


def _rank_class(rank: int) -> str:
    if rank == 0:
        return _RANK_CLASSES[0]
    return _RANK_CLASSES[1] if rank <= 3 else _RANK_CLASSES[2]


def _species_classes(species: pd.Series) -> tuple[pd.Series, list[str]]:
    """The largest len(SPECIES_SLOTS) species keep their own label; the rest fold into one."""
    largest = species.value_counts().index[: len(SPECIES_SLOTS)].tolist()
    labels = [f"species {s}" for s in sorted(largest)]
    classes = species.map(lambda s: f"species {s}" if s in largest else _OTHER_SPECIES)
    return classes, labels


def _population_tooltip(x_title: str, y_title: str) -> list:
    return [
        alt.Tooltip("id:Q", title="solution"),
        alt.Tooltip("species:Q"),
        alt.Tooltip("fitness:Q", format=".4f"),
        alt.Tooltip("rank:Q", title="front"),
        alt.Tooltip("x:Q", title=x_title, format=".4f"),
        alt.Tooltip("y:Q", title=y_title, format=".4f"),
    ]


def chart_pareto_scatter(population: pd.DataFrame, staircase: pd.DataFrame, x_title: str, y_title: str,
                         color_by: str, mode: str):
    """The whole population on two objectives. Colored by m-D front (front 0 in the accent,
    later fronts in receding neutrals) or by species (the largest three, the rest neutral).
    Infeasible individuals are hollow. The dashed staircase is front 0 re-computed in these two
    dimensions only -- labelled as such, since a projection of an m-D front is not a front."""
    df = population.copy()
    if color_by == "species":
        df["color_class"], domain = _species_classes(df["species"])
        domain = domain + [_OTHER_SPECIES]
        color_range = SPECIES_SLOTS[: len(domain) - 1] + [NEUTRAL_WEAK[mode]]
        legend_title = "species"
    else:
        df["color_class"] = df["rank"].map(_rank_class)
        domain = _RANK_CLASSES
        color_range = [COLOR_BEST, NEUTRAL_STRONG[mode], NEUTRAL_WEAK[mode]]
        legend_title = "m-D front"
    df["feasibility"] = df["feasible"].map({True: "feasible", False: "infeasible (hollow)"})

    color = alt.Color("color_class:N", title=legend_title, scale=alt.Scale(domain=domain, range=color_range))
    points = (
        alt.Chart(df)
        .mark_point(size=64, strokeWidth=1.5)
        .encode(
            x=alt.X("x:Q", title=x_title, scale=_OBJECTIVE_SCALE),
            y=alt.Y("y:Q", title=y_title, scale=_OBJECTIVE_SCALE),
            color=color,
            fill=alt.condition("datum.feasible", color, alt.value("transparent")),
            order=alt.Order("rank:Q", sort="descending"),
            tooltip=_population_tooltip(x_title, y_title) + [alt.Tooltip("feasibility:N")],
        )
    )
    step = pd.DataFrame(staircase).assign(series=_PROJECTION_LABEL)
    line = (
        alt.Chart(step)
        .mark_line(interpolate="step-before", strokeDash=[5, 3], strokeWidth=2, color=COLOR_BEST)
        .encode(
            x="x:Q",
            y="y:Q",
            opacity=alt.Opacity("series:N", title=None, legend=alt.Legend(symbolDash=[5, 3], symbolStrokeColor=COLOR_BEST, labelLimit=240)),
        )
    )
    return alt.layer(points, line).properties(title="Population on two objectives", height=theme.CHART_HEIGHT_TALL).interactive()


def chart_front0_scatter_matrix(front0: pd.DataFrame, labels: list[str]):
    """Every pair of objectives for the front-0 members (one series, one color)."""
    return (
        alt.Chart(front0)
        .mark_circle(size=40, color=COLOR_BEST, opacity=0.7)
        .encode(
            x=alt.X(alt.repeat("column"), type="quantitative", scale=_OBJECTIVE_SCALE),
            y=alt.Y(alt.repeat("row"), type="quantitative", scale=_OBJECTIVE_SCALE),
            tooltip=[alt.Tooltip("id:Q", title="solution")],
        )
        .properties(width=110, height=110)
        .repeat(row=labels, column=labels)
    )


def chart_parallel_coordinates(long_df: pd.DataFrame, axes: list[str], mode: str):
    """Front 0 on every raw partial, one line per individual, in a receding neutral; the scalar
    (weighted-sum) best individual in the accent. Lines that touch 0 on an axis are the
    degenerate specialists."""
    df = long_df.assign(series=long_df["scalar_best"].map({True: "scalar best", False: "front 0 member"}))
    base = alt.Chart(df).encode(
        x=alt.X("axis:N", title=None, sort=axes, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("value:Q", title="partial fitness", scale=alt.Scale(domain=[0, 1])),
        detail="id:N",
        tooltip=[alt.Tooltip("id:Q", title="solution"), alt.Tooltip("axis:N"), alt.Tooltip("value:Q", format=".4f")],
    )
    members = base.transform_filter("!datum.scalar_best").mark_line(strokeWidth=1, opacity=0.35, color=NEUTRAL_STRONG[mode])
    best = base.transform_filter("datum.scalar_best").mark_line(strokeWidth=2.5, color=COLOR_BEST, point=True)
    label = (
        alt.Chart(df[df["scalar_best"] & (df["axis"] == axes[-1])])
        .mark_text(align="left", dx=8, color=COLOR_BEST)
        .encode(x=alt.X("axis:N", sort=axes), y="value:Q", text="series:N")
    )
    return alt.layer(members, best, label).properties(title="Front 0 on every raw partial", height=theme.CHART_HEIGHT_TALL)


def _generation_scale(mode: str) -> alt.Scale:
    # One hue, light->dark, with the latest generation the most prominent: darkest on a light
    # surface, lightest on a dark one. The ramp starts at 30% so early generations stay visible.
    return alt.Scale(scheme=alt.SchemeParams(name="blues", extent=[0.3, 1.0]), reverse=mode == "dark")


def chart_front_evolution(points: pd.DataFrame, steps: pd.DataFrame, x_title: str, y_title: str, mode: str):
    """Front 0 of sampled generations on two objectives, colored by generation."""
    color = alt.Color("gen_display:Q", title="generation", scale=_generation_scale(mode))
    dots = alt.Chart(points).mark_circle(size=64).encode(
        x=alt.X("x:Q", title=x_title, scale=_OBJECTIVE_SCALE),
        y=alt.Y("y:Q", title=y_title, scale=_OBJECTIVE_SCALE),
        color=color,
        tooltip=[alt.Tooltip("gen_display:Q", title="generation"), alt.Tooltip("x:Q", title=x_title, format=".4f"), alt.Tooltip("y:Q", title=y_title, format=".4f")],
    )
    lines = alt.Chart(steps).mark_line(interpolate="step-before", strokeWidth=2).encode(
        x="x:Q", y="y:Q", color=color, detail="gen_display:Q"
    )
    return alt.layer(lines, dots).properties(title="Front 0 over generations (2-D projection)", height=theme.CHART_HEIGHT_TALL).interactive()


def chart_pareto_metrics(metrics_long: pd.DataFrame):
    """Small multiples, one per metric, each on its own y-scale -- never a shared or dual axis."""
    return (
        alt.Chart(metrics_long)
        .mark_line(strokeWidth=2, color=COLOR_BEST, point=alt.OverlayMarkDef(size=30, color=COLOR_BEST))
        .encode(
            x=alt.X("gen_display:Q", title="generation"),
            y=alt.Y("value:Q", title=None),
            tooltip=[alt.Tooltip("metric:N"), alt.Tooltip("gen_display:Q", title="generation"), alt.Tooltip("value:Q", format=".4f")],
        )
        .properties(width=210, height=150)
        .facet(facet=alt.Facet("metric:N", title=None), columns=3)
        .resolve_scale(y="independent")
    )


def chart_objective_correlation(corr_long: pd.DataFrame, labels: list[str], mode: str):
    """Pearson correlation between objectives across the population, diverging around 0:
    negative (blue) cells are genuine trade-offs, strongly positive (vermillion) cells are
    redundant objectives. The diagonal (always +1) is left blank so it cannot dominate, and so
    are objectives with no variance."""
    # Lab interpolation: the default path from a dark-gray midpoint to vermillion passes through
    # purple, which reads as a third hue.
    scale = alt.Scale(domain=[-1, 0, 1], range=[DIVERGING_NEGATIVE, DIVERGING_MIDPOINT[mode], DIVERGING_POSITIVE], interpolate="lab")
    corr_long = corr_long[corr_long["row"] != corr_long["column"]]
    cells = alt.Chart(corr_long).mark_rect(stroke=DIVERGING_MIDPOINT[mode], strokeWidth=1).encode(
        x=alt.X("column:N", title=None, sort=labels, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("row:N", title=None, sort=labels),
        color=alt.Color("correlation:Q", title="correlation", scale=scale),
        tooltip=[alt.Tooltip("row:N"), alt.Tooltip("column:N"), alt.Tooltip("correlation:Q", format="+.2f")],
    )
    text = alt.Chart(corr_long).mark_text(fontSize=11, color=TEXT_INK[mode]).encode(
        x=alt.X("column:N", sort=labels), y=alt.Y("row:N", sort=labels), text=alt.Text("correlation:Q", format="+.2f")
    )
    # Step sizing: a fixed width/height on a discrete-axis layer collapses under
    # st.altair_chart(width="content"); a per-cell step keeps cells readable at any m.
    return alt.layer(cells, text).properties(title="Objective conflict", width=alt.Step(72), height=alt.Step(44))


def chart_hypervolume_by_run(hv_df: pd.DataFrame, estimate: bool):
    """Final-generation hypervolume of every run in an experiment (one dot per run)."""
    title = "Final hypervolume per run" + (" (Monte-Carlo estimate)" if estimate else "")
    box = alt.Chart(hv_df).mark_boxplot(color=COLOR_BEST, opacity=0.35, outliers=False, size=40).encode(y=alt.Y("hypervolume:Q"))
    dots = alt.Chart(hv_df).mark_circle(size=64, color=COLOR_BEST).encode(
        y=alt.Y("hypervolume:Q", title="hypervolume"),
        tooltip=[alt.Tooltip("run:N"), alt.Tooltip("hypervolume:Q", format=".4f")],
    )
    return alt.layer(box, dots).properties(title=title, height=theme.CHART_HEIGHT, width=160)


def chart_front_union(points: pd.DataFrame, union_staircase: pd.DataFrame, x_title: str, y_title: str, mode: str):
    """Every run's final front 0 on two objectives (one neutral series), with the 2-D front of
    their union in the accent."""
    dots = alt.Chart(points).mark_circle(size=40, color=NEUTRAL_STRONG[mode], opacity=0.7).encode(
        x=alt.X("x:Q", title=x_title, scale=_OBJECTIVE_SCALE),
        y=alt.Y("y:Q", title=y_title, scale=_OBJECTIVE_SCALE),
        tooltip=[alt.Tooltip("run:N"), alt.Tooltip("x:Q", title=x_title, format=".4f"), alt.Tooltip("y:Q", title=y_title, format=".4f")],
    )
    step = pd.DataFrame(union_staircase).assign(series="2-D front of the union")
    line = alt.Chart(step).mark_line(interpolate="step-before", strokeWidth=2, color=COLOR_BEST).encode(
        x="x:Q", y="y:Q", opacity=alt.Opacity("series:N", title=None)
    )
    return alt.layer(dots, line).properties(title="Final fronts across runs", height=theme.CHART_HEIGHT_TALL).interactive()


# --- Scalar vs Pareto page ------------------------------------------------------------------

def _arm_color(arms: list[str]) -> alt.Color:
    """Color keyed by arm in the page's fixed order (scalar first), so an arm keeps its color
    whatever else is selected."""
    return alt.Color("arm:N", title="arm", scale=alt.Scale(domain=arms, range=theme.ARM_SLOTS[:len(arms)]),
                     legend=alt.Legend(orient="top"))


def chart_arm_success(success: pd.DataFrame, arms: list[str], mode: str):
    """Success rate (every partial above the target) per task and arm, as bars with the k/n count."""
    base = alt.Chart(success).encode(
        y=alt.Y("task:N", title=None, axis=alt.Axis(labelLimit=240)),
        yOffset=alt.YOffset("arm:N", sort=arms),
        # The domain runs past 100% so a full bar still has room for its "k/n" label.
        x=alt.X("success_rate:Q", title="runs that met the target on every partial",
                axis=alt.Axis(format="%", values=[0, 0.25, 0.5, 0.75, 1.0]), scale=alt.Scale(domain=[0, 1.12])),
        tooltip=[alt.Tooltip("task:N"), alt.Tooltip("arm:N"), alt.Tooltip("label:N", title="successful runs"),
                 alt.Tooltip("success_rate:Q", title="success rate", format=".0%")],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height={"band": 0.8}).encode(color=_arm_color(arms))
    labels = base.mark_text(align="left", dx=4, color=TEXT_INK[mode]).encode(text="label:N")
    return alt.layer(bars, labels).properties(title="Success rate", height=alt.Step(14))


def chart_arm_trajectories(bands: pd.DataFrame, arms: list[str], title: str, y_title: str, mode: str,
                           target: float | None = None):
    """Median (line) and interquartile range (band) across runs per generation, one panel per task."""
    color = _arm_color(arms)
    band = alt.Chart().mark_area(opacity=0.18).encode(
        x=alt.X("generation:Q", title="generation"), y=alt.Y("q1:Q", title=y_title), y2="q3:Q", color=color)
    line = alt.Chart().mark_line(strokeWidth=2).encode(
        x="generation:Q", y="median:Q", color=color,
        tooltip=[alt.Tooltip("arm:N"), alt.Tooltip("generation:Q"), alt.Tooltip("median:Q", format=".3f"),
                 alt.Tooltip("q1:Q", title="25th percentile", format=".3f"), alt.Tooltip("q3:Q", title="75th percentile", format=".3f")],
    )
    layers = [band, line]
    if target is not None:
        layers.append(alt.Chart().mark_rule(strokeDash=[4, 4], color=TEXT_INK[mode], opacity=0.6).encode(
            y=alt.datum(target)))
    return (
        alt.layer(*layers, data=bands)
        .properties(width=260, height=180)
        .facet(facet=alt.Facet("task:N", title=None), columns=3)
        .resolve_scale(x="independent")
        .properties(title=title)
    )


def chart_arm_final_values(finals: pd.DataFrame, arms: list[str], value: str, title: str):
    """Every run's final value as a dot, with the arm's median as a tick, one panel per task."""
    color = _arm_color(arms)
    dots = alt.Chart().mark_circle(size=64, opacity=0.75).encode(
        x=alt.X("arm:N", sort=arms, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{value}:Q", title=title, scale=alt.Scale(domain=[0, 1])),
        xOffset=alt.XOffset("jitter:Q"),
        color=color,
        tooltip=[alt.Tooltip("arm:N"), alt.Tooltip("run:N"), alt.Tooltip(f"{value}:Q", format=".3f")],
    ).transform_calculate(jitter="random()")
    medians = alt.Chart().mark_tick(thickness=3, size=36).encode(
        x=alt.X("arm:N", sort=arms), y=f"median({value}):Q", color=color)
    return (
        alt.layer(dots, medians, data=finals)
        .properties(width=200, height=200)
        .facet(facet=alt.Facet("task:N", title=None), columns=3)
        .properties(title=title)
    )
