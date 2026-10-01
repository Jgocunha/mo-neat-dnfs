# mo-neat-dnfs

## Multi-Objective NeuroEvolution of Augmenting Dynamic Neural Field Topologies

[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?style=flat-square&logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/20)
[![CMake](https://img.shields.io/badge/CMake-3.31%2B-064F8C?style=flat-square&logo=cmake&logoColor=white)](https://cmake.org)

**mo-neat-dnfs** is the multi-objective branch of [neat-dnfs](https://github.com/Jgocunha/neat-dnfs),
split off at neat-dnfs v0.3.0. The two projects are developed separately from here on.

It adds **Pareto (multi-objective) selection**. Instead of ranking solutions by the weighted sum of their
partial fitnesses, it can rank them by dominance over an objective vector, using NSGA-II non-dominated
sorting and crowding distance inside NEAT's species loop. Scalar selection remains the default and behaves
exactly as in neat-dnfs.

**What neat-dnfs is.** The framework itself (evolving Dynamic Neural Field architectures with NEAT, the
genome and species model, the benchmark tasks and the ablation studies) is described in the
[neat-dnfs README](https://github.com/Jgocunha/neat-dnfs#readme), its
[wiki](https://github.com/Jgocunha/neat-dnfs/wiki) and its [API docs](https://jgocunha.github.io/neat-dnfs/).
This README covers only what differs here.

---

## Multi-objective selection

Work in progress. The design, the evidence behind it and the phase-by-phase implementation log are in
[`.claude/notes/MOO/PLAN.md`](.claude/notes/MOO/PLAN.md). In short:

* **Objectives are grouped partial fitnesses.** Each task partitions its partial-fitness terms into
  2–3 complete behavioural requirements, so that no trivial controller (always on, always off) can
  maximise an objective.
* **Selection keeps NEAT.** Speciation, fitness sharing, offspring allocation, champion elitism,
  crossover and mutation stay. Every place that compared two scalar fitnesses compares Pareto rank and
  crowding distance instead.
* **An optional feasibility floor** (Deb 2002's constrained domination) puts solutions with any partial
  below the floor behind every feasible one.
* **The weighted sum is still computed.** It picks the reported best solution and decides when a run
  ends, so scalar and Pareto runs stay comparable.
* **Off by default.** A run is in Pareto mode only when its config says so.

---

## Differences from neat-dnfs

| | neat-dnfs | mo-neat-dnfs |
|---|---|---|
| Nested project folder | `neat-dnfs/` | `mo-neat-dnfs/` |
| Executables | `neat-dnfs-evol`, `-inc-evol`, `-sol-eval` | `mo-neat-dnfs-evol`, `-inc-evol`, `-sol-eval` |
| Reference config | `config/neat_dnfs.json` | `config/mo_neat_dnfs.json` |
| Installed data directory | `share/neat-dnfs/` | `share/mo-neat-dnfs/` |

**Unchanged:** the C++ namespace `neat_dnfs`, include paths, the `NEAT_DNFS_*` CMake options, macros and
environment variables, and every output format. Runs recorded by neat-dnfs open in this project's
dashboard, and the neat-dnfs documentation applies here with the names above.

---

## Building

Needs CMake 3.31.6+, a C++20 compiler and [vcpkg](https://vcpkg.io) (`VCPKG_ROOT` must be set), plus
[`imgui-platform-kit`](https://github.com/Jgocunha/imgui-platform-kit) and
[`dynamic-neural-field-composer`](https://github.com/Jgocunha/dynamic-neural-field-composer).
`mo-neat-dnfs/scripts/` has setup, build and package scripts for Windows, Linux and macOS.

```bash
export VCPKG_ROOT=/path/to/vcpkg
cd mo-neat-dnfs
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

`scripts/package.sh` (`scripts\package.bat` on Windows) builds a ready-to-run archive. mo-neat-dnfs has
no published releases yet.

---

## Usage

```bash
mo-neat-dnfs-evol --list
mo-neat-dnfs-evol --task xor --runs 1
mo-neat-dnfs-evol --task and --ablation no-crossover --runs 30 --pop 500 --gens 200 --target 0.9
```

[`mo-neat-dnfs/apps/README.md`](mo-neat-dnfs/apps/README.md) documents every flag, task, ablation preset
and the config layering. Results are written to `data/` in the directory you run from.

### Analysis Tools (`analysis/`)

A Streamlit dashboard with seven pages, split into three scopes:

* **Single run** -- Fitness, Species, Topology, Mutations
* **Across runs** -- Experiment (aggregates every run in one experiment), Compare (several experiments side by side)
* **Run context** -- Provenance (build, dependency, and machine facts recorded for the selected run)

Install dependencies once with `pip install -r analysis/requirements.txt`, then run
`launch-visualizer.bat` (Windows) or `launch-visualizer.sh` (macOS/Linux) to open the app, and
select the data root and experiment/run from the sidebar. Each run keeps a persisted cache in
`<run>/.viz_cache/`, safe to delete at any time -- it is rebuilt automatically the next time that
run is opened.

---

## Citation

mo-neat-dnfs builds on neat-dnfs. If you use it in your research, please cite:

> J. G. Cunha, W. Erlhagen, R. H. Cuijpers, E. Bicho, "NEAT-DNFs: A NeuroEvolutionary Framework for Evolving Dynamic Neural Field Architectures," in *Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '26)*, Association for Computing Machinery, New York, NY, USA, 2026, pp. 966–974. https://doi.org/10.1145/3795095.3805169
