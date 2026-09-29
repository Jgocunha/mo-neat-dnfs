# Multi-objective (Pareto) selection for neat-dnfs — implementation plan

> **Moved (2026-09-29).** This plan now lives in, and is implemented in, **mo-neat-dnfs**
> (`C:\dev-files\mo-neat-dnfs`), a local-only repository split off from neat-dnfs v0.3.0.
> Read `neat-dnfs/...` paths below as `mo-neat-dnfs/...`, `config/neat_dnfs.json` as
> `config/mo_neat_dnfs.json` and `neat-dnfs-*` binaries/targets as `mo-neat-dnfs-*`; the C++
> namespace is still `neat_dnfs`. "One PR per phase" now means one *local branch* per phase
> (no remote). See "Repository split" in the implementation notes at the bottom.

Status: investigation complete, **decisions confirmed by the user on 2026-09-29 (section 6)**,
implementation not started. Branch used for probing: `feat/moo-investigation` (only change on it:
`.claude/notes/MOO/` added to `.gitignore`).
Sandbox with the evidence: `.claude/notes/MOO/sandbox/`:
- `pareto_probe.py`: front sizes and correlations of raw partials, with optional ε
- `grouping_check.py RUN "0,2|1,3,4|5,6,7"`: raw vs grouped fronts, and the share of specialists
- `constraint_check.py RUN GROUPS FLOOR`: the constrained-domination probe (§1.5)
- `proto_figure.py` → `proto_pareto_views.png`: the prototype visualizer views
- `evco.txt`, `motw.txt`: extracted paper text

Read sections 0 and 6 before writing any code. Section 6 records decisions the user has already
made: **do not re-ask them, and do not deviate from them.** Anything *new* that would break
backwards compatibility still needs the user's approval first (per CLAUDE.md).

---

## 0. TL;DR

**Goal.** Stop selecting on the weighted sum `fitness = Σ wᵢ·partialᵢ`, and select on Pareto
dominance over an objective vector instead. The weights stop deciding selection and are only used
afterwards, to pick one solution from the front for reporting. Also add a **Pareto page** to the
Streamlit visualizer.

**Recommended approach: "NSGA-II ranking inside NEAT's species loop".**
- Keep everything NEAT-specific: speciation, fitness sharing, species offspring allocation,
  pruning, champion elitism, crossover and mutation.
- After `evaluate()`, run NSGA-II's fast non-dominated sort and crowding distance over the whole
  population. Each solution gets `(paretoRank, crowdingDistance)`.
- Every place that currently *compares* two scalar fitnesses uses a single comparator instead:
  lower rank wins, and on a tie the larger crowding distance wins.
- The one place that needs a *magnitude* (between-species offspring allocation) gets a
  weight-free, rank-derived `selectionFitness`.
- Everything is off by default. `SelectionConstants.mode = "scalar"` behaves exactly as today,
  and `"pareto"` is turned on by a config preset.

**Objectives = grouped partial fitnesses, not raw partials.** Each task declares
`objectiveGroups`, which partition its partial-fitness terms into 2–3 *complete behavioural
requirements*. Section 1 shows the data behind this: raw partials break down at 8 objectives, and
they let degenerate "always-on / always-off" controllers onto the front.

**Plus an optional feasibility floor (Deb 2002's constrained-domination).** A solution with any
raw partial below `feasibilityFloor` is infeasible. Feasible solutions always beat infeasible
ones, and among infeasible solutions the smaller total shortfall wins. On AND this removes every
specialist from front 0 from generation 5 onwards (§1.5). The config default is off, and the
Pareto preset turns it on at 0.1.

**Build order (each step is PR-sized):**
1. `pareto` core module — pure functions with tests.
2. Config block and objective grouping.
3. `objectives.jsonl` output.
4. **Visualizer Pareto page.** It also works on every existing run, retroactively.
5. Pareto selection mode.
6. Preset, per-task groups and documentation.
7. Scalar-vs-Pareto evaluation experiment.

The visualizer comes before selection on purpose. It is the measuring instrument, and it delivers
value on existing data even if Pareto selection turns out not to help.

---

## 1. Evidence gathered (why this design)

All numbers come from real runs under `neat-dnfs/data/`, produced by
`sandbox/pareto_probe.py` and `sandbox/group_probe.py`. Every run used today's **scalar**
selection. The question was: if we had ranked these populations by dominance, how much selection
pressure would there be?

### 1.1 Four objectives: plain NSGA-II is fine

| Task (pop 500) | front-0 size, % of pop | # fronts | pairwise objective correlation |
|---|---|---|---|
| Selection Instability (45 gens) | 2–12 % | 13–19 | mostly weak; one pair −0.17 |
| Memory Instability (11 gens) | 1–21 % | 15–58 | f2↔f4 **+0.93** (redundant) |
| Detection Instability (7 gens) | 5–21 % | 11–21 | one pair −0.28 |

With 4 objectives, dominance still separates the population into many fronts, so selection
pressure survives. The best weighted-sum individual was always on front 0 (`bestScalarRank=0`).

### 1.2 Eight objectives (AND): raw partials break down

AND run, pop 300, 25 gens (`data/AND/2026-09-28 15h14m03s`):

| Objective space | front-0 size | # fronts |
|---|---|---|
| 8 raw partials, ε = 0 | **23–69 %** | **4–8** |
| 8 raw partials, ε = 0.02 | 2–44 % (growing late) | 9–37 |
| **3 semantic groups**, ε = 0 | **2–15 %** | **21–42** |

This is the many-objective failure described by Deb & Jain 2014 (NSGA-III), which the lit review
cites. Once most of the population is non-dominated, rank no longer discriminates. ε-dominance
partly helps. Grouping helps a lot.

### 1.3 Raw partials put degenerate specialists on the front

See the middle panel of `sandbox/proto_pareto_views.png`. With the 8 raw AND partials, front 0 is
full of individuals that score **0** on `f1_2`, `f2_2` or `f3`: output always on, or always off.
Each is non-dominated because it is perfect on the scenarios where "always on" (or "always off")
happens to be right.

The strong negative correlations are exactly the AND trade-off:
- `f3` (output fires when both inputs are on) against `f1_2`/`f2_2` (output silent on a single
  input): −0.43 to −0.58.
- `f3` against `f4_3` (output returns to rest): −0.50.

**Design rule for objectives.** Each objective must be a *complete behavioural requirement across
all its scenarios*, so that no trivial controller (always-on, always-off, do-nothing) can maximise
it. The scenario terms that conflict *for trivial controllers* go into the **same** group. With
AND grouped as input {f1_1,f2_1} / output-logic {f1_2,f2_2,f3} / rest {f4_*}, the trivial
specialists sit at output = 0.667, where the 0.97+ solutions dominate them (left panel of the
figure).

Grouping **reduces** specialists but does not remove them. On AND, the share of front-0 members
with some raw partial below 0.1 falls from 70–96 % (raw) to 17–58 % (grouped) from generation 5
on. The feasibility floor (§1.5) finishes the job.

### 1.4 Every proposed grouping checked against a real run

Short scalar-selection runs (pop 300, 20 generations; best fitness reached ≈ 0.70–0.75, so none
solved) were made on 2026-09-29 for xor, memory-trace, dmts and ior, and checked with
`grouping_check.py`. The four-objective tasks use the §1.1 runs. The "conflict" column lists the
pooled raw-partial correlations that bear on the grouping.

| Task | Grouping | Grouped front 0 (gens 5–20) | Conflict evidence | Verdict |
|---|---|---|---|---|
| and | `0,2 \| 1,3,4 \| 5,6,7` | 2–15 %, 24–38 fronts | 4↔1 −0.58, 4↔3 −0.43, 4↔7 −0.50: the AND trade-off, inside the output group | **keep** |
| xor | `0,1,2 \| 3` | 0.3–1.3 %, 54–283 fronts | 2↔0 −0.58, 2↔1 −0.43: "silent on both" vs "fire on one", inside the logic group | **keep**; the run never left the all-specialist stage (100 % in both spaces), so XOR needs the floor |
| memory-trace | `0,2 \| 1,3,4 \| 5,6,7` | 2–5 %, 33–43 fronts | 5↔7 −0.92, 6↔7 −0.96 (output bump vs output silent: the strongest conflict seen, inside group 3); 0↔2 +0.69 | **keep** (upgraded from low confidence) |
| dmts | `0,2,4 \| 1,3,5` | 1–2 %, 52–125 fronts | nf1 terms all positive (+0.44…+0.61); nf2: 3↔1 −0.39, 3↔5 −0.45 inside the group; 1↔5 +0.74 | **keep** |
| ior | `0,2 \| 1,3,4` | 1–2 %, 62–175 fronts | 1↔4 **+0.99** (redundant, same group); 3↔2 −0.39 *across* groups (a genuine front trade-off) | **keep**; front-0 specialists 0–33 % |
| memory-instability | `0,2 \| 1,3` | (4-obj data, §1.1) | 1↔3 +0.93 (same group) | **keep** |
| detection- / selection-instability | `0,1 \| 2,3` | (4-obj data, §1.1) | weak correlations; groups follow "detect/select vs return to rest" | **keep** |

With two grouped objectives, front 0 is small (1–2 %) and nearly a total order. Selection pressure
is then close to scalar selection, and diversity comes from speciation and crowding, as in NEAT
today. This is expected. Phase 7 checks whether it costs diversity.

### 1.5 Feasibility floor (constrained-domination, Deb 2002 §V)

`constraint_check.py` on AND, grouped objectives:

| gen | feasible share, floor 0.1 | front-0 specialists, floor 0.1 | feasible share, floor 0.3 |
|---|---|---|---|
| 1 | 0 % (ranked by shortfall) | 100 % | 0 % |
| 5 | 2.3 % | **0 %** | 1.3 % |
| 10 | 12.0 % | **0 %** | 9.7 % |
| 20 | 20.0 % | **0 %** | 13.7 % |
| 25 | 21.0 % | **0 %** | 19.7 % |

When nobody is feasible (early generations, and all of the short XOR run), the ordering by total
shortfall `Σ max(0, floor − partialᵢ)` still gives a gradient toward generalists. It is itself a
weight-free scalarisation, and it applies only below the floor. 0.1 is the preset value: it is
the same threshold the probes use to define a specialist, and it is lenient enough to leave
10–20 % of a mid-run population feasible.

### 1.6 Literature (papers are in this folder; the lit review is `lit-rev.tex` §4.4)

| Method | What it does | Fit for neat-dnfs |
|---|---|---|
| **NEAT-PS** (cited in Abramovich 2016) | SPEA2 Pareto-strength squashed into a scalar, then plain NEAT | Least work, but criticised: elitism no longer guarantees monotone progress in *all* objectives. Rejected. |
| **MM-NEAT** (Schrum & Miikkulainen, `EVCO_a_00181.pdf`) | NSGA-II (μ+λ) replaces NEAT selection; **no speciation** | Throws away speciation, which our own ablation shows matters. Rejected as the default. |
| **NEAT-MODS** (Abramovich & Moshaiov 2016, `Multi-objective_topology…pdf`) | parents∪offspring sorted by NSGA-II; *species* selected first (≤ N/#obj), then individuals round-robin across species by within-species rank; crossover "fitter parent" = the better within-species rank | The most principled option, but it replaces the generational loop with μ+λ, which is a large rewrite of `reproduceAndSelect`. **Deferred to a possible phase 8.** Borrow its crossover rule. |
| **NSGA-II** (Deb 2002) | fast non-dominated sort + crowding distance | The **core we adopt**: ranking and crowding. |
| **NSGA-III** (Deb & Jain 2014) | reference points replace crowding, for ≥ 4 objectives | Only needed if a task cannot be grouped to ≤ 3–4 objectives (for example the thesis's 7-objective packaging task). **Deferred.** The design keeps the comparator swappable so it can be added later. |

---

## 2. Where scalar fitness drives the algorithm today (the change surface)

Everything below lives under `neat-dnfs/`. Line numbers are as of commit `2f26cfb976`.

| # | Site | Current use of `fitness` | Pareto-mode replacement |
|---|---|---|---|
| 1 | `Species::sortMembersByFitness` (species.cpp) → `pruneWorsePerformingMembers`, `assignChampion` | sort descending by fitness | sort with the **comparator** (rank ↑, crowding ↓) |
| 2 | `Species::assignChampion` improvement flag | `members[0].fitness > champion.fitness` | "a member of this species entered the Pareto archive this generation" |
| 3 | `Population::calculateAdjustedFitness` | `fitness / speciesSize` | `selectionFitness / speciesSize` (see §3.3) |
| 4 | `Population::assignOffspringBasedOnAdjustedFitness` | adjusted fitness | unchanged (it already uses adjusted fitness) |
| 5 | `Population::hasFitnessImprovedOverTheLastGenerations` | best fitness increased | "the archive accepted ≥ 1 new point this generation" |
| 6 | `Population::sortSpeciesListByChampionFitness`, `getBestActiveSpecies` | champion fitness | champion **comparator** |
| 7 | `Solution::crossover` (solution.cpp:622) | fitter parent by fitness; `fitnessDifference < 1e-6` ⇒ random disjoint inheritance | fitter parent by **comparator**; "neither preferred" (same rank *and* same crowding, or mutually non-dominated at the same rank; see §3.3) ⇒ the existing random-inheritance branch |
| 8 | `Population::upkeepBestSolution` / `bestSolution` | max fitness | **unchanged**: still the max weighted sum (the "a posteriori decision maker") |
| 9 | `Population::preserveGlobalBestSolution` | evicts the min-fitness solution | evict the **worst by comparator** |
| 10 | `Population::endConditionMet` | `bestSolution.fitness > targetFitness` | **unchanged (D2, confirmed)**: weighted sum > `targetFitness`, so both modes stop under the same rule |
| 11 | `Population::mutate` elitism exemption | best + champions skip mutation | unchanged |
| 12 | `validateElitism` | scalar best must survive | unchanged (bestSolution is still scalar-best and still preserved) |

Scalar fitness is still computed by every task's `testPhenotype()` in both modes. It keeps
feeding `bestSolution`, `overview.jsonl`, every existing file and the visualizer, so runs stay
comparable across modes.

---

## 3. Design

### 3.1 Objective vector

- New `SolutionParameters` fields: `std::vector<double> objectives; int paretoRank{-1};
  double crowdingDistance{0.0}; double selectionFitness{0.0}; double constraintViolation{0.0};`
  **Do not** change `SolutionParameters::toString()` or `operator==`. The visualizer's regexes
  parse `toString()` output (`_PARTIAL_FIT_RE`, `parse_solution_blob`), and
  `statistics/generation_N.txt` must stay byte-compatible.
- `objectives` is derived from `partialFitness` **in the base class**, right after
  `testPhenotype()` in `Solution::evaluate()`. No task file changes.
- `groupObjectives(partials, groups, weights)` computes each objective as the weighted mean of its
  group's partials, using the task's existing `fitnessWeights` renormalised *within* the group
  (equal weights ⇒ plain mean). An empty `objectiveGroups` means one objective per partial.
- Validate at construction (`Solution::loadFitnessWeights(slug, expectedCount)` already knows the
  partial count). The groups must be a partition of `[0, expectedCount)`: every index exactly
  once, and no empty group. Throw `std::runtime_error` naming the task otherwise, matching the
  existing wrong-length-weights behaviour.
- Also add a test that every task's `partialFitness.size() == fitnessWeights.size()` after
  `evaluate()`. The grouping relies on it, so check first whether `test_solutions_tasks.cpp`
  already covers it.

### 3.2 Configuration (JSON-driven, per CLAUDE.md)

Add a new top-level block to `config/neat_dnfs.json`:

```json
"SelectionConstants": {
  "mode": "scalar",
  "objectiveGroups": [],
  "dominanceEpsilon": 0.0,
  "feasibilityFloor": 0.0,
  "archiveCapacity": 100
}
```

| Field | Meaning | Default | Valid range (validate; throw on violation) |
|---|---|---|---|
| `mode` | `"scalar"` = today's selection, `"pareto"` = §3.3 | `"scalar"` | one of the two strings |
| `objectiveGroups` | partition of the partial indices; `[]` = one objective per partial | `[]` | checked per task at solution construction (§3.1) |
| `dominanceEpsilon` | ε in `dominates()`, absorbing simulation noise | `0.0` | `[0, 0.5)` |
| `feasibilityFloor` | a raw partial below it makes the solution infeasible (§3.3); `0` = off | `0.0` | `[0, 1)` |
| `archiveCapacity` | maximum number of points in the Pareto archive | `100` | `≥ 1` |

- `constants.h`: add `struct SelectionConstants` with these fields, each with its default as an
  in-class initialiser. `mode` is an `enum class SelectionMode { Scalar, Pareto }`, parsed from a
  string, and an unknown string is an error. Add `static void reset()` restoring every default,
  like `AblationConstants::reset()`, so tests stay isolated.
- `config_loader.cpp`: add `"SelectionConstants"` to `checkNoUnknownTopLevelKeys`'s `known` set
  and add a block in `applyConfig`.
- **Decision D1 (confirmed): the block is optional.**
  - If `SelectionConstants` is absent, call `SelectionConstants::reset()` and do not throw. A
    user's pre-existing full `--config` file therefore keeps working and runs in scalar mode.
  - If the block is present, each of its keys is still optional individually and falls back to
    its default. A **mistyped key inside the block** (for example `"dominanceEpsilion"`) must
    throw, because otherwise the optionality hides typos. Check the block's keys against the known
    field set. That is stricter than the existing blocks, and deliberately so.
  - The same applies to the new `PopulationConstants.saveObjectives`: optional, default `true`.
    Read it with `if (pc.contains("saveObjectives"))`; the other `PopulationConstants` keys stay
    required.
  - Document the exception in the `ConfigLoader` class comment in `config_loader.h`, which
    currently claims "nothing here has a compiled-in fallback". Name these fields as the
    exception and give the reason (backwards compatibility with user configs).
  - The reference `config/neat_dnfs.json` still lists every field explicitly, so it stays the
    complete reference set.
  - Tests: (1) a config with no block loads, and mode == scalar; (2) a partial block fills the
    remaining fields with defaults; (3) a mistyped key inside the block throws; (4) no
    `saveObjectives` key ⇒ `true`.
- Per-task groups go in `config/solutions/<task>.json` as a sparse override
  (`SelectionConstants.objectiveGroups`); see §3.6 for the proposals.
- Turning Pareto on: new preset `config/ablations/pareto-selection.json`:
  ```json
  { "AblationConstants": { "label": " Pareto" },
    "SelectionConstants": { "mode": "pareto", "dominanceEpsilon": 0.01, "feasibilityFloor": 0.1 } }
  ```
  The label suffix gives Pareto runs their own `data/<Task> Pareto/` folder, so the dashboard's
  Compare page shows scalar against Pareto with no extra work. Known limitation: only one
  `--ablation` can be passed at a time, so Pareto × no-crossover needs its own preset file.
  Accept that for v1.
- `ConfigLoader::applyAblation` merges any top-level block, so no loader change is needed beyond
  the `known` set. Verify this with a test.

### 3.3 The Pareto machinery: new `include/neat/pareto.h` + `src/neat/pareto.cpp`

These are pure free functions in namespace `neat_dnfs` with no RNG and no globals, so they are
deterministic and easy to test. The module goes under `neat/`, not `neat_tools/`, because it is
core algorithm and not a general helper (it does not duplicate anything in `neat_tools/`; check
`utils.h` anyway before writing anything).

```cpp
/// One individual as the ranking sees it.
struct RankedPoint { std::vector<double> objectives; double violation{0.0}; };

/// maximisation; a dominates b iff a_i >= b_i - eps for all i and a_j > b_j + eps for some j
[[nodiscard]] bool dominates(std::span<const double> a, std::span<const double> b, double epsilon);
/// Deb 2002 constrained-domination: a feasible point (violation == 0) beats an infeasible one;
/// between two infeasible points the smaller violation wins; between two feasible points, dominates()
[[nodiscard]] bool constrainedDominates(const RankedPoint& a, const RankedPoint& b, double epsilon);
/// sum over partials of max(0, floor - partial_i); 0 when floor == 0 (floor disabled)
[[nodiscard]] double constraintViolation(std::span<const double> partials, double floor);
/// Deb 2002 fast non-dominated sort under constrainedDominates; fronts as index lists, front 0 first
[[nodiscard]] std::vector<std::vector<size_t>> nonDominatedSort(std::span<const RankedPoint> points, double epsilon);
/// NSGA-II crowding distance for the members of one front (boundary points = +inf; a
/// zero-range objective contributes 0), returned in the order of `front`
[[nodiscard]] std::vector<double> crowdingDistances(std::span<const RankedPoint> points, std::span<const size_t> front);
[[nodiscard]] std::vector<double> groupObjectives(std::span<const double> partials, const std::vector<std::vector<size_t>>& groups, std::span<const double> weights);
```

With the floor off (violation always 0), `constrainedDominates` reduces to `dominates`, so
`feasibilityFloor = 0` is plain NSGA-II.

**`ParetoArchive` contract** (same header):
- It holds entries `{solutionId, speciesId, generationFound, objectives, partials, fitness}`.
- It contains **feasible points only** (violation 0).
- `[[nodiscard]] bool tryInsert(entry, epsilon)`:
  - reject the entry if an existing member dominates it, or is ε-equal to it on every objective;
  - otherwise remove every member it dominates and add it;
  - above `capacity`, evict the member with the smallest crowding distance in the archive, and
    never evict a boundary point;
  - return whether the entry is now in the archive.
- `members()`, `size()`, and `[[nodiscard]] double bestViolationSeen() const`, used for the
  improvement signal when nothing is feasible yet (see below).
- Tests: insert into empty; reject a dominated or equal entry; evict dominated members; truncate at
  capacity (boundary kept); reject an infeasible entry.

Every public entity gets Doxygen (`@brief`, one `@param` per parameter, `@return`). CI enforces
this through `gemini-doc-sync.yml`. Add both files to the **explicit** lists in `CMakeLists.txt`
(around lines 140–176): there is no GLOB, and a missed file is silently not compiled.

**Population wiring (Pareto mode only):**
- A new private `Population::rankObjectives()` runs on the main thread immediately after
  `evaluate()`, inside the same profiler scope, or in a new `"rank"` scope that is added to
  `profile.csv`'s fixed column list. It:
  1. builds a `RankedPoint` per solution from `objectives` and
     `constraintViolation(partialFitness, feasibilityFloor)`, and stores the violation in a new
     `SolutionParameters::constraintViolation` field;
  2. runs `nonDominatedSort(points, dominanceEpsilon)`;
  3. computes `crowdingDistances` per front;
  4. writes `paretoRank`, `crowdingDistance` and
     `selectionFitness = (numFronts − rank) / numFronts` (which lies in (0, 1] and needs no
     weights);
  5. offers every **feasible** front-0 member to the `ParetoArchive`, recording which ids and
     species were accepted this generation. Those are the improvement signals for rows 2 and 5 of
     §2:
     - *population improved* ⇔ the archive accepted ≥ 1 entry this generation, **or**, while the
       archive is still empty, the population's minimum violation fell below
       `bestViolationSeen()` by more than `dominanceEpsilon`;
     - *species improved* ⇔ one of its members was accepted this generation (or, while the
       archive is empty, that species holds the new minimum violation). Feed this into the
       existing `generationsSinceFitnessImproved` counter in place of the fitness comparison, so
       the threshold constants and the stagnation logic stay exactly as they are.
- Log one DEBUG sentence per generation from here, for example: "gen 7: 5 fronts, front 0 holds
  23 of 300 (18 feasible), archive 41 (+3 accepted), 2 species improved, 4 stagnant". This is a
  good permanent DEBUG line.
- The ranking runs after all solutions are evaluated and before `speciate()`. `speciate()`
  calls `assignChampion`, which needs the ranks. Do not run it inside the parallel evaluation.
- In scalar mode, `selectionFitness = fitness`, and the ranking step still runs **for output
  only** (§3.4). It draws no RNG and does not affect selection. If the reviewer objects, gate it
  behind `saveObjectives`.

**The comparator:** `[[nodiscard]] bool Solution::isPreferredTo(const Solution& other) const`.
- Scalar mode: `fitness > other.fitness`, the exact expression used today.
- Pareto mode: `rank < other.rank || (rank == other.rank && crowding > other.crowding)`.
- For crossover's "equal parents" test (row 7 of §2), add
  `[[nodiscard]] bool Solution::isEquivalentForSelection(const Solution& other) const`.
  - Scalar mode: `|Δfitness| < 1e-6`, today's expression.
  - Pareto mode: `rank == other.rank && !constrainedDominates(either way)`.
- The crowding values used by the comparator are computed **globally, per front** (NSGA-II), not
  within the species. `pruneWorsePerformingMembers` therefore keeps a species' members that sit
  in sparse regions of the *population's* front. This is the NEAT-MODS within-species order
  simplified to reuse the global sort.
- **Decision D4 (confirmed): elitism rules are unchanged.** `bestSolution` stays the scalar-best
  and is preserved by `preserveGlobalBestSolution`. Species champions (species > 5 members) are
  copied as today, but chosen by comparator. Nothing else is added: no extreme-point elitism and
  no archive re-injection into the population.

  Note that this changes how often crossover takes the random-inheritance branch in Pareto mode.
  That is intended: NEAT-MODS makes the same move.

**Invariant: the scalar path must be unchanged.** Every rewired call site must reduce to today's
expression in scalar mode, and must add **no RNG draws** (seeding, issue #44, has not landed, so
bit-identical regression tests are impossible — review this by reading the diff). All 196 tests
of `ctest -LE slow` must pass untouched.

### 3.4 Output: new files only; no existing format changes

- `objectives.jsonl` in the run directory, one line per generation, written in **both** modes:
  ```json
  {"generation": 7, "mode": "pareto", "epsilon": 0.01, "feasibilityFloor": 0.1,
   "objectiveGroups": [[0,2],[1,3,4],[5,6,7]],
   "individuals": [{"id": 812, "species": 3, "fitness": 0.91,
                    "partialFitness": [...], "objectives": [...],
                    "rank": 0, "crowding": 0.42, "violation": 0.0}, ...],
   "archive": {"size": 37, "acceptedThisGeneration": [812, 830]}}
  ```
  It is gated by a new `PopulationConstants::saveObjectives` flag (default `true`), following the
  `saveStructuredOverview` precedent from v0.3.0. Write it from `PopulationFileManager` next to
  `savePerGenerationOverviewJson`.
- At the end of a Pareto run: write `pareto_archive.json` (id, generation found, objectives,
  partials, fitness) and save each archive member's phenotype under `pareto_front/`, the same way
  `saveAllSolutionsWithFitnessAbove` writes `best_solutions/last_generation/`. Keep only the
  archive members that are still alive in the final population, since the phenotype of a dead
  solution is gone. Say so in the Doxygen.
- `overview.jsonl`: optionally add a `"pareto": {"front0Size", "numFronts", "archiveSize"}` key.
  The visualizer does not read `overview.jsonl` yet (it parses the prose file), so this is safe.
  Grep `analysis/` to confirm before adding it.

### 3.5 Visualizer: new "Pareto" page (`analysis/viz/`)

Follow the existing structure: page function in `app.py`, `render_pareto_view` in `views.py`,
charts in `plots.py` (**Altair**, colours from `theme.py`; load the `dataviz` skill before
writing any chart), and pure logic in a **new `viz/pareto.py`** with pytest coverage in
`analysis/tests/test_pareto.py`.

**Data loading (`parsing.py`):**
- `load_objectives(run_dir)` reads `objectives.jsonl` when it exists.
- **Fallback for every existing run:** `_scan_run_statistics_uncached` already reads each
  individual's partial vector from `statistics/generation_N.txt` but only keeps
  best/average. Extend it to also return a per-individual `obj_df` with columns
  `generation, id, species, fitness, p1..pN`. This changes the on-disk cache contents
  (`statistics_scan.*.parquet` plus `meta.json`), so add a cache-schema version to the meta, and
  treat a mismatch as a miss. Otherwise stale caches will be read.
- For fallback runs, compute rank and crowding in Python (`viz/pareto.py`) with a user-chosen ε
  and grouping.

**`viz/pareto.py` (pure, numpy):** `dominates`, `non_dominated_sort` (vectorised O(mN²)),
`crowding_distance`, `group_objectives`, `hypervolume`.
- `hypervolume` works in [0,1]^m with reference point 0. It is exact for m ≤ 3 (2-D sweep; 3-D
  by slicing) and Monte-Carlo for m > 3 (fixed seed, 100k samples; the tooltip must say
  "estimate").
- All partials are expected in [0,1]. Clip values outside that range and show a warning if any
  appear.
- Tests use hand-computed fronts: a 2-D staircase, duplicates, an ε case, the crowding boundary
  = inf, and the HV of a single point = the product of its coordinates.

**Page layout (validated shape: `sandbox/proto_pareto_views.png`):**
1. **Controls.** Generation slider (reuse the `display_gen` 1-based convention). Objective space:
   raw partials / grouped (the groups default to the run's recorded `objectiveGroups`; for old
   runs, a text input like `0,2 | 1,3,4 | 5,6,7`). ε and feasibility-floor sliders, which
   default to the run's recorded values (or 0 for old runs) and only affect post-hoc ranking. x/y
   objective pickers.
2. **KPI row.** Front-0 size (and %), number of fronts, feasible share (when the floor is > 0),
   specialist share of front 0 (some raw partial < 0.1: the §1.3 metric), hypervolume, archive
   size, m, and
   "rank of the scalar-best individual". The last one directly shows whether the weighted sum is
   picking a front solution.
3. **2-D front scatter.** The whole population, coloured by m-D rank (front 0 in the accent
   colour, other ranks in graded neutrals; infeasible points drawn hollow), with a toggle to
   colour by species. Overlay the 2-D
   non-dominated staircase as a dashed line and **label it as the 2-D projection**, because a
   projection of an m-D front is not a front. Tooltip: id, species, fitness, partials,
   objectives.
4. **Scatter matrix** of all objective pairs for front 0 (Altair `repeat`). Hide it when
   m > 8.
5. **Parallel coordinates** of front 0 (Altair fold transform), with the scalar-best highlighted.
   This is the main many-objective view, and it is where degenerate specialists show up (§1.3).
6. **Front evolution.** Front 0 of evenly sampled generations on the chosen pair, with a
   sequential colour ramp by generation (reuse `_render_sampling_controls`).
7. **Metrics over generations.** HV (feasible points only), front-0 %, number of fronts,
   feasible %, specialist % (line charts).
8. **Objective conflict heatmap.** The correlation matrix across the population on a diverging
   ramp −1..1. Negative cells are real trade-offs, and near-+1 cells are redundant objectives
   (Memory Instability f2↔f4 = 0.93). This is the evidence the user needs to choose groupings.
9. **Front-0 table** with row select that feeds the existing `_render_solution_record` genome
   inspector.

**Across runs:**
- Experiment page: the distribution of final-generation HV across runs, and the union of final
  fronts across runs on a chosen pair.
- Compare page: an HV box plot per experiment with a Mann–Whitney U test. Reuse
  `stats.mann_whitney_u` and `chart_cross_experiment_boxplot`. This is the scalar-vs-Pareto
  verdict view.
- `report.py`: add a "Pareto" section to the run export when objective data exists.

Update `_PAGE_TAGLINES`, `_SCOPE_RUN` and the navigation. Old runs must render without errors:
where data is missing, show a plain "not recorded" state, as the Provenance page does.

### 3.6 Objective groups per task (confirmed, D3; 0-based indices into `partialFitness`)

Each grouping follows the design rule of §1.3 and has been checked against a real run (§1.4). Put
each one in its task's `config/solutions/<task>.json` as
`"SelectionConstants": {"objectiveGroups": [...]}`. The `sandbox/grouping_check.py` command in the
last column reproduces the evidence.

| Task | partials (push order in `testPhenotype`) | `objectiveGroups` | Objectives | Check |
|---|---|---|---|---|
| and (8) | f1_1 f1_2 f2_1 f2_2 f3 f4_1 f4_2 f4_3 | `[[0,2],[1,3,4],[5,6,7]]` | input representation / AND output logic / return to rest | `"0,2\|1,3,4\|5,6,7"` |
| xor (4) | f1 f2 f3 f4 | `[[0,1,2],[3]]` | XOR output logic / return to rest | `"0,1,2\|3"` |
| detection-instability (4) | f1 f2 f3 f4 | `[[0,1],[2,3]]` | detect / return to rest | `"0,1\|2,3"` |
| memory-instability (4) | f1 f2 f3 f4 | `[[0,2],[1,3]]` | nf1 tracks the stimulus / nf2 forms and holds memory | `"0,2\|1,3"` |
| selection-instability (4) | f1 f2 f3 f4 | `[[0,1],[2,3]]` | select one of two / return to rest | `"0,1\|2,3"` |
| memory-trace (8) | f1 f2 f5 f6 f7 f8 f9 f10 | `[[0,2],[1,3,4],[5,6,7]]` | trace formation / input handling / trace-biased selection and decay | `"0,2\|1,3,4\|5,6,7"` |
| dmts (6) | f1 f2 f3 f4 f5 f6 | `[[0,2,4],[1,3,5]]` | nf1 perception / nf2 memory and match | `"0,2,4\|1,3,5"` |
| ior (5) | f1 f2 f3 f4 f5 | `[[0,2],[1,3,4]]` | nf1 perception / nf2 inhibition-of-return | `"0,2\|1,3,4"` |

Add a test that loads every task's config and constructs its solution, so each grouping is
validated as a partition of that task's partial count. This also catches a future change to a
task's partial count that forgets its grouping.

---|---|---|---|
| and (8) | f1_1 f1_2 f2_1 f2_2 f3 f4_1 f4_2 f4_3 | `[[0,2],[1,3,4],[5,6,7]]` | input representation / AND output logic / return to rest (**validated**) |
| xor (4) | f1 f2 f3 f4 | `[[0,1,2],[3]]` | XOR output logic (fire on A, fire on B, silent on A+B) / return to rest |
| detection-instability (4) | f1 f2 f3 f4 | `[[0,1],[2,3]]` | detect / return to rest (the instability's own trade-off) |
| memory-instability (4) | f1 f2 f3 f4 | `[[0,2],[1,3]]` | nf1 tracks the stimulus / nf2 forms and **holds** memory (f2↔f4 r = 0.93 supports this) |
| selection-instability (4) | f1 f2 f3 f4 | `[[0,1],[2,3]]` | select one of two / return to rest |
| memory-trace (8) | f1 f2 f5 f6 f7 f8 f9 f10 | `[[0,2],[1,3,4],[5,6,7]]` | trace formation / input handling / trace-biased selection and decay (**low confidence**) |
| dmts (6) | f1 f2 f3 f4 f5 f6 | `[[0,2,4],[1,3,5]]` | nf1 perception / nf2 memory and match (**low confidence**) |
| ior (5) | f1 f2 f3 f4 f5 | `[[0,2],[1,3,4]]` | nf1 perception / nf2 inhibition-of-return behaviour (**low confidence**) |

Before committing any grouping, run the probe on real data. The visualizer's conflict heatmap
(item 8) is the tool for this once it exists; until then use `sandbox/pareto_probe.py`. The
check: front 0 should hold ≲ 20 % of the population in mid-run generations, and the
parallel-coordinates view must not show front-0 members at 0 on whole scenarios.

---

## 4. Backwards-compatibility checklist (mandatory; verify each)

- [ ] Default `mode = "scalar"`, and every rewired call site reduces to today's exact expression
      in scalar mode, with no new RNG draws.
- [ ] `ctest -LE slow` passes unchanged. `ctest -L slow` behaves like unmodified `main` (it is
      stochastic: re-run before calling anything a regression).
- [ ] `ctest -L Golden` still passes.
- [ ] `Solution::toString()`, `per_generation_overview.txt`, `statistics/*.txt` and existing
      JSON outputs are byte-format-unchanged. New data goes only into new files or new keys.
- [ ] Every existing run under `data/` opens on every existing dashboard page, and on the new
      Pareto page through the fallback.
- [ ] Visualizer disk caches: the schema version is bumped, and old caches are treated as a miss
      rather than misread.
- [ ] Config: the handling of a user's custom `--config` file with no `SelectionConstants` block
      follows decision **D1**: no block ⇒ scalar mode, no error (§3.2).
- [ ] `apps/README.md` flags table and the preset list are updated; nothing is removed.

---

## 5. Phases (TDD: write the failing test first, watch it fail, then implement)

Each phase is a separate branch and PR, named per CLAUDE.md (`feat/…`, `test/…`), with
conventional lowercase commits. Build and test through the **`build-and-test` skill**
(`cmake --build build/x64-release --config Release --target neat-dnfs-test --parallel 4`, then
`ctest -LE slow` from the build tree). Add every new test file to the explicit test list in
`CMakeLists.txt` (around lines 380–404); a missing entry means the test never runs while the suite
still passes.

**Phase 1 — `pareto` core** (`feat/pareto-core`)
- Tests `tests/test_pareto.cpp`, tagged `[Pareto]`, must cover:
  - dominance, with and without ε, including equal vectors (not dominated);
  - the sort on a known 2-D set: exact front membership and order;
  - all-identical points: one front;
  - crowding with boundary points = +inf, and interior values checked by hand;
  - `groupObjectives` with equal and unequal weights;
  - `constraintViolation`: floor 0 ⇒ always 0; hand-computed shortfall sums;
  - `constrainedDominates`: feasible beats infeasible even when the infeasible point is better
    on every objective; smaller violation wins between infeasible points; it reduces to
    `dominates` when both violations are 0;
  - the sort with a mix of feasible and infeasible points: every feasible point ranks ahead of
    every infeasible one;
  - the archive contract as listed in §3.3 (insert, reject dominated or equal, evict dominated,
    capacity truncation keeps boundary points, reject infeasible).
- Implement. Verify: the new tests pass and the 196 existing tests still pass.

**Phase 2 — config and objective vector** (`feat/selection-config`)
- D1 is settled (§3.2): implement the optional block exactly as specified there.
- Tests:
  - the loader parses `SelectionConstants`;
  - an unknown `mode` string throws;
  - a bad grouping (overlap, gap, out of range, empty group) throws at solution construction and
    names the task;
  - `objectives` equals the grouped partials after `evaluate()` on a stub solution;
  - an ablation preset can set `mode` and `feasibilityFloor`;
  - out-of-range `dominanceEpsilon`, `feasibilityFloor` or `archiveCapacity` throws;
  - the four D1 tests listed in §3.2;
  - `SelectionConstants::reset()` restores the defaults.
- Implement §3.1–3.2. Verify: the full default lane is green.

**Phase 3 — `objectives.jsonl`** (`feat/objectives-output`)
- Tests in `test_population_file_manager.cpp`: after a two-generation run on a stub solution, the
  file has two lines, parses as JSON, contains one individual per solution, and every individual
  has `objectives.size()` equal to the number of groups. Rank 0 is present, and ranks are
  consistent with `constrainedDominates`. With `saveObjectives = false` no file is written.
- Implement §3.4 (the scalar-mode ranking-for-output part).
- Verify: run `neat-dnfs-evol --task xor --runs 1 --pop 50 --gens 3` and inspect the file.

**Phase 4 — visualizer Pareto page** (`feat/viz-pareto`)
- pytest first:
  - `viz/pareto.py` functions (hand-computed cases);
  - fallback parsing of a small synthetic `statistics/generation_1.txt` fixture;
  - `load_objectives` on a synthetic `objectives.jsonl`;
  - cache-schema bump invalidates old meta.
- Implement §3.5. Verify:
  - `pytest` is green in `analysis/`;
  - launch the dashboard (the `run` skill, or `analysis/launch-visualizer.bat`) and open the Pareto
    page on an **old** run (for example `data/AND/2026-09-28 15h14m03s`) and on a new Phase-3
    run;
  - screenshot each section and check them against `sandbox/proto_pareto_views.png`;
  - check light and dark theme.

**Phase 5 — Pareto selection mode** (`feat/pareto-selection`)
- D2 and D4 are settled: the end condition and elitism stay as they are (§2 row 10, §3.3).
- Add a stub solution to `tests/test_stub_solution.h`, `ObjectiveStubSolution`, whose
  `testPhenotype` sets two *conflicting* deterministic partials from the genome (for example
  `numFieldGenes/10` and `1 − numConnectionGenes/10`) and `fitness` = their mean.
- Unit tests, one per rewired row of §2:
  - in both modes: sort order, champion, crossover fitter-parent / equal-parent choice,
    adjusted fitness from `selectionFitness`, and the worst-evicted solution;
  - improvement signals from the archive, including the archive-empty (all infeasible) fallback;
  - in scalar mode, the comparator must agree with `fitness >` on random pairs.
- Integration test (`[Population]`, fast): a Pareto-mode `evolve()` on the stub runs 5
  generations under `ValidationPolicy::Throw` with zero violations, `objectives.jsonl` ranks are
  consistent, and the archive is non-empty.
- Implement §3.3.
- Instrument, don't guess: read the per-generation DEBUG sentence from §3.3 on one real `and` run
  (`Logger::setMinLogLevel(DEBUG)`). Confirm that the archive grows early and then settles, that
  the feasible share rises roughly as in §1.5, and that stagnation fires eventually rather than
  never. That last point is the noise-churn risk in §7. If the archive churns every generation,
  raise ε before going further.

**Phase 6 — presets, groups, docs** (`feat/pareto-preset`)
- D3 is settled: use the §3.6 groupings as given.
- Add `config/ablations/pareto-selection.json` and `objectiveGroups` in each
  `config/solutions/<task>.json`.
- Add `config/ablations/pareto-selection-no-floor.json` (label `" Pareto NoFloor"`, floor 0).
  It exists for the Phase 7 floor comparison.
- Add an `[Evolution]` slow test, at the lane's standard budget: `pareto` mode on `xor` and `and`
  completes; every generation's `objectives.jsonl` ranks are consistent; the archive is
  non-empty by the last generation on `and`; and the recorded best fitness never falls below its
  high-water mark by more than `elitismFitnessEpsilon` (elitism is unchanged, D4). Do **not**
  assert success rates here: they are stochastic, and Phase 7 establishes them.
- Docs:
  - `apps/README.md`: preset list and a worked example;
  - `CHANGELOG.md` `[Unreleased]`;
  - Doxygen on every new or changed public entity;
  - run the `docs-check` skill.
- Then run `project-code-review` and `pr`.

**Phase 7 — evaluation (decides whether Pareto becomes a recommended default for any task)**
- Protocol: for `xor`, `and` and `selection-instability`, run
  `--runs 30 --pop 500 --gens 200` in scalar and in Pareto mode (Pareto through
  `--ablation pareto-selection`). This is the same budget and the same target (D2).
  - Floor arm: `and` and `xor` only, also under `--ablation pareto-selection-no-floor`.
  - Budget warning: 8 arms × 30 runs at pop 500 × 200 gens is long. Ask the user for the machine
    time or a reduced protocol before launching, and run it in the background.
- Metrics, all from the dashboard:
  - success rate, where success means *every* partial is at or above its target (the definition
    the lit review uses);
  - generations to success;
  - final hypervolume in grouped objective space;
  - front-0 spread;
  - specialist share of front 0 (§1.3 metric);
  - number of active species over time (the §7 offspring-allocation risk);
  - evolved architecture size.
- Statistics: Mann–Whitney U on the Compare page.
- Write the result to `.claude/notes/` (tracked) as a short findings note.

**Phase 8 (optional, only if Phase 7 shows front regression or poor diversity):** NEAT-MODS-style
μ+λ species-first selection, or NSGA-III reference points for tasks that cannot be grouped to
≤ 4 objectives. Both slot in behind the same `SelectionMode` enum and comparator.

---

## 6. Decisions (confirmed by the user, 2026-09-29; do not re-ask)

| # | Question | Decision | Where it is specified |
|---|---|---|---|
| D1 | A user's custom `--config` with no `SelectionConstants` block | **Optional block.** Missing ⇒ scalar mode with defaults, no error. The same goes for `PopulationConstants.saveObjectives` (default `true`). Mistyped keys *inside* the block still throw. | §3.2 |
| D2 | End condition in Pareto mode | **Unchanged**: `bestSolution.fitness > targetFitness` (scalar best), so both modes stop under the same rule and stay comparable | §2 row 10 |
| D3 | Objective groupings per task | **Accepted as proposed** in §3.6. All eight have since been checked against real runs (§1.4). | §3.6 |
| D4 | Elitism in Pareto mode | **Keep today's rules**: scalar-best preserved plus species champions (chosen by comparator); no extra elitism | §3.3 |

Added after the decisions, and within their spirit: the **feasibility floor** (§1.5, §3.2–3.3).
It is off by default, so it changes nothing for existing users, and is on (0.1) only in the
Pareto preset. If Phase 7 shows the floor hurts, set it to 0 in the preset. That needs no code
change.

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Too many objectives, so rank stops discriminating (seen at 8 raw objectives) | groupings (§3.6), ε-dominance, and the KPI "front-0 %" on the Pareto page; NSGA-III in Phase 8 |
| Degenerate specialists on the front (seen in §1.3) | the grouping rule in §1.3, plus the feasibility floor (§1.5), which removed them from AND's front 0 from generation 5; the parallel-coordinates view makes them visible |
| Floor too strict for a hard task (XOR's 20-generation run had no individual clearing 0.1 on every partial) | while nothing is feasible, ranking by total shortfall still orders the population, and the archive-empty improvement rule (§3.3) keeps stagnation detection working; Phase 7 compares floor 0.1 against 0 on AND and XOR |
| Simulation noise (see `elitismFitnessEpsilon`'s comment: swings of ~0.1 near bump boundaries) flips dominance and churns the archive, which then always looks "improving" and hides stagnation | preset ε = 0.01, tuned in Phase 7; the archive uses the same ε |
| Rank-based `selectionFitness` changes how offspring are allocated between species (flatter than raw fitness) | intended, since it removes the absolute-difference crowd-out that Inden et al. describe; watch species counts on the Species page in Phase 7 |
| The scalar path is silently altered | §3.3 invariant, the §4 checklist, and review focused on RNG draws and comparator equivalence |
| Performance | the sort is O(m·N²): N = 1000, m = 8 is about 8M comparisons per generation, negligible next to the DNF simulations. Check with the `NEAT_DNFS_PROFILE` build (`profile.csv`) |
| Visualizer caches misread | schema version in the cache meta (§3.5) |

## 8. Out of scope for this plan

A structural-complexity objective (Clune et al. 2013), a novelty/diversity objective (Silva et
al.), NEAT-MODS μ+λ selection, NSGA-III, and objective names in config (the dashboard keeps using
`p1..pN` / `g1..gK` labels). Each is a candidate follow-up once Phase 7 has a verdict.

---

## Implementation notes (added during implementation, 2026-09-29)

### Environment and test counts
- **The "196 tests" figure is stale.** On v0.3.0, `ctest -C Release -LE slow` runs **207** tests
  (205 `fast.*` + 2 `golden.*`; `-LE slow` excludes only `slow`, so the Golden label is in it).
- **On this machine 3 of them fail on unmodified `main`**, caused by a stale system-wide
  `dnf-composer` in `C:/Program Files/` (there is no `deps/`); see
  `.claude/local-notes/stale-dnf-composer-install.md`. The regression bar used in every phase is
  therefore "the same tests pass, the same 3 fail".
- clang-tidy runs over `src/neat/*.cpp` with `WarningsAsErrors: '*'`, so every `if`/`for` body
  in new `src/neat` code needs braces. Running it locally works for self-contained files:
  `clang-tidy src/neat/pareto.cpp -- -std=c++20 -Iinclude`.

### Phase 1 (PR #117): underspecified points, and how they were resolved
- **ε-dominance can be cyclic.** With ε > 0 and m ≥ 3, a ≻ b ≻ c ≻ a is possible, for example
  a=(.25,.18,.10), b=(.10,.25,.18), c=(.18,.10,.25), ε=.1. Plain fast non-dominated sort then
  never places those points. Resolution: when no remaining point has domination count 0, the
  remaining points with the *smallest* count form the next front. The Python `non_dominated_sort`
  (Phase 4) must do the same, or the two will disagree.
- **`ParetoArchiveEntry` needs a `violation` field.** It is not in the §3.3 list, but "reject
  infeasible" needs it. `bestViolationSeen()` is the minimum violation of *every* entry offered
  to `tryInsert`, including rejected infeasible ones, and starts at +inf. For the archive-empty
  improvement rule to work, Phase 5 must offer every front-0 member, infeasible ones included
  (not just "every feasible front-0 member" as §3.3 step 5 says). While nothing is feasible,
  front 0 is exactly the minimum-violation set, so this is enough. Phase 5 must read
  `bestViolationSeen()` *before* offering the generation's entries.
- **"Never evict a boundary point" can be impossible.** With capacity 1 (or 2 for m = 2), every
  member is a boundary point. Resolution: the newcomer is the one dropped, and `tryInsert`
  returns false. More generally, `tryInsert` returns false whenever the newcomer is itself the
  most crowded member.
- **Crowding with a zero-range objective**: it contributes nothing, *including* the +inf
  boundary marks, so a front of identical points has all-zero crowding distances. A front of
  size ≤ 2 is all +inf.
- **`groupObjectives` with group weights summing to 0** uses the plain mean. No shipped task has
  a zero weight.
- Fronts list indices in ascending order, which makes the sort output deterministic.

### Plan text
- §3.6 has a leftover second table after the first one (it starts with a stray `---|---|---|---|`
  row). It still carries the old "low confidence" labels that §1.4 retired. The first table is
  the authoritative one.

### Phase 2 (PR #118, stacked on #117)
- **Stacked PRs get no CI.** `ci.yml` and `static-analysis.yml` trigger only on
  `pull_request: branches: [main]`. Each stacked branch gets a manual
  `gh workflow run ci.yml --ref <branch>` (and the same for `static-analysis.yml`). The results
  show on the branch's commits, not on the PR's checks tab.
- **Where the groups come from.** `Solution::loadFitnessWeights(slug, n)` validates
  `SelectionConstants::objectiveGroups` (the *merged* config: reference → task → ablation →
  `--config`) and copies it per instance, next to the weights. Unlike the weights, the groups are
  therefore not re-read from `config/solutions/<slug>.json`, so an ablation preset or a user config
  can override them, which is what §3.2 wants.
- **Phase 6 must fix the test helper `ScopedTaskConfig` (tests/test_helpers.h) first.** It calls
  `ConfigLoader::loadConfig(ref, slug)` but restores only `xSize`/`dx`. Once the task configs carry
  `objectiveGroups`, `ScopedTaskConfig("and")` leaves AND's 8-index groups in the global, and the
  next test that constructs a 4-partial task (XOR) will throw. Make it restore `SelectionConstants`
  (or call `SelectionConstants::reset()`) on destruction.
- The `SolutionParameters` fields are added when first used, not all in Phase 2: `objectives` now,
  `paretoRank`/`crowdingDistance`/`constraintViolation` in Phase 3 (output), `selectionFitness` in
  Phase 5.
- "Every task's `partialFitness.size() == fitnessWeights.size()`": already covered.
  `test_solutions_tasks.cpp` asserts each task's exact partial count after `evaluate()` (4/4/4/8/6/5/8/4),
  and the loader rejects a weight array of any other length. No new test was added.
- A stray `.claude/notes/tsan-random-initial-topology-flake.md` (untracked, not in any PR) explains
  why PR #117's TSan job failed: a pre-existing hang/slowness in one ablation test, also seen on `main`.

### Phase 3 (PR #119, stacked on #118)
- **Infinite crowding is written as `null`** in `objectives.jsonl`, because JSON has no infinity
  (nlohmann would write `null` silently anyway). The Phase 4 loader must map `null` → `inf`.
- **The ranking is gated on the file being written** (`fileManager && saveObjectives`), since
  nothing else reads it yet. Phase 5 must widen the gate `Population::isRankingObjectives()` to
  `|| SelectionConstants::mode == SelectionMode::Pareto`.
- **An archive entry's `speciesId` is stale.** The ranking runs *before* `speciate()` (as §3.3
  requires), so `ParetoArchiveEntry::speciesId` is whatever the solution carried from the previous
  generation. The file's per-individual `species` is written after `speciate()`, so it is current.
  Phase 5's "species improved" signal must map the accepted *ids* to their species after
  `speciate()`, not trust `entry.speciesId`.
- `acceptedThisGeneration` means accepted at insertion. A later insertion in the same generation
  can evict an accepted entry, but the archive still changed, so the improvement signal holds.
- Not done here, on purpose: the optional `pareto` key in `overview.jsonl` (the user's rule is "no
  format change to any existing output file"), and `pareto_archive.json` / `pareto_front/`, which
  §3.4 describes for Pareto *runs*, so they belong in Phase 5. Profiling: the ranking is timed
  inside the `"evaluate"` scope, because a new `"rank"` column would change `profile.csv`.
- Early generations can form an almost total order: XOR gen 0 at pop 50 gave 50 fronts of size 1,
  because near-identical clones differ only in f4. That is not a bug.

### Repository split (2026-09-29)
- At the user's request the work was moved out of neat-dnfs into a new local repository,
  **mo-neat-dnfs** (`C:\dev-files\mo-neat-dnfs`): a clone keeping neat-dnfs history, with no git
  remote. `main` = neat-dnfs v0.3.0 plus one project-level rename commit (folder, CMake project and
  targets, binaries, `config/mo_neat_dnfs.json`, share dir, visualizer entry point; namespace,
  `NEAT_DNFS_*` names and every output format unchanged). This plan and the sandbox scripts are
  tracked; the papers and probe logs are gitignored.
- The phase branches were restacked onto that `main`: `feat/pareto-core` → `feat/selection-config`
  → `feat/objectives-output` → `feat/viz-pareto`.
- Review feedback from neat-dnfs PRs #117–#119, addressed on the branch each belongs to:
  - CodeRabbit had no actionable comments on #117 or #118, and did not review #119 (quota). Its only
    flag was a docstring-coverage warning (7.7 % and 8.1 % against its 80 % threshold, counting .cpp
    definitions and test helpers). Every new internal helper, private member and test fixture is now
    documented. Public API was already complete in the headers, which is where the repo's
    convention puts it.
  - Codecov on #119: 96.1 % patch coverage. The one untested *line* was the ERROR branch taken when
    `objectives.jsonl` cannot be opened; a test now covers it. The other listed lines are inside
    multi-line JSON initialisers that the existing test executes and asserts on.
  - CI on the stacked branches (dispatched manually) was green, including the Linux slow lane and
    both sanitizers. #117's only failure was the known TSan flake
    (`.claude/notes/tsan-random-initial-topology-flake.md`).
- The neat-dnfs PRs #117–#119 were then closed and their branches deleted; neat-dnfs is back to
  plain `main`.
