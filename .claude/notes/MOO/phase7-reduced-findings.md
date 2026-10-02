# Reduced Phase 7 run: findings (2026-10-02)

A reduced version of PLAN.md's Phase 7 protocol, run on an instrumented scratch build to see what
Pareto selection does at each stage of a generation. **n = 5 runs per arm: read these as
mechanisms and directions, not as statistically settled results.**

## Setup

- Code: `feat/pareto-front-per-generation` (PR #10) plus scratch `[moo]` INFO trace lines (branch
  `scratch/moo-instrumentation`, never merged). The trace covers ranking, offspring allocation
  per species, pruning, crossover parent choice and elitism, in both modes.
- Budget: `--runs 5 --pop 200 --gens 60 --target 0.95`. Arms: `and`, `xor` and
  `selection-instability` under scalar and `pareto-selection` (ε 0.01, floor 0.1), plus
  `pareto-selection-no-floor` on `and` and `xor`. The 8 arms took 24 minutes on 12 cores.
- Data, logs and `analyze.py` are in `.claude/temp/phase7-reduced/` (gitignored). Point the
  dashboard's data root there to browse the runs.
- Post-hoc metrics use the task's grouped objectives with ε 0.01 and floor 0.1 for every arm, so
  the arms are comparable. A "specialist" has some raw partial below 0.1.

## Outcomes

| Arm | reached 0.95 | final best (median) | best's lowest partial (median) | best is a specialist | final feasible hypervolume (median) |
|---|---|---|---|---|---|
| AND scalar | 5/5 (gens 3–51, median 10) | 0.966 | 0.81 | 0/5 | 0.88 |
| AND Pareto | 5/5 (gens 9–28, median 12) | 0.958 | 0.83 | 0/5 | 0.87 |
| AND Pareto, no floor | 5/5 (gens 3–13, median 5) | 0.966 | 0.91 | 0/5 | 0.90 |
| XOR scalar | 0/5 | 0.744 | 0.00 | 4/5 | 0.00 |
| XOR Pareto | 1/5 | 0.892 | 0.79 | 0/5 | 0.86 |
| XOR Pareto, no floor | 2/5 | 0.790 | 0.22 | 2/5 | 0.72 |
| Selection instability scalar | 5/5 | 0.975 | 0.92 | 0/5 | 0.95 |
| Selection instability Pareto | 3/5 | 0.955 | 0.92 | 0/5 | 0.91 |

- **XOR is where Pareto helps.** Scalar selection leaves 4/5 runs with a best solution that
  scores 0 on one scenario. Pareto with the floor leaves none, and its median best fitness is
  0.89 against 0.74.
- **AND is too easy at 0.95** to separate the arms. One scalar run "succeeded" with a partial at
  0.33, which is the plan's point that weighted-sum success can hide a failed behaviour.
- **Selection instability regresses under Pareto**: 2/5 runs plateau at fitness 0.804 from about
  generation 20 to 60 (mechanism 3).

## Mechanisms the trace exposed

1. **Species stagnation fires far more often in Pareto mode.** Species-stagnation warnings:
   XOR 546 vs 9, selection instability 150 vs 0, AND 22 vs 0. In Pareto mode a species "improves"
   only when one of its members enters the *global* archive. Once the archive settles (often on
   a single point) almost no species ever improves again, so stagnation takes their offspring
   and gives them to the top species. Scalar mode compares each species with *its own* best,
   which is far easier to meet. The two signals are not equivalent.
2. **While nothing is feasible, selection runs on noise.** Deb's rule compares two infeasible
   solutions only by total violation. On XOR almost every solution has one partial at about 0, so
   violations are about 0.100: exact ties early (front 0 held 102 of 200) and noise-level
   differences later (151 fronts in 200 individuals). The objectives play no part. In the stuck run, 40–45% of crossovers between distinct parents took
   the lower-fitness parent as fitter (17% across all XOR Pareto runs), and the archive-empty improvement
   fallback (violation must drop by more than ε) never fires. Even so, the floor arm beat the
   no-floor arm on XOR overall, so the floor is worth keeping; the infeasible comparison is the
   part to fix.
3. **ε-ties bloat front 0 near convergence.** In the plateaued selection-instability runs the
   archive held one point, and 59–77 of 200 solutions (about 37%) were on front 0. Within ε = 0.01
   they are mutually non-dominated, and inside a front only crowding orders them, so pruning
   and crossover reward spread rather than quality. 77% of crossovers between distinct parents took the
   random-inheritance branch (equivalent parents). The Phase 5 AND run showed the same front-0
   growth (20%).
4. **The reported best is often not a front-0 solution.** With the floor, the generation's
   highest weighted-sum solution was on front 0 in only 52% (XOR) to 84% (selection instability)
   of generations; the rest of the time it was an infeasible specialist. Under D2/D4 that is what
   Pareto mode reports as `bestSolution` and preserves by elitism (one slot; elitism had to evict
   a solution to restore it 1.5–2.3× as often as in scalar mode).
5. **Pruning works as designed.** In every Pareto arm, pruned members sit 13–26 fronts below the
   kept ones on average. In scalar arms the kept members are about 0.1 fitter.
6. **Pre-existing, not from this work:**
   - **Species explode and collapse in a sawtooth.** With the fixed `compatibilityThreshold` 3.5,
     XOR scalar reached 119 species in a population of 200 as genomes grew. Population
     stagnation then gave every offspring to the top two species and the count collapsed.
   - **Generation numbering is off by one between output files.** `solutions/gen N/` and
     `statistics/generation_N.txt` hold the population that `objectives.jsonl`,
     `overview.jsonl` (and `pareto_front/gen N/`, `generationFound`) call generation N−1. The
     reason is that `upkeep()` increments the generation before `savePerGenerationData()`.

## Candidate changes (each needs a decision; none is implemented)

- **(1)** Make Pareto species improvement per-species: a species improves when a member enters
  that species' own non-dominated history (a per-species archive), mirroring scalar mode's "beats
  its own best".
- **(2)** Treat violations within ε as equal, and compare such infeasible pairs by objective
  dominance (constrained domination with an ε-tie).
- **(3)** Rank with a smaller ε (or ε = 0) and keep ε 0.01 only for the archive's noise absorption.
  Rank ε is a preset value, so this is a config change for the experiment, not code.
- Re-run this reduced protocol after each change, then the full Phase 7 protocol.
