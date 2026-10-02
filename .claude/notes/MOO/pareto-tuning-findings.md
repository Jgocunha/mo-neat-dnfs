# Tuning Pareto selection against scalar (2026-10-02)

Follows `phase7-reduced-findings.md`. Code: `feat/pareto-selection-fixes`, after the neat-dnfs #120
sync, so every task runs its retuned fitness, and a run succeeds only when the best solution
clears the target (0.95) on **every** partial. Tuning budget: 5 runs × pop 100 × 50 generations
per arm. Data and logs are in `.claude/temp/tuning*/` (gitignored). With 5 runs per arm, read the
tables as directions; the final comparison (below) uses more runs.

## What each tuning round changed, and why

Every new option is opt-in; with none set, Pareto mode behaves as in Phase 5.

| Round | Variant | Pareto settings | What it tested |
|---|---|---|---|
| 1 | V1–V3 | ranking ε 0 or 0.01, archive ε 0.01, violation tie band 0.01, floor 0.1 or 0 | the three fixes from the reduced run |
| 1b | V4, V5 | archive ε 0 | whether noise-level progress should count |
| 2 | V6, V7 | + `stagnationSignal: fitness` | identical stagnation bookkeeping in both modes |
| 3 | V8, V9 | + `frontTieBreak: fitness` | pushing toward "all objectives high" instead of spread |
| 4 | V10 | + `offspringAllocation: fitness` | identical between-species sharing in both modes |

## Mechanisms found

1. **Stagnation fires far more in Pareto mode, and it culls innovation.** Scalar mode's
   "improved" test (`champion fitness > previous champion fitness`) passes about half the time on
   re-evaluation noise alone, so its stagnation almost never fires: on XOR, 0 species-stagnation
   events. Any noise-robust Pareto signal is stricter: 276 events with the per-species fronts
   (V1), 235 even with archive ε 0 (V4), because on XOR every violation sits at exactly 0.1. The
   resulting offspring reassignment wiped out the species carrying new hidden fields: V1's XOR
   runs never grew past the minimal 3 fields and 2 connections. With `stagnationSignal: fitness`
   (V6) XOR had 3 population and 6 species events, against scalar's 5 and 1, and one run grew 5
   fields and reached a real generalist (lowest partial 0.66).
2. **Crowding spreads the population; the success criterion is a corner.** On
   selection-instability (pure parameter tuning, minimal topology) V6 succeeded 0/5 against
   scalar's 4/5 (p = 0.016). NSGA-II's crowding tie-break favours a front's extremes; success
   needs every partial high at once. `frontTieBreak: fitness` (V8) raised it to 2/5.
3. **The rank-derived offspring share is steep.** `(fronts - rank) / fronts` spans (0, 1], while
   weighted fitness near convergence spans a few hundredths, so Pareto arms concentrated
   offspring in few species: 2.1 against 3.6 active species on selection-instability, 4.3
   against 31.5 on AND. Round 4 tests sharing by weighted fitness.
4. **Fast runs overwrote each other.** Run directories were named to the second; every
   detection-instability run finishes within a second, so arms recorded 3–4 of their 5 runs.
   Fixed: a later run in the same second gets `<timestamp> (2)`.

## Results by round (success = every partial above 0.95)

| Task | scalar | V6 | V7 | V8 | V9 |
|---|---|---|---|---|---|
| AND | 0/5 | 2/5 | 2/5 | 1/5 | 0/5 |
| Detection instability | 3/3 | 4/4 | 4/4 | 4/4 | 5/5 |
| Memory instability | 1/5 | 2/5 | 4/5 | 3/5 | 1/2 |
| Memory trace | 0/5 (best's lowest partial 0.82) | 0/5 (0.00) | 0/5 (0.00) | 0/5 (0.00) | 0/5 (0.00) |
| Selection instability | 4/5 | 0/5 | 2/5 | 2/5 | 1/5 |
| XOR | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |

XOR is unsolved by either mode at this budget: populations hold specialists for "fire on input
1" and "fire on input 2", and the generalist needs an inhibitory hidden field, which the
add-field mutation (probability 0.0005) rarely supplies. Memory trace: scalar's final populations
are all feasible; in 3 of 5 V8 runs none is.

## Round 4 and 5: offspring sharing and ranking epsilon

- V10 (V8 plus `offspringAllocation: fitness`) brought memory trace back to comparable: a real
  generalist (lowest partial 0.91) in 2/5 runs against scalar's 3/5 (p = 0.60).
- Selection instability looked like a Pareto weakness at 5 runs per arm. At 15 runs it is not:
  scalar 5/15, V10 2/15, **V11** (V10 plus dominance ε 0.01) 6/15. ε 0.01 absorbs re-evaluation
  noise in the saturated second objective (f3/f4 near 1), and the fitness tie-break decides
  among the ε-ties. V11 became the shipped `pareto-selection` preset.

## Final comparison (shipped preset against scalar)

Every task but hri-packaging. 12 runs per arm, pop 100, 100 generations, target 0.95 on every
partial, binary at `29c9f06`. The runs are in `mo-neat-dnfs/data/<Task>` and
`mo-neat-dnfs/data/<Task> Pareto`, and the dashboard's Scalar vs Pareto page shows them. "Gens"
is the median generation of success among the runs that succeeded.

| Task | scalar success | Pareto success | Fisher p | best's lowest partial (median) | final best fitness (median) | verdict |
|---|---|---|---|---|---|---|
| AND | 3/12 (gens 30) | **7/12** (gens 51) | 0.21 | 0.920 vs **0.952** | 0.983 vs 0.984 | comparable |
| Detection instability | 12/12 (gens 4) | 12/12 (gens 2) | 1.00 | 0.956 vs 0.957 | 0.980 vs 0.978 | comparable |
| Memory instability | 9/12 (gens 12) | 7/12 (gens 9) | 0.67 | 0.956 vs 0.953 | 0.977 vs 0.978 | comparable |
| Memory trace | 0/12 | 0/12 | 1.00 | 0.000 vs 0.000 | 0.839 vs 0.785 (p = 0.45) | comparable |
| Selection instability | 7/12 (gens 68) | 5/12 (gens 20) | 0.68 | 0.951 vs 0.932 | 0.985 vs 0.981 (p = 0.08) | comparable |
| XOR | 4/12 (gens 53) | 4/12 (gens 82) | 1.00 | 0.170 vs **0.455** | 0.771 vs **0.858** | comparable |

- **No task differs significantly at α = 0.05** (Fisher's exact test on success; Mann–Whitney U on
  the best generalist, the final best fitness and the feasible hypervolume, all p > 0.05).
- Pareto leads on the tasks with several conflicting scenarios: AND (58% against 25% success)
  and XOR (lowest partial 0.46 against 0.17, fitness 0.86 against 0.77). Scalar leads slightly on
  the single-behaviour instability tasks and on memory trace's final fitness. None of the leads
  is significant with 12 runs; 30 runs per arm would be needed to tell them apart.
- Memory trace is unsolved by both at this budget, so neither mode's comparison says much there.

## What the tuned preset is, and is not

With the shipped preset, Pareto mode keeps scalar mode's stagnation rules and between-species
offspring sharing. It differs in which members survive pruning, which member is a species'
champion, which crossover parent is the fitter one, and in the floor that ranks every solution
with a near-zero partial behind every complete one. The front-based signals of §3.3 are still
there, off by default, for further experiments.
