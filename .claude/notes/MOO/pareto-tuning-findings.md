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
