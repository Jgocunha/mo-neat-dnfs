"""Check a proposed objective grouping on a real run.
usage: grouping_check.py RUN_DIR "0,2|1,3,4|5,6,7"
Per sampled generation, for raw partials vs the grouping: front-0 share, #fronts, and the
share of front 0 that is a degenerate specialist (some raw partial < 0.1).
Also prints the pooled correlation matrix of raw partials."""
import sys, re
from pathlib import Path
import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("pp", str(Path(__file__).with_name("pareto_probe.py")))
pp = importlib.util.module_from_spec(spec); spec.loader.exec_module(pp)

run = Path(sys.argv[1]); groups = [[int(i) for i in g.split(",")] for g in sys.argv[2].split("|")]
files = sorted((run / "statistics").glob("generation_*.txt"), key=lambda p: int(re.findall(r"\d+", p.name)[0]))
pooled = []
print(f"{run}  groups={groups}")
print(f"{'gen':>5} | {'raw f0%':>7} {'fronts':>6} {'spec%':>6} | {'grp f0%':>7} {'fronts':>6} {'spec%':>6}")
for p in files:
    A = pp.load_gen(p)
    if A is None: continue
    F = A[:, 1:]; pooled.append(F)
    assert sorted(i for g in groups for i in g) == list(range(F.shape[1])), "grouping must partition partials"
    G = np.column_stack([F[:, g].mean(axis=1) for g in groups])
    row = []
    for X in (F, G):
        fr, _ = pp.fronts(X); f0 = fr[0]
        spec = 100 * np.mean(F[f0].min(axis=1) < 0.1)
        row += [100 * len(f0) / len(X), len(fr), spec]
    g = int(re.findall(r"\d+", p.name)[0])
    if g in (1, 5, 10, 15, 20, 25) or p == files[-1]:
        print(f"{g:>5} | {row[0]:7.1f} {row[1]:6d} {row[2]:6.1f} | {row[3]:7.1f} {row[4]:6d} {row[5]:6.1f}")
print("raw partial correlation (pooled):")
print(np.array2string(np.corrcoef(np.vstack(pooled).T), precision=2, suppress_small=True))
