"""Deb 2002 constrained-domination probe: a solution is feasible iff every raw partial >= FLOOR.
Violation = sum(max(0, FLOOR - partial)). Ranking: feasible first (Pareto-sorted on grouped
objectives), infeasible after, ordered by violation. Reports feasible share and specialist share
of the resulting front 0.
usage: constraint_check.py RUN_DIR "0,2|1,3,4|5,6,7" FLOOR"""
import sys, re
from pathlib import Path
import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("pp", str(Path(__file__).with_name("pareto_probe.py")))
pp = importlib.util.module_from_spec(spec); spec.loader.exec_module(pp)
run = Path(sys.argv[1]); groups = [[int(i) for i in g.split(",")] for g in sys.argv[2].split("|")]; floor = float(sys.argv[3])
files = sorted((run / "statistics").glob("generation_*.txt"), key=lambda p: int(re.findall(r"\d+", p.name)[0]))
print(f"{run} floor={floor}")
for p in files:
    g = int(re.findall(r"\d+", p.name)[0])
    if g not in (1, 5, 10, 15, 20) and p != files[-1]: continue
    F = pp.load_gen(p)[:, 1:]
    G = np.column_stack([F[:, grp].mean(axis=1) for grp in groups])
    viol = np.clip(floor - F, 0, None).sum(axis=1); feas = viol == 0
    if feas.any():
        idx = np.flatnonzero(feas); fr, _ = pp.fronts(G[idx]); f0 = idx[fr[0]]
    else:
        f0 = np.flatnonzero(viol == viol.min())
    print(f"  gen {g:>3}: feasible {100*feas.mean():5.1f}%  front0 n={len(f0):3d}  specialists in front0 {100*np.mean(F[f0].min(axis=1) < 0.1):5.1f}%")
