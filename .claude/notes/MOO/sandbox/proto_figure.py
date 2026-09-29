"""Prototype of the proposed visualizer Pareto views, on a real AND run."""
import sys, re
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import importlib.util
spec = importlib.util.spec_from_file_location("pp", str(Path(__file__).with_name("pareto_probe.py"))); pp = importlib.util.module_from_spec(spec); spec.loader.exec_module(pp)
run = Path(sys.argv[1]); out = Path(__file__).with_name("proto_pareto_views.png")
files = sorted((run/"statistics").glob("generation_*.txt"), key=lambda p: int(re.findall(r"\d+", p.name)[0]))
groups = [[0,2],[1,3,4],[5,6,7]]; gnames = ["input repr.", "output (AND)", "return to rest"]
gens = [pp.load_gen(p) for p in files]
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
cm = plt.cm.viridis
for k, idx in enumerate([0, 6, 12, 18, 24]):
    A = gens[idx]; F = A[:,1:]; G = np.column_stack([F[:, g].mean(axis=1) for g in groups])
    fr, rank = pp.fronts(G); f0 = G[fr[0]]
    c = cm(k/4)
    if idx == 24: ax[0].scatter(G[:,1], G[:,2], s=6, color="0.8", label="gen 25 population")
    o = np.argsort(-f0[:,1]); ax[0].plot(f0[o,1], f0[o,2], "o-", ms=3, color=c, lw=1, label=f"gen {idx+1} front 0 (3-obj)")
ax[0].set_xlabel(gnames[1]); ax[0].set_ylabel(gnames[2]); ax[0].set_title("Front evolution (2-D projection of 3-obj front)"); ax[0].legend(fontsize=7)
A = gens[-1]; fit, F = A[:,0], A[:,1:]; fr, rank = pp.fronts(F); f0 = F[fr[0]]
best = np.argmax(fit)
for row in f0: ax[1].plot(range(1,9), row, color="tab:blue", alpha=0.15)
ax[1].plot(range(1,9), F[best], color="tab:red", lw=2, label="best weighted-sum individual")
ax[1].set_xticks(range(1,9)); ax[1].set_xticklabels(["f1_1","f1_2","f2_1","f2_2","f3","f4_1","f4_2","f4_3"])
ax[1].set_title(f"Parallel coordinates, gen 25 front 0 (8 obj, n={len(f0)})"); ax[1].legend(fontsize=7); ax[1].set_ylim(0,1.05)
sizes = {"8 raw objectives": [], "8 raw, eps=0.02": [], "3 grouped objectives": []}
for A in gens:
    F = A[:,1:]; n = len(F)
    pp.EPS = 0.0; sizes["8 raw objectives"].append(100*len(pp.fronts(F)[0][0])/n)
    pp.EPS = 0.02; sizes["8 raw, eps=0.02"].append(100*len(pp.fronts(F)[0][0])/n)
    pp.EPS = 0.0; G = np.column_stack([F[:, g].mean(axis=1) for g in groups]); sizes["3 grouped objectives"].append(100*len(pp.fronts(G)[0][0])/n)
for k, v in sizes.items(): ax[2].plot(range(1, len(v)+1), v, label=k)
ax[2].set_xlabel("generation"); ax[2].set_ylabel("% of population non-dominated"); ax[2].set_title("Selection pressure: size of front 0"); ax[2].legend(fontsize=8)
fig.suptitle(f"Prototype Pareto views -- AND task, pop 300, scalar-selection run ({run.name})")
fig.tight_layout(); fig.savefig(out, dpi=110); print(out)
