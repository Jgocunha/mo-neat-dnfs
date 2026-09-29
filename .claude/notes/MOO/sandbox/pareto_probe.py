"""Probe existing neat-dnfs run data: per generation, how many individuals are
non-dominated on the partial-fitness vector, how many fronts NSGA sorting gives,
and how correlated the objectives are. Answers: is plain NSGA-II viable on the raw
partials, or does the objective count collapse selection pressure?"""
import re, sys, itertools
from pathlib import Path
import numpy as np

PART_RE = re.compile(r"fit\.\:\s*([0-9eE\.\+\-]+).*?part\.\:\s*\(([^)]*)\)")
GENOME_RE = re.compile(r"genome\s*\(\s*(\d+)\s*,\s*(\d+)")

def load_gen(path):
    rows = []
    for line in path.open():
        m = PART_RE.search(line)
        if not m: continue
        parts = [float(t) for t in m.group(2).split(",") if t.strip()]
        rows.append([float(m.group(1))] + parts)
    return np.array(rows) if rows else None

EPS = 0.0
def dominates(a, b):
    return np.all(a >= b - EPS) and np.any(a > b + EPS)

def fronts(F):
    n = len(F); S = [[] for _ in range(n)]; cnt = np.zeros(n, int); rank = np.zeros(n, int)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if dominates(F[i], F[j]): S[i].append(j)
            elif dominates(F[j], F[i]): cnt[i] += 1
    cur = [i for i in range(n) if cnt[i] == 0]; k = 0; out = []
    while cur:
        out.append(cur); nxt = []
        for i in cur:
            rank[i] = k
            for j in S[i]:
                cnt[j] -= 1
                if cnt[j] == 0: nxt.append(j)
        cur = nxt; k += 1
    return out, rank

def probe(run):
    stats = Path(run) / "statistics"
    files = sorted(stats.glob("generation_*.txt"), key=lambda p: int(re.findall(r"\d+", p.name)[0]))
    print(f"\n== {run}")
    allF = []
    for p in files:
        A = load_gen(p)
        if A is None: continue
        fit, F = A[:, 0], A[:, 1:]
        U = np.unique(F.round(4), axis=0)
        fr, rank = fronts(F)
        best = np.argmax(fit)
        print(f"{p.name:>20}: n={len(F):4d} m={F.shape[1]} uniq={len(U):4d} "
              f"front0={len(fr[0]):4d} ({100*len(fr[0])/len(F):5.1f}%) nfronts={len(fr):3d} "
              f"bestScalarRank={rank[best]}")
        allF.append(F)
    if allF:
        F = np.vstack(allF)
        C = np.corrcoef(F.T)
        print("objective corr (all gens pooled):")
        print(np.array2string(C, precision=2, suppress_small=True))

if __name__ == "__main__":
    EPS = float(sys.argv[1])
    print("EPS", EPS)
    for r in sys.argv[2:]:
        probe(r)
