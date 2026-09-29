"""Collapse AND's 8 partials into semantic groups (mean within group) and re-measure
front sizes -- does grouping restore NSGA-II selection pressure without epsilon?"""
import sys, re
from pathlib import Path
import numpy as np
sys.argv = [sys.argv[0], "0"] + sys.argv[1:]
import importlib.util
spec = importlib.util.spec_from_file_location("pp", str(Path(__file__).with_name("pareto_probe.py"))); pp = importlib.util.module_from_spec(spec); spec.loader.exec_module(pp)
GROUPINGS = {
  "3 groups: input{f1_1,f2_1} output{f1_2,f2_2,f3} rest{f4_*}": [[0,2],[1,3,4],[5,6,7]],
  "2 groups: output-behaviour vs input+rest": [[1,3,4],[0,2,5,6,7]],
}
run = Path(sys.argv[2])
files = sorted((run/"statistics").glob("generation_*.txt"), key=lambda p: int(re.findall(r"\d+", p.name)[0]))
for name, groups in GROUPINGS.items():
    print("\n", name)
    for p in files[::4]:
        A = pp.load_gen(p); F = A[:,1:]
        G = np.column_stack([F[:, g].mean(axis=1) for g in groups])
        fr, _ = pp.fronts(G)
        print(f"  {p.name:>18}: front0={len(fr[0]):4d} ({100*len(fr[0])/len(G):5.1f}%) nfronts={len(fr)}")
