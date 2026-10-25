#!/usr/bin/env python3
"""IBM power-grid benchmarks (Nassif, ASP-DAC 2008) against their published solutions.

    python3 tests/ibmpg_diff.py [-k ibmpg3] [--mode cpu] [--atol 1e-4]

Reads datasets/ibmpg/ibmpgN.{spice,solution}.bz2 (git-ignored; see ENGINE_HANDOFF.md), runs the engine's
DC operating point (build/acutesim_engine_dump) and compares every node of the solution
file. The solutions print 6 significant digits, so the default tolerance is 100 uV absolute. Exit 1 on FAIL.
"""
import argparse, bz2, glob, os, re, resource, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests", "ngspice_diff"))
import run_diff as rd  # noqa: E402  (shared engine-path default)
GROUND = {"0", "GND"}  # ngspice: vss / ground are ordinary nodes

def node_order(netlist):
    """Engine node numbering: first appearance in R/V/I cards (the benchmarks contain only those)."""
    order = {}
    for line in netlist.splitlines()[1:]:
        t = line.split()
        if len(t) < 3 or t[0][0] in "*.": continue
        for name in (t[1].upper(), t[2].upper()):
            if name not in GROUND: order.setdefault(name, len(order))
    return order

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", default="")
    ap.add_argument("--mode", default="cpu")
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--engine", default=rd.ENGINE)
    args = ap.parse_args()
    cases = sorted(glob.glob(os.path.join(ROOT, "datasets/ibmpg/ibmpg*.spice.bz2")),
                   key=lambda p: int(re.search(r"ibmpg(\d+)", p).group(1)))
    if not cases: sys.exit("no datasets/ibmpg/*.spice.bz2 (download: see ENGINE_HANDOFF.md)")
    fails = 0
    with tempfile.TemporaryDirectory(prefix="ibmpg_") as tmp:
        for path in cases:
            name = os.path.basename(path).split(".")[0]
            if args.k not in name: continue
            text = bz2.open(path, "rt").read()
            ref = {}
            for line in bz2.open(path.replace(".spice.", ".solution."), "rt"):
                t = line.split()
                if len(t) == 2: ref[t[0].upper()] = float(t[1])
            cir = os.path.join(tmp, name + ".spice")
            open(cir, "w").write(text)
            t0 = time.time()
            out = subprocess.run([args.engine, cir, args.mode, "dc"], capture_output=True, text=True).stdout
            secs = time.time() - t0
            mem = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20  # macOS: bytes -> MB
            rows = [l for l in out.splitlines() if l[:1].isdigit() or l[:1] == "-"]
            conv = "CONVERGED 1" in out
            order = node_order(text)
            worst, where, missing = 0.0, "", 0
            if conv and rows:
                v = [float(x) for x in rows[-1].split()[1:]]
                for node, exp in ref.items():
                    k = order.get(node)
                    if k is None and exp == 0.0: continue  # the solution lists ground ("G") at 0 V
                    if k is None or k >= len(v): missing += 1; continue
                    if abs(v[k] - exp) > worst: worst, where = abs(v[k] - exp), node
            ok = conv and rows and missing == 0 and worst <= args.atol
            fails += not ok
            print(f"{'PASS' if ok else 'FAIL':<5} {name:<8} nodes={len(order):>8} compared={len(ref) - missing:>8} "
                  f"worst={worst:.3g} V at {where or '-'}  {secs:7.1f} s  peak RSS<={mem:,.0f} MB"
                  + ("" if conv else "  NOT CONVERGED") + (f"  missing={missing}" if missing else ""), flush=True)
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
