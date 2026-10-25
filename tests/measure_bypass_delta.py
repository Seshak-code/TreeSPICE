#!/usr/bin/env python3
"""Measure device bypass ON vs OFF quantitative delta and bound compliance."""
import glob, math, os, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DIFF_DIR = os.path.join(HERE, "ngspice_diff")
sys.path.insert(0, DIFF_DIR)

from run_diff import analysis, node_map, run_ngspice, interp, num, cards

ENGINE = os.environ.get("TREESPICE_DUMP", os.path.join(ROOT, "build", "acutesim_engine_dump"))  # env: another build
OSDI_DIR = os.path.abspath(os.path.join(ROOT, "datasets", "osdi"))

cases = [
    "qa4_bypass_comp.cir",
    "qa4_bypass_currmirror.cir",
    "qa4_bypass_diffpair.cir",
    "qa4_bypass_ring.cir",
    "qa4_bypass_opamp_openloop.cir",
    "qa4_bypass_comp_trippoint.cir",
]

class Args:
    pass

args = Args()
args.tmp = tempfile.mkdtemp(prefix="bypass_meas_")
args.engine = ENGINE
args.ngspice = "ngspice"
args.timeout = 300

shutil.copytree(os.path.join(DIFF_DIR, "netlists", "lib"), os.path.join(args.tmp, "lib"))
pdk = os.path.join(ROOT, "datasets", "pdk")
if os.path.isdir(pdk):
    os.symlink(pdk, os.path.join(args.tmp, "pdk"))

results = []
env_base = dict(os.environ, TREESPICE_OSDI_PATH=OSDI_DIR)

for fname in cases:
    path = os.path.join(DIFF_DIR, "netlists", fname)
    text = open(path).read()
    an = analysis(text)
    nm = node_map(text)
    names = list(nm)

    # Reference run from ngspice
    ref_xs, ref_dict = run_ngspice(args, text, an, names)

    # Run engine ON
    eng_cir = os.path.join(args.tmp, "eng.cir")
    open(eng_cir, "w").write(text)
    cmd = [args.engine, eng_cir, "cpu", "tran", repr(an[1]), repr(an[2])]
    p_on = subprocess.run(cmd, cwd=args.tmp, capture_output=True, text=True, env=env_base)
    p_off = subprocess.run(cmd, cwd=args.tmp, capture_output=True, text=True, env=dict(env_base, TREESPICE_BYPASS="0"))

    pts_on = [[float(x) for x in l.split()] for l in p_on.stdout.splitlines() if l and (l[0].isdigit() or l[0] == "-")]
    pts_off = [[float(x) for x in l.split()] for l in p_off.stdout.splitlines() if l and (l[0].isdigit() or l[0] == "-")]

    assert len(pts_on) > 0 and len(pts_off) > 0, f"No points for {fname}"

    worst_on_off_abs = 0.0
    worst_on_off_rel = 0.0
    bound_violation = False
    max_bound_ratio = 0.0

    worst_on_ng_abs = 0.0
    worst_off_ng_abs = 0.0
    drift_details = ""

    # Check oscillator period/frequency if ring oscillator
    if "ring" in fname:
        # compute zero-crossings of node n1 (Vdd/2 = 0.9V)
        idx_n1 = names.index("N1") + 1
        crossings_on, crossings_off = [], []
        for i in range(1, len(pts_on)):
            t1, v1 = pts_on[i-1][0], pts_on[i-1][idx_n1]
            t2, v2 = pts_on[i][0], pts_on[i][idx_n1]
            if (v1 <= 0.9 and v2 > 0.9):
                crossings_on.append(t1 + (0.9 - v1) / (v2 - v1) * (t2 - t1))
        for i in range(1, len(pts_off)):
            t1, v1 = pts_off[i-1][0], pts_off[i-1][idx_n1]
            t2, v2 = pts_off[i][0], pts_off[i][idx_n1]
            if (v1 <= 0.9 and v2 > 0.9):
                crossings_off.append(t1 + (0.9 - v1) / (v2 - v1) * (t2 - t1))
        periods_on = [crossings_on[j+1] - crossings_on[j] for j in range(len(crossings_on)-1)]
        periods_off = [crossings_off[j+1] - crossings_off[j] for j in range(len(crossings_off)-1)]
        avg_per_on = sum(periods_on)/len(periods_on) if periods_on else 0
        avg_per_off = sum(periods_off)/len(periods_off) if periods_off else 0
        drift_details = f"Ring: {len(periods_on)} periods, Ton={avg_per_on*1e12:.2f}ps, Toff={avg_per_off*1e12:.2f}ps, delta={abs(avg_per_on-avg_per_off)*1e15:.2f}fs"

    for row_on, row_off in zip(pts_on, pts_off):
        t_on = row_on[0]
        for ki, name in enumerate(names):
            v_on = row_on[ki + 1]
            v_off = row_off[ki + 1]
            diff = abs(v_on - v_off)
            rel = diff / max(abs(v_off), 1e-3)
            bound = 1e-3 * abs(v_off) + 1e-6

            if diff > worst_on_off_abs:
                worst_on_off_abs = diff
            if rel > worst_on_off_rel:
                worst_on_off_rel = rel

            ratio = diff / bound
            if ratio > max_bound_ratio:
                max_bound_ratio = ratio
            if diff > bound:
                bound_violation = True

            v_ng = interp(ref_xs, ref_dict[name], t_on)
            diff_on_ng = abs(v_on - v_ng)
            diff_off_ng = abs(v_off - v_ng)
            if diff_on_ng > worst_on_ng_abs: worst_on_ng_abs = diff_on_ng
            if diff_off_ng > worst_off_ng_abs: worst_off_ng_abs = diff_off_ng

    results.append({
        "name": fname,
        "pts": len(pts_on),
        "worst_on_off_abs": worst_on_off_abs,
        "worst_on_off_rel": worst_on_off_rel,
        "max_bound_ratio": max_bound_ratio,
        "bound_violation": bound_violation,
        "worst_on_ng": worst_on_ng_abs,
        "worst_off_ng": worst_off_ng_abs,
        "drift": drift_details,
    })

shutil.rmtree(args.tmp)

print("| Testcase | Pts | Max |V_on - V_off| | Max Rel Diff | Max / Bound | Bound Satisfied? | vs ngspice (ON) | vs ngspice (OFF) |")
print("|---|---|---|---|---|---|---|---|")
for r in results:
    sat = "PASS (<= Bound)" if not r["bound_violation"] else "FAIL (VIOLATION)"
    print(f"| `{r['name']}` | {r['pts']} | {r['worst_on_off_abs']:.3e} V | {r['worst_on_off_rel']:.3e} | {r['max_bound_ratio']:.3f}x | **{sat}** | {r['worst_on_ng']:.3e} V | {r['worst_off_ng']:.3e} V |")

for r in results:
    if r["drift"]:
        print(f"\n{r['drift']}")
sys.exit(1 if any(r["bound_violation"] for r in results) else 0)
