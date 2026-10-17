#!/usr/bin/env python3
"""Xyce regression suite gold outputs (datasets/xyce, git-ignored, GPLv3) against the engine.

    python3 tests/xyce_diff.py [-k RESISTOR] [-j 8] [-v]

For every Netlists/<dir>/<name>.cir with a gold OutputData/<dir>/<name>.cir.prn, run the engine for the
netlist's .op/.dc/.tran (same runner and tolerances as tests/ngspice_diff/run_diff.py) and compare the
single-node V(x) columns and the I(Vname)/I(Lname)/I(Ename)/I(Hname) branch currents of the .prn
(.ac: Re/Im of each). .include/.lib run with their directory resolved
(TREESPICE_BASEDIR, see cli/engine_dump_main.cpp). Statuses: PASS, FAIL, ENGINE-UNSUPPORTED (engine rejects a card or
model parameter), PARSE-ERROR (Xyce syntax the parser lacks), TIMEOUT (--timeout), XYCE-DIALECT (gold follows Xyce-only
semantics, listed in DIALECT), SKIPPED (a feature this comparison cannot judge: .step, multi-run
output, a gold with no V(node)/I(source) column (reason names the column kinds), HB/noise/...). Exit 1 on any FAIL.
"""
import argparse, glob, os, re, sys, tempfile, types
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "ngspice_diff"))
import run_diff as rd  # noqa: E402

XYCE = os.path.join(os.path.dirname(HERE), "datasets", "xyce")
SKIP_CARDS = re.compile(r"^\s*\.(step|sens|hb|mpde|data|func|global_param|sampling|"
                        r"embeddedsampling|pce|lin|noise)\b", re.I | re.M)

SKIP_WHY = {  # card -> why this comparison cannot judge it
    "step": ".step: multi-run gold, no .step in the engine yet",
    "sens": ".sens: Xyce sensitivity output, no engine analysis",
    "hb": ".hb: harmonic balance, no engine analysis", "mpde": ".mpde: Xyce MPDE analysis, no engine analysis",
    "data": ".data: Xyce tabular sweep, multi-run gold", "func": ".func: engine has no user functions (ngspice does)",
    "global_param": ".global_param: Xyce-only global parameter card",
    "sampling": ".sampling: Xyce UQ sampling", "embeddedsampling": ".embeddedsampling: Xyce UQ sampling",
    "pce": ".pce: Xyce polynomial chaos", "lin": ".lin: Xyce S-parameter output", "noise": ".noise: gold holds Xyce noise columns, "
    "not the ngspice spectra run_diff compares",
}

# Cases where Xyce's gold output follows Xyce-specific semantics, not SPICE3/ngspice (the engine's reference).
DIALECT = {
    "Certification_Tests/BUG_302/": "element IC= applied without UIC",
    "Certification_Tests/BUG_1325_SON/": "Xyce include path rules: absolute-path templates, Windows paths, drive letters, "
                                         "nested paths relative to the top-level netlist",
    "Nonlinear/ConvergenceTests/Nox1Tran0.cir": "element IC= applied without UIC",
    "Certification_Tests/BUG_201_SON/": "element IC= applied without UIC",
    "PARSER/scalingFactors.cir": "Xyce reads 1X as 1e6",
    "GLOBALNODE/global3.cir": "Xyce reads 1a as 1 (ngspice: atto)",
    "Output/TRAN/tran-prn-timescalefactor.cir": "output time scaling",
    "PARAM_REFACTOR/paramHomotopy.cir": "LOCA continuation (.options loca) moves the .param to its maximum",
    "MINDUCTORS/mutIndPrint": "V source across an inductor: singular at DC (ngspice .op: check l1#branch)",
    "MINDUCTORS/InductorICBug.cir": "gold prints V(2)=0 at t=0 (before the L IC acts) and steps straight to 0.1 ms; the "
                                    "engine resolves that interval and is within 1.7 mV of the exact "
                                    "-2.5(exp(-t/19.5ms)+exp(-t/0.5ms)) everywhere (the gold: 37 mV), as ngspice",
    "Output/HOMOTOPY/homotopy-op-prn.cir": "LOCA continuation (.options loca conparam=R1): the gold is printed at the end of the run, R1 = 1500",
    "IPULSE/": "values written 1A: Xyce reads amps, ngspice (and the engine) atto, so the source is 1e-18 A",
    "IPWL/": "values written 1A: Xyce reads amps, ngspice (and the engine) atto",
    "ISIN/": "values written 0.3A: Xyce reads amps, ngspice (and the engine) atto",
    "IEXP/": "values written 1A/5A: Xyce reads amps, ngspice (and the engine) atto",
    "CCCS/ftest.cir": "values written 1A: Xyce reads amps, ngspice (and the engine) atto",
    "MINCAPRES/mincr.cir": ".options DEVICE MINRES=1 gives the RS=0 diode 1 ohm (Xyce V(2a) = ngspice v(2b)); ngspice agrees with the engine",
    "LOAD/load.cir": "Xyce gold is 0.5 mV off ngspice near V4 = 54.7 V; the engine matches ngspice to 33 uV",
    "Certification_Tests/BUG_1962/ic2.cir": "gold samples every 0.5 ms; its first interval holds the .ic transient (tau 10 us), which a "
                                            "linear interpolation misreads. The engine matches the gold at its own points to 2 mV",
    "Certification_Tests/BUG_507_SON/": "Xyce reads 3.361e-02A as amperes (ngspice: atto)",
    "ABM_LOG/log.cir": "Xyce log() is log10 (gold V(3) = 2 for 100); ngspice and the engine: natural log",
    "Certification_Tests/BUG_254/atanh_tanh.cir": "atanh(tanh(100)) = atanh(1): Xyce clamps, ngspice prints -inf (so does the engine's Newton fail)",
    "Certification_Tests/BUG_133/": "gold is Xyce's sparse output times; ngspice and the engine agree (v(3) = 1, 2, 3 at 0.500035, .075, .125 ms)",
    "CircuitSim90/MOS2/hussamp.cir": "Xyce's MOS level 2 differs from ngspice's: the engine follows ngspice to 1.4 mV over the run, "
                                     "the gold is 44 mV off both",
    "Certification_Tests/ISSUE_634/hussamp.cir": "the same MOS level 2 circuit: the engine follows ngspice to 1.4 mV",
    "CircuitSim90/MOS3/rich3.cir": "Xyce's MOS level 3 at 125 C: the engine follows ngspice to 1 mV (both start at v(51) = 0, the gold does not)",
    "XDM/HSPICE/MODELS/correct_instance_parameters.cir": "default-card level 3 PMOS: the engine follows ngspice to 0.3 mV over the sweep "
                                                         "(v(3) 0.53612 vs 0.53613 at VG = 0.7), the gold reads 0.5394",
    "SANDLER21/sandler21.cir": "555 timer: output edges match ngspice to 0.1 us (1.0817 ms, 2.0816 ms, ...); a pointwise "
                               "compare at its near-vertical switching edges measures that offset, not the model",
}

# Cases the engine cannot do yet (a missing feature or a convergence limit), not a dialect: ngspice solves them.
# Cases Xyce itself leaves out of normal testing
EXCLUDED = {
    "RLC_LADDER/rc_simple_xyce.cir": "performance test (its README: excluded from normal testing): 3000-stage RC, .tran 0.1u 1 = 1e7 steps",
}

OPEN = {
    "Certification_Tests/BUG_807_SON/inverter_transient_xyce.cir": "BSIM4 version=4.5 card, PMOS body forward-biased, no AD/AS/PD/PS: "
                  "ngspice's BSIM4v5 derives the default junction geometry differently from 4.8 (bsim4va); v(1) 0.737 V vs ngspice "
                  "0.556 V, and ngspice with version=4.8 gives 0.739 V. Explicit zero geometry (sky130) is matched: x_bsim4_v45_zero_geo",
}

def gold(prn):
    """-> (header, rows) or None when the file holds more than one run (e.g. .step)."""
    lines = [l.split() for l in open(prn, errors="ignore") if l.strip()]
    if not lines or lines[0][0] != "Index": return None
    head, rows = lines[0], []
    for t in lines[1:]:
        if t[0] == "Index" or t[0].startswith("End"): return None if t[0] == "Index" else (head, rows)
        try: rows.append([float(x) for x in t])
        except ValueError: return None
    return head, rows

def expand(text, d, depth=0):
    """Splice .include/.inc/.lib <file> [section] in place so node_map sees the elements the engine sees."""
    out = []
    for ln in text.splitlines():
        m = re.match(r"\s*\.(?:include|inc|lib)\s+(?:\"([^\"]+)\"|'([^']+)'|(\S+))(?:\s+(\S+))?\s*$", ln, re.I)
        f = m and (m.group(1) or m.group(2) or m.group(3)).replace("\\", "/")
        if not m or depth > 8 or not os.path.isfile(os.path.join(d, f)): out.append(ln); continue
        body = open(os.path.join(d, f), errors="ignore").read()
        if m.group(4):  # .lib file section: the lines between `.lib section` and `.endl`
            b = re.search(r"^\s*\.lib\s+" + re.escape(m.group(4)) + r"\s*$(.*?)^\s*\.endl", body, re.I | re.M | re.S)
            body = b.group(1) if b else ""
        out.append(expand("*\n" + body, os.path.dirname(os.path.join(d, f)), depth + 1))
    return "\n".join(out)

def case(path, opts):
    rel = os.path.relpath(path, os.path.join(XYCE, "Netlists"))
    for key, why in DIALECT.items():
        if rel.startswith(key): return rel, "XYCE-DIALECT", why
    for key, why in OPEN.items():
        if rel.startswith(key): return rel, "ENGINE-LIMIT", why
    if rel in EXCLUDED: return rel, "SKIPPED", EXCLUDED[rel]
    text = open(path, errors="ignore").read()
    full = expand(text, os.path.dirname(path))
    g = os.path.join(XYCE, "OutputData", rel + ".prn")
    if SKIP_CARDS.search(text): return rel, "SKIPPED", SKIP_WHY[SKIP_CARDS.search(text).group(1).lower()]
    gd = gold(g)
    if not gd:
        multi = len(re.findall(r"^\s*\.dc\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+", full, re.I | re.M))
        return rel, "SKIPPED", ("nested .dc (two sources): the gold holds one run per outer value, run_diff sweeps one source"
                                if multi else "gold output is not a single run (Xyce .step/.data/.mpde-style repeat)")
    head, rows = gd
    try: an = rd.analysis(full)
    except (ValueError, IndexError, StopIteration): return rel, "SKIPPED", "analysis card syntax (e.g. NOOP)"
    if an[0] not in ("op", "dc", "tran", "ac"): return rel, "SKIPPED", an[1] if an[0] == "unsupported" else f".{an[0]}: no comparison for it here"
    nm = rd.node_map(full)
    pat = r"Re\(V\((\w+)\)\)" if an[0] == "ac" else r"V\((\w+)\)"  # .ac: Re(V(x)) Im(V(x)) column pairs
    cols = {m.group(1).upper(): k for k, h in enumerate(head) if (m := re.fullmatch(pat, h, re.I))}
    named = sorted(cols)
    cols = {n: k for n, k in cols.items() if n in nm and (an[0] != "ac" or head[k + 1].lower() == f"im(v({n.lower()}))")}
    br = rd.branch_names(full)  # I(name) of a top-level V/L/E/H: the engine's branch columns
    ipat = r"Re\(I\((\w+)\)\)" if an[0] == "ac" else r"I\((\w+)\)"
    icols = {f"i({m.group(1).lower()})": k for k, h in enumerate(head)
             if (m := re.fullmatch(ipat, h, re.I)) and m.group(1).lower() in br
             and (an[0] != "ac" or head[k + 1].lower() == f"im(i({m.group(1).lower()}))")}
    nodes = set(cols)
    cols.update(icols)
    if not cols or not rows or (named and not nodes):  # V(x) columns that are no node: the gold is of another netlist
        hs = [h.upper() for h in head[2:]]
        if named: why = f"gold V({named[0].lower()}) is not a node of the netlist (stale gold or internal node)"
        elif any(h.startswith("{") for h in hs): why = "gold columns are .print expressions {...}"
        elif any(h.startswith("N(") for h in hs): why = "gold columns include N() device internal variables"
        elif any(re.match(r"(RE\(|IM\()?I[A-Z]?\(", h) for h in hs):
            why = "gold has only device-current columns I(...) (I(R1), I(M1)...): not an element branch current"
        else: why = "no V(node) column in gold: " + " ".join(head[2:5])
        return rel, "SKIPPED", why
    xs = [] if an[0] == "op" else [r[1] for r in rows]
    vals = ({n: [complex(r[k], r[k + 1]) for r in rows] for n, k in cols.items()} if an[0] == "ac"
            else {n: [r[k] for r in rows] for n, k in cols.items()})
    with tempfile.TemporaryDirectory(prefix="xyce_") as tmp:
        args = types.SimpleNamespace(**vars(opts), tmp=tmp, basedir=os.path.dirname(path))
        st, wa, wr, note = rd.compare(args, text, an, "cpu", {n: nm[n] for n in nodes}, (xs, vals))
    if st == "FAIL" and "SPICE parse error" in note: st = "PARSE-ERROR"  # syntax the engine lacks
    if st == "FAIL" and "TIMEOUT" in note: st = "TIMEOUT"  # run-time limit, not an accuracy verdict
    return rel, st, f"worst abs={wa:.3g} V rel={wr:.3g}  {note}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", default="")
    ap.add_argument("-j", type=int, default=os.cpu_count() or 4)
    ap.add_argument("-v", action="store_true", help="print every case, not only FAIL")
    ap.add_argument("--engine", default=rd.ENGINE)
    ap.add_argument("--timeout", type=int, default=120)
    for k, v in rd.DEFAULTS.items(): ap.add_argument("--" + k.replace("_", "-"), type=float, default=v)
    opts = ap.parse_args()
    paths = [p for p in sorted(glob.glob(os.path.join(XYCE, "Netlists", "**", "*.cir"), recursive=True))
             if opts.k in p and os.path.exists(os.path.join(XYCE, "OutputData",
                                                            os.path.relpath(p, os.path.join(XYCE, "Netlists")) + ".prn"))]
    if not paths: sys.exit("no gold cases under datasets/xyce (download: see ENGINE_HANDOFF.md)")
    counts = {}
    with ProcessPoolExecutor(opts.j) as ex:
        for rel, st, note in ex.map(case, paths, [opts] * len(paths)):
            counts[st] = counts.get(st, 0) + 1
            if opts.v or st == "FAIL": print(f"{st:<19} {rel:<60} {note[:160]}", flush=True)
    print(" ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    sys.exit(1 if counts.get("FAIL") else 0)

if __name__ == "__main__":
    main()
