#!/usr/bin/env python3
"""Differential test: TreeSPICE (acutesim_engine_dump) vs ngspice on the same SPICE netlists.
Stdlib only. Usage: python3 tests/ngspice_diff/run_diff.py [-k substr] [--modes cpu,gpu] [-v]
Exit status is non-zero if any netlist FAILs (or ngspice itself errors)."""
import argparse, atexit, bisect, glob, itertools, math, os, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
# The standalone TreeSPICE build, else AcuteSim's build when TreeSPICE is its acutesim_engine/ submodule.
ENGINE = next((p for p in (os.path.join(ROOT, "build", "acutesim_engine_dump"),
                           os.path.join(ROOT, "..", "build", "acutesim_engine", "acutesim_engine_dump"))
               if os.access(p, os.X_OK)), os.path.join(ROOT, "build", "acutesim_engine_dump"))
GROUND = {"0", "GND"}  # ngspice: vss / ground are ordinary nodes
NODE_POS = {**{c: [1, 2] for c in "RCLVIDHFB"}, **{c: [1, 2, 3, 4] for c in "MEGT"},
            "Q": [1, 2, 3], "J": [1, 2, 3], "S": [1, 2, 3, 4], "W": [1, 2], "K": []}
SUFFIX = {"t": 1e12, "g": 1e9, "meg": 1e6, "k": 1e3, "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15}

# Tolerance policy (all overridable by flags).
# DC: engine voltage sources are 1 mOhm Norton sources, inductors 1 mOhm shorts, E sources 100 uOhm, so
#   node voltages carry ~I*1mOhm error (<=100 uV at 100 mA); atol=200 uV covers that.  rtol=1e-3 is
#   ngspice's own RELTOL, i.e. the accuracy either solver actually guarantees for nonlinear devices.
# Transient: engine steps are <= the .tran step (coarse, LTE-limited); the reference is run with
#   tmax=tstep/10.  Error is judged against each node's swing (max-min of the reference): 2% of swing,
#   with a 5 mV floor so near-constant nodes (virtual grounds) are not judged on noise, and a timing
#   tolerance of tstep/10: an engine sample may match the reference anywhere within +-tstep/10 of its
#   time. On a steep edge 2% of swing alone is a sub-ps timing demand (a 100-stage chain accumulated
#   0.95 ps = 0.1% of its delay against a reference converged to 0.15 ps; ngspice at its own default
#   step is 3 ps off by stage 10).
# Branch currents (I(V1), I(L1), ...): DC/AC atol 10 nA + rtol 1e-3. Transient: as a node, with 10% of swing or 2% of the
#   peak (1 uA floor); an instant outside that passes when the mean over its .tran step agrees (the engine resolves
#   switching spikes narrower than the reference's steps, with the same charge). A reference current above 1 kA is a diverged run and is not compared.
# Both simulators run at ngspice's default 27 C (the engine's Vt is kT/q at 300.15 K since 2026-10-15).
DEFAULTS = dict(dc_atol=2e-4, dc_rtol=1e-3, tran_rtol=0.02, tran_atol=5e-3, i_atol=1e-8, i_tran_atol=1e-6, i_tran_rtol=0.1)

def num(s):
    m = re.fullmatch(r"([-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?)([a-z]*)", s.lower())
    if not m: raise ValueError(f"bad number {s!r}")
    suf = m.group(2)
    return float(m.group(1)) * (SUFFIX["meg"] if suf.startswith("meg") else SUFFIX.get(suf[:1], 1.0) if suf else 1.0)

def temps(text):
    """The .temp card's list of Celsius values (ngspice itself takes one: each is run on its own)."""
    return next((list(map(num, t[1:])) for t in map(str.split, cards(text)) if t[0].lower() == ".temp"), [])

def steps(text):
    """.step cards (list v.. | start stop incr) of param NAME, temp or a V/I/R element, as (kind, name, values).
    ngspice 47 has no .step: each combination (first card outermost, as the engine) is run on its own."""
    out = []
    for w in (t[1:] for t in map(str.split, cards(text)) if t[0].lower() == ".step"):
        kind = w[0].lower() if w[0].lower() in ("param", "temp") else "elem"
        name, rest = (w[1], w[2:]) if kind == "param" else (w[0], w[1:])
        vals = (list(map(num, rest[1:])) if rest[0].lower() == "list" else
                [num(rest[0]) + k * num(rest[2]) for k in range(int(math.floor((num(rest[1]) - num(rest[0])) / num(rest[2]) + 1e-9)) + 1)])
        out.append((kind, name, vals))
    return out

def step_text(lines, combo):
    """The netlist lines (after the title) with one .step combination applied; a temp step is returned apart."""
    lines, temp = list(lines), None
    for (kind, name, _), v in combo:
        if kind == "param": lines.append(f".param {name}={v!r}")  # the last .param wins
        elif kind == "temp": temp = v
        else:
            i = next(i for i, l in enumerate(lines) if l.split()[:1] and l.split()[0].lower() == name.lower())
            w = lines[i].split()
            k = next((j + 1 for j, x in enumerate(w) if x.lower() == "dc"), 3)  # V/I: after DC, else the first value; R: the value
            w[k] = repr(v)
            lines[i] = " ".join(w)
    return lines, temp

def toks(line): return re.sub(r"[(),=]", " ", line).split()

def cards(text):
    """Yield (index, stripped line) of non-comment lines; '+' continuations are joined to the previous."""
    out = []
    for i, raw in enumerate(text.splitlines()):
        line = raw.split(";")[0].strip()
        if i == 0 or not line or line[0] == "*": continue
        if line[0] == "+" and out: out[-1] += " " + line[1:]
        else: out.append(line)
    return out

# OSDI libraries (.osdi <file>): compiled Verilog-A models, git-ignored; build them with tests/build_osdi.sh
OSDI_DIR = os.environ.get("TREESPICE_OSDI_PATH") or os.path.join(ROOT, "datasets", "osdi")
os.environ.setdefault("TREESPICE_OSDI_PATH", OSDI_DIR)
def osdi_libs(text): return re.findall(r"^\s*\.osdi\s+(\S+)", text, re.I | re.M)

def with_includes(text, base, depth=0):
    """The netlist and the .include / .lib files it pulls in (they may need OSDI models or a PDK too)."""
    out = [text]
    for f in re.findall(r"^\s*\.(?:include|inc|lib)\s+[\"']?([^\s\"']+)", text, re.I | re.M):
        p = os.path.normpath(os.path.join(base, f))
        if depth < 3 and os.path.isfile(p): out.append(with_includes(open(p, errors="ignore").read(), os.path.dirname(p), depth + 1))
    return "\n".join(out)

def needed_osdi(text):
    """OSDI libraries a netlist needs: its .osdi cards, and the ones the native BSIM cards run on."""
    libs = set(osdi_libs(text))
    for lvl in re.findall(r"^\s*\.model\s+\S+\s+\w+\s*\(?[^\n]*?\blevel\s*=\s*(\d+)", text, re.I | re.M):
        libs |= {"54": {"bsim4.osdi"}, "14": {"bsim4.osdi"}, "49": {"bsim3.osdi"}, "8": {"bsim3.osdi"},
                 "10": {"bsimsoi.osdi"}, "58": {"bsimsoi.osdi"}}.get(lvl, set())
    return sorted(libs)

def node_map(text):
    """Engine column order: first appearance in top-level element cards (name -> 1-based column)."""
    m, depth = {}, 0  # depth: open .subckt definitions (they nest)
    models = {toks(l)[1].upper() for l in cards(text) if l.lower().startswith(".model")}
    for line in cards(text):
        t = toks(line)
        if t[0][0] == ".":
            kw = t[0].upper()
            if kw == ".END": break
            depth += (kw == ".SUBCKT") - (kw == ".ENDS" and depth > 0)
            continue
        if depth: continue
        c = t[0][0].upper()
        pos = range(1, len(t) - 1) if c == "X" else NODE_POS.get(c, [])
        if c == "X":  # X<name> <nodes...> <subckt> [params:] [k=v ...]: the words before the parameters
            w = re.sub(r"\s*=\s*", "=", line).split()
            w = w[:next((k for k, x in enumerate(w) if "=" in x or x.upper() == "PARAMS:"), len(w))]
            pos = range(1, len(w) - 1)
        if c in "EGFH" and len(t) > 3 and t[3].upper() in ("POLY", "VALUE", "VOL", "CUR", "TABLE"):
            pos = [1, 2]  # behavioural forms: the control nodes are not at the linear positions
        if c == "N":  # N<name> <terminals...> <model> [params]: terminals precede the first model name
            pos = range(1, next((k for k in range(1, len(t)) if t[k].upper() in models), 1))
        # Q c b e [s] model ...: the 4th field is a substrate node when the 5th is a name (engine rule)
        if c == "Q" and len(t) >= 6 and not re.fullmatch(r"[-+.\d].*", t[5]) and t[5].upper() not in ("AREA", "OFF", "IC", "M", "TEMP", "DTEMP"):
            pos = [1, 2, 3, 4]
        for p in pos:
            if p < len(t) and t[p].upper() not in GROUND: m.setdefault(t[p].upper(), len(m) + 1)
    return m

def branch_names(text):
    """Top-level branch-current elements (ngspice name#branch): V, L, linear E, H and B voltage sources."""
    out, in_sub = [], False
    for line in cards(text):
        t = toks(line)
        if t[0][0] == ".":
            kw = t[0].upper()
            in_sub = kw == ".SUBCKT" or (in_sub and kw != ".ENDS")
            continue
        poly = len(t) > 3 and t[3].upper() in ("POLY", "VALUE", "VOL", "CUR", "TABLE")  # ngspice rebuilds these: no e1#branch
        if not in_sub and not (poly and t[0][0].upper() in "EH") and (t[0][0].upper() in "VLEH" or (t[0][0].upper() == "B" and re.search(r"\bv\s*=", line, re.I))):
            out.append(t[0].lower())
    return out

def analysis(text):
    """-> ('op',) | ('dc', src, start, stop, step[, src2, start2, ...]) | ('tran', tstep, tstop, uic) | ('ac', sweep, n, f1, f2)
       | ('noise', "v(out[,ref])", src, sweep, n, f1, f2) | ('pz', in+, in-, out+, out-, cur|vol, pol|zer|pz) | ('sens', "v(out[,ref])" | "i(vsrc)") | ('unsupported', why)"""
    for line in cards(text):
        t = line.split()
        kw = t[0].lower()
        if kw == ".ac": return ("ac", t[1].lower(), int(num(t[2])), num(t[3]), num(t[4]))
        if kw == ".sens": return ("sens", t[1].lower()) if len(t) == 2 else ("unsupported", ".sens form (AC / several outputs)")
        if kw == ".pz": return ("pz", *t[1:7])
        if kw == ".noise":
            k = next(i for i, x in enumerate(t) if x.lower() in ("dec", "oct", "lin"))
            return ("noise", " ".join(t[1:k - 1]), t[k - 1], t[k].lower(), int(num(t[k + 1])), num(t[k + 2]), num(t[k + 3]))
        if kw == ".dc":  # SRC start stop step, repeated for each nested sweep (the first source is the innermost)
            if len(t) < 5 or (len(t) - 1) % 4: return ("unsupported", ".dc form (LIST/DEC/...)")
            return ("dc", *[x if k % 4 == 0 else num(x) for k, x in enumerate(t[1:])])
        if kw == ".tran":
            n = [num(x) for x in t[1:] if x.lower() != "uic"]
            return ("tran", n[0], n[1], any(x.lower() == "uic" for x in t[1:]))
    return ("op",)

# ---------------------------------------------------------------- engine side
def run_engine(args, text, mode, kind, step=0.0, stop=0.0):
    path = os.path.join(args.tmp, "eng.cir")
    open(path, "w").write(text)
    cmd = [os.path.abspath(args.engine), path, mode, kind] + ([repr(step), repr(stop)] if kind == "tran" else [])
    env = {**os.environ, **({"TREESPICE_BASEDIR": args.basedir} if getattr(args, "basedir", None) else {})}
    try: out = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout, env=env, cwd=args.tmp).stdout  # cwd: bsim4.osdi writes bsim4v5.out there
    except subprocess.TimeoutExpired: return dict(conv=False, err="TIMEOUT", pts=[])
    conv, err, pts, total, branches, nodes, pz, sens = False, "", [], None, [], [], dict(POLE=[], ZERO=[]), {}
    dnodes, devents = [], []
    for ln in out.splitlines():
        if ln.startswith("DNODES"): dnodes = ln.split()[1:]
        elif ln.startswith("DEVENT"): devents.append((float(ln.split()[1]), int(ln.split()[2]), ln.split()[3]))
        elif ln.startswith("BRANCHES"): branches = ln.split()[1:]
        elif ln.startswith("NODES"): nodes = ln.split()[1:]
        elif ln.startswith("TOTAL"): total = [float(x) for x in ln.split()[1:]]
        elif ln.startswith(("POLE ", "ZERO ")): pz[ln[:4]].append(complex(*map(float, ln.split()[1:3])))
        elif ln.startswith("SENS "): sens[ln.split()[1]] = float(ln.split()[2])
        elif ln.startswith("CONVERGED"): conv = ln.split()[1] == "1"
        elif ln.startswith("ERROR"): err = ln[6:].strip()
        elif ln and (ln[0].isdigit() or ln[0] == "-"):
            try: pts.append([float(x) for x in ln.split()])
            except ValueError: pass
    return dict(conv=conv, err=err, pts=pts, total=total, branches=branches, nn=len(nodes), pz=pz, sens=sens,
                nodes=nodes, dnodes=dnodes, devents=devents)

# ---------------------------------------------------------------- ngspice side
def tran_tmax(text, an):
    """Reference max step: tstep/10, and 1/20 of the shortest PULSE edge/width or SIN period.
    With only tstep/10, ngspice crossed a 5 us gate ramp in one 2.5 us step and carried the error
    for the rest of the run (qa_mos_coarse: 114 mV off a dense reference that the engine matched)."""
    spans = []
    for line in cards(text):
        t = toks(line)
        if t[0][0].upper() not in "VI": continue
        u = [x.upper() for x in t]
        try:
            if "PULSE" in u: spans += [num(x) for x in t[u.index("PULSE") + 4:u.index("PULSE") + 7]]  # tr tf pw
            if "SIN" in u and len(t) > u.index("SIN") + 3: spans.append(1 / num(t[u.index("SIN") + 3]))
        except (ValueError, ZeroDivisionError): pass
    edge = min([s for s in spans if s > 0], default=math.inf) / 20
    # ponytail: floor at tstop/1e5 so a ns edge in a ms run stays ~1e5 steps; such a run is checked at that density only.
    return max(min(an[1] / 10, edge), an[2] / 1e5)

def run_ngspice(args, text, an, names, cur=(), dn=()):
    """Return (xs, {name: [values]}) for the analysis, or raise RuntimeError. xs = [] for .op.
    Analysis/print cards are removed and replaced by a .control block (batch mode, ngspice defaults);
    the only option set is tmax on .tran so the reference is dense (see tran_tmax)."""
    drop = re.compile(r"\.(op|dc|tran|ac|noise|pz|sens|print|plot|four|end|osdi|save|probe|temp|step|meas|measure)\b", re.I)
    kept, skip, ctrl = [], False, False
    for i, ln in enumerate(text.splitlines()):
        if re.match(r"\s*\.control\b", ln, re.I): ctrl = True
        if ctrl:  # the netlist's own .control script is not the circuit
            ctrl = not re.match(r"\s*\.endc\b", ln, re.I)
            continue
        if i > 0 and ln.strip()[:1] == "+":
            if not skip: kept.append(ln)
            continue
        skip = i > 0 and bool(drop.match(ln.strip()))
        if not skip: kept.append(ln)
    cmd = {"op": "op", "dc": "dc " + " ".join(repr(x) if isinstance(x, float) else x for x in an[1:]) if an[0] == "dc" else "",
           "tran": "tran {0!r} {1!r} 0 {2!r}{3}".format(an[1], an[2], tran_tmax(text, an), " uic" if an[3] else "") if an[0] == "tran" else "",
           "ac": "ac {1} {2} {3!r} {4!r}".format(*an) if an[0] == "ac" else "", "noise": "", "pz": "", "sens": ""}[an[0]]
    keys = list(names) + [f"i({c})" for c in cur]
    vecs = " ".join(f"{k[2:-1]}#branch" if k.startswith("i(") else f"v({k.lower()})" for k in keys)
    out = os.path.join(args.tmp, "ng.dat")
    if os.path.exists(out): os.remove(out)
    ctl = [".control", cmd, f"wrdata {out} {vecs}"] + ([f"eprint {' '.join(dn)}"] if dn else []) + ["quit", ".endc", ".end"]
    if an[0] == "sens":  # every element parameter ngspice reports, as "name = value" (ngspice prints the dependence on each)
        ctl = [".control", f"sens {an[1]}", "print all", "quit", ".endc", ".end"]
    if an[0] == "pz":  # ngspice prints "name = re,im" (a lone vector as "all = re,im"): one run per kind, marked
        ctl = [".control"] + [x for k in ("pol", "zer") if an[6] in (k, "pz") for x in
                              (f"echo {k.upper()}", "pz " + " ".join(an[1:6]) + " " + k, "print all")] + ["quit", ".endc", ".end"]
    if an[0] == "noise":  # spectra in plot noise1, integrated totals in noise2
        ctl = [".control", "noise {1} {2} {3} {4} {5!r} {6!r}".format(*an), "setplot noise1",
               f"wrdata {out} onoise_spectrum inoise_spectrum", "setplot noise2",
               "print onoise_total inoise_total", "quit", ".endc", ".end"]
    for lib in osdi_libs(text):  # ngspice loads OSDI with pre_osdi (copied: its paths cannot hold spaces)
        dst = os.path.join(args.tmp, os.path.basename(lib))
        shutil.copy(os.path.join(OSDI_DIR, lib) if not os.path.isabs(lib) else lib, dst)
        ctl.insert(1, f"pre_osdi {dst}")
    cir = os.path.join(args.tmp, "ng.cir")
    rows = []
    # ngspice takes one .temp value and has no .step: each .temp value and .step combination is run on its own, the rows joined
    if an[0] == "sens":
        cir = os.path.join(args.tmp, "ng.cir")
        open(cir, "w").write("\n".join(kept) + "\n" + "\n".join(ctl) + "\n")
        p = subprocess.run([args.ngspice, "-b", cir], capture_output=True, text=True, timeout=args.timeout, cwd=args.tmp)
        got = {m[0].lower(): num_or_nan(m[1]) for m in re.findall(r"^(\S+) = (\S+)$", p.stdout, re.M)}
        if not got: raise RuntimeError("ngspice sens: " + " | ".join((p.stdout + p.stderr).strip().splitlines()[-3:]))
        return [], got
    if an[0] == "pz":
        cir = os.path.join(args.tmp, "ng.cir")
        open(cir, "w").write("\n".join(kept) + "\n" + "\n".join(ctl) + "\n")
        p = subprocess.run([args.ngspice, "-b", cir], capture_output=True, text=True, timeout=args.timeout, cwd=args.tmp)
        got, kind = dict(POL=[], ZER=[]), None
        for ln in p.stdout.splitlines():
            m = re.match(r"(POL|ZER)$|(?:pole\(\d+\)|zero\(\d+\)|all) = (\S+),(\S+)", ln.strip())
            if m and m.group(1): kind = m.group(1)
            elif m: got[kind].append(complex(float(m.group(2)), float(m.group(3))))
        if not any(got.values()) and "Error" in p.stdout + p.stderr: raise RuntimeError("ngspice pz: " + " | ".join((p.stdout + p.stderr).strip().splitlines()[-3:]))
        return [], dict(POLE=got["POL"], ZERO=got["ZER"])
    sd = steps(text)
    combos = itertools.product(*[[((k, n, vs), v) for v in vs] for k, n, vs in sd])
    for combo in combos:
        body, stemp = step_text(kept[1:], combo)
        for t in ([stemp] if stemp is not None else temps(text) if an[0] in ("op", "dc") else temps(text)[:1]) or [None]:
            if os.path.exists(out): os.remove(out)
            open(cir, "w").write("\n".join(kept[:1] + ([f".temp {t!r}"] if t is not None else []) + body + ctl) + "\n")
            p = subprocess.run([args.ngspice, "-b", cir], capture_output=True, text=True, timeout=args.timeout, cwd=args.tmp)  # cwd: ngspice BSIM4v5 paramchk writes bsim4v5.out
            if not os.path.exists(out) or os.path.getsize(out) == 0:
                raise RuntimeError("ngspice produced no data: " + " | ".join((p.stdout + p.stderr).strip().splitlines()[-3:]))
            rows += [[float(x) for x in ln.split()] for ln in open(out) if ln.strip()]
    if an[0] == "noise":
        tot = dict(re.findall(r"(\w+_total) = (\S+)", p.stdout))
        return [r[0] for r in rows], {"onoise": [r[1] for r in rows], "inoise": [r[3] for r in rows],
                                      "total": [float(tot["onoise_total"]), float(tot["inoise_total"])]}
    # wrdata layout: (scale, v1, scale, v2, ...); complex (.ac): (scale, re, im) per vector.
    # For .op the "scale" is meaningless.
    xs = [] if an[0] == "op" else [r[0] for r in rows]
    if dn:  # eprint: "Time or Step", the node names, then rows "<t> <v1> <v2> ..."
        table = [l.split() for l in p.stdout.splitlines() if re.match(r"^\d\.\d+e[+-]\d+\s", l) and len(l.split()) == len(dn) + 1]
        dig = {n: transitions((float(t[0]), t[1 + k]) for t in table) for k, n in enumerate(dn)}
        vals = {n: [r[2 * k + 1] for r in rows] for k, n in enumerate(keys)}
        vals["__digital__"] = dig
        return xs, vals
    if an[0] == "ac":
        return xs, {n: [complex(r[3 * k + 1], r[3 * k + 2]) for r in rows] for k, n in enumerate(keys)}
    return xs, {n: [r[2 * k + 1] for r in rows] for k, n in enumerate(keys)}

def num_or_nan(s):
    """A number as a simulator printed it; NaN in any spelling (MSVC's runtime prints -nan(ind))."""
    try: return float(s)
    except ValueError: return math.nan if "nan" in s.lower() else math.inf if "inf" in s.lower() else float(s)

def transitions(seq):
    """(t, value) samples -> the changes: the value at the first time, then each new one (the last at a time wins)."""
    out = []
    for t, v in seq:
        if out and out[-1][0] == t: out[-1] = (t, v)
        else: out.append((t, v))
        while len(out) > 1 and out[-1][1] == out[-2][1]: out.pop()
    return out

# ---------------------------------------------------------------- comparison
def interp(xs, ys, t):
    i = bisect.bisect_left(xs, t)
    if i == 0: return ys[0]
    if i >= len(xs): return ys[-1]
    if xs[i] == xs[i - 1]: return ys[i]
    w = (t - xs[i - 1]) / (xs[i] - xs[i - 1])
    return ys[i - 1] + w * (ys[i] - ys[i - 1])

def col(r, idx): return [row[idx] for row in r["pts"]]

def unsupported(r):
    return not r["conv"] and bool(re.search(r"unsupported|not supported|unknown (element|card)", r["err"], re.I))

def compare(args, text, an, mode, nm, ref):
    """-> (status, worst_abs, worst_rel, note)"""
    xs, vals = ref
    dig = vals.get("__digital__")  # XSPICE event nodes: {node: [(t, value) changes]}
    # node names, then "i(name)" branch currents; a reference current above 1 kA is a diverged run, not a value
    names = [n for n in vals if n != "__digital__" and (not n.startswith("i(") or max(abs(v) for v in vals[n]) <= 1e3)] if an[0] not in ("noise", "pz", "sens") else []
    def ecol(r, n, ac=False):  # engine column of a node or branch current (None: the engine has no such column)
        if n in nm: return 2 * nm[n] - 1 if ac else nm[n]
        b = n[2:-1]
        if b not in r["branches"]: return None
        k = r["branches"].index(b)
        return 1 + 2 * r["nn"] + 2 * k if ac else 1 + r["nn"] + k
    def atol(n, tran=False):
        return (args.i_tran_atol if tran else args.i_atol) if n.startswith("i(") else (args.tran_atol if tran else args.dc_atol)
    worst_abs = worst_rel = 0.0
    bad = []
    def judge(label, got, exp, tol, scale=None):
        # rel = |err| / max(|ref|, 1 mV) for DC;  |err| / node swing for transient (scale)
        nonlocal worst_abs, worst_rel
        e = abs(got - exp) if math.isfinite(got) else math.inf
        worst_abs = max(worst_abs, e); worst_rel = max(worst_rel, e / (scale or max(abs(exp), 1e-3)))
        if not e <= tol: bad.append(f"{label}: {got:.6g} vs ngspice {exp:.6g}")
    if an[0] == "sens":
        r = run_engine(args, text, mode, "sens")
        if unsupported(r): return "ENGINE-UNSUPPORTED", 0, 0, r["err"]
        if not r["conv"]: return "FAIL", math.inf, math.inf, f"engine not converged: {r['err']}"
        # Element values ngspice reports at the top level: R (r, r_m, r_scale), V (v), I (i, i_m). Device parameters are
        # not implemented: the engine lists those devices as UNSUPPORTED and the harness does not compare them.
        want, names, in_sub = [], {}, False
        for t in map(toks, cards(text)):
            if t[0][0] == ".": in_sub = t[0].upper() == ".SUBCKT" or (in_sub and t[0].upper() != ".ENDS")
            elif not in_sub and t[0][0].upper() in "RVI": names[t[0].lower()] = t[0][0].upper()
        for n, c in names.items(): want += [n] + ([n + "_m", n + "_scale"] if c == "R" else [n + "_m"] if c == "I" else [])
        scale = max([abs(vals[n]) for n in want if n in vals], default=0)
        for n in want:
            if n not in vals: continue  # ngspice does not report it (e.g. v1_m)
            if n not in r["sens"]: bad.append(f"engine has no {n}"); continue
            got, exp = r["sens"][n], vals[n]
            worst_abs = max(worst_abs, abs(got - exp)); worst_rel = max(worst_rel, abs(got - exp) / max(abs(exp), 1e-300))
            if abs(got - exp) > 1e-6 * scale + args.dc_rtol * abs(exp): bad.append(f"d/d{n}: {got:.6g} vs ngspice {exp:.6g}")
    elif an[0] == "pz":
        r = run_engine(args, text, mode, "pz")
        if unsupported(r): return "ENGINE-UNSUPPORTED", 0, 0, r["err"]
        if not r["conv"]: return "FAIL", math.inf, math.inf, f"engine not converged: {r['err']}"
        for kind in ("POLE", "ZERO"):  # order-insensitive: each reference root takes the nearest unused engine root
            exp, got = vals[kind], list(r["pz"][kind])
            # ngspice's root search misses the highest-frequency poles of device circuits (7 of 9 on x_pz_bjt_amp): the
            # engine may have extra roots, but only above every root ngspice reports. A root at the origin is matched to 1 mrad/s.
            if len(got) < len(exp): bad.append(f"{kind.lower()}s: engine {len(got)} vs ngspice {len(exp)}"); continue
            for z in exp:
                near = min(got, key=lambda g: abs(g - z))
                got.remove(near)
                worst_abs = max(worst_abs, abs(near - z)); worst_rel = max(worst_rel, abs(near - z) / max(abs(z), 1.0))
                if abs(near - z) > 1e-3 + args.dc_rtol * abs(z): bad.append(f"{kind.lower()} {near:.6g} vs ngspice {z:.6g}")
            for g in got:
                if abs(g) <= max(map(abs, exp), default=0): bad.append(f"engine has extra {kind.lower()} {g:.6g} inside ngspice's range")
    elif an[0] == "noise":
        r = run_engine(args, text, mode, "noise")
        if unsupported(r): return "ENGINE-UNSUPPORTED", 0, 0, r["err"]
        if not r["conv"] or len(r["pts"]) != len(xs) or not r["total"]:
            return "FAIL", math.inf, math.inf, f"engine gave {len(r['pts'])} of {len(xs)} frequencies: {r['err']}"
        def rel(label, got, exp):  # noise densities and totals: relative (DC rtol)
            nonlocal worst_abs, worst_rel
            e = abs(got - exp); worst_abs = max(worst_abs, e); worst_rel = max(worst_rel, e / max(abs(exp), 1e-300))
            if not e <= args.dc_rtol * abs(exp): bad.append(f"{label}: {got:.6g} vs ngspice {exp:.6g}")
        for k, row in enumerate(r["pts"]):
            rel(f"f={xs[k]:g} onoise", row[1], vals["onoise"][k])
            rel(f"f={xs[k]:g} inoise", row[2], vals["inoise"][k])
        rel("onoise_total", r["total"][0], vals["total"][0])
        rel("inoise_total", r["total"][1], vals["total"][1])
    elif an[0] == "ac":
        r = run_engine(args, text, mode, "ac")
        if unsupported(r): return "ENGINE-UNSUPPORTED", 0, 0, r["err"]
        if not r["conv"] or len(r["pts"]) != len(xs):
            return "FAIL", math.inf, math.inf, f"engine gave {len(r['pts'])} of {len(xs)} frequencies: {r['err']}"
        for k, row in enumerate(r["pts"]):
            if abs(row[0] - xs[k]) > 1e-9 * xs[k]: bad.append(f"f={row[0]:g} vs ngspice {xs[k]:g}")
            for n in names:
                c = ecol(r, n, True)
                if c is None: bad.append(f"no engine column for {n}"); continue
                got, exp = complex(row[c], row[c + 1]), vals[n][k]
                e = abs(got - exp)
                worst_abs = max(worst_abs, e); worst_rel = max(worst_rel, e / max(abs(exp), 1e-3))
                if not e <= atol(n) + args.dc_rtol * abs(exp):
                    bad.append(f"f={xs[k]:g} {n if n.startswith('i(') else f'v({n})'}: {got:.6g} vs ngspice {exp:.6g}")
    elif an[0] in ("op", "dc"):
        r = run_engine(args, text, mode, "dc")   # .op: one row; .dc: one row per sweep point
        if unsupported(r): return "ENGINE-UNSUPPORTED", 0, 0, r["err"]
        if not r["conv"] or not r["pts"]: return "FAIL", math.inf, math.inf, f"engine not converged: {r['err']}"
        want = len(xs) or len(next(iter(vals.values()), [None]))  # .op: one row per .temp value
        if len(r["pts"]) != want:
            return "FAIL", math.inf, math.inf, f"engine gave {len(r['pts'])} points, ngspice {want}"
        for k, row in enumerate(r["pts"]):
            at = "" if an[0] == "op" and want == 1 else f"row {k} " + (f"{an[1]}={row[0]:g} " if an[0] == "dc" else "")
            if an[0] == "dc" and abs(row[0] - xs[k]) > 1e-9 * max(1.0, abs(xs[k])): bad.append(f"{at}sweep value vs ngspice {xs[k]:g}")
            for n in names:
                c, exp = ecol(r, n), vals[n][k]
                if c is None: bad.append(f"no engine column for {n}"); continue
                judge(f"{at}{n if n.startswith('i(') else f'v({n})'}", row[c], exp, atol(n) + args.dc_rtol * abs(exp))
    else:
        tstep, tstop = an[1], an[2]
        r = run_engine(args, text, mode, "tran", tstep, tstop)
        if unsupported(r): return "ENGINE-UNSUPPORTED", 0, 0, r["err"]
        if not r["pts"]: return "FAIL", math.inf, math.inf, f"engine produced no waveform: {r['err']}"
        ts = col(r, 0)
        if ts[-1] < tstop * 0.999: bad.append(f"engine stopped at t={ts[-1]:g} < tstop={tstop:g} ({r['err']})")
        sel = [i for i, t in enumerate(ts) if xs[0] <= t <= xs[-1]]  # reference may start at TSTART
        # timing tolerance (see the policy above). With XSPICE A devices, tstep: an adc_bridge samples at accepted time
        # points, so an event it starts (and every analog edge after it) can sit one analog step from ngspice's.
        dt = tstep if dig is not None else tstep / 10
        def charge(tt, yy):  # cumulative trapezoid integral, as a function of time
            q = [0.0]
            for j in range(1, len(tt)): q.append(q[-1] + 0.5 * (yy[j] + yy[j - 1]) * (tt[j] - tt[j - 1]))
            return lambda t: interp(tt, q, t)
        for n in names:
            ref_n = vals[n]
            c = ecol(r, n)
            if c is None: bad.append(f"no engine column for {n}"); continue
            eng = col(r, c)
            qe, qr = charge(ts, eng), charge(xs, ref_n)
            swing = max(ref_n) - min(ref_n)
            tol = max(atol(n, True), args.tran_rtol * swing)
            if n.startswith("i("): tol = max(tol, args.i_tran_rtol * swing, 0.02 * max(abs(v) for v in ref_n))
            for i in sel:
                # The reference's range over [t - dt, t + dt]; plus, where it jumps more than tol inside
                # one cell (a switching edge), that cell's bracket: interpolation would invent a value.
                win = [interp(xs, ref_n, ts[i] - dt), interp(xs, ref_n, ts[i] + dt)]
                win += ref_n[bisect.bisect_left(xs, ts[i] - dt):bisect.bisect_right(xs, ts[i] + dt)]
                k = bisect.bisect_left(xs, ts[i])
                if 0 < k < len(xs) and abs(ref_n[k] - ref_n[k - 1]) > tol: win += [ref_n[k - 1], ref_n[k]]
                got = min(max(eng[i], min(win)), max(win))
                if n.startswith("i(") and abs(eng[i] - got) > tol:
                    # A switching spike narrower than the reference's steps carries the same charge: forgive the
                    # instant when the mean over its .tran step agrees.
                    a, b = max(ts[i] // tstep * tstep, xs[0]), min((ts[i] // tstep + 1) * tstep, xs[-1])
                    if b - a > 1e-9 * tstep and abs((qe(b) - qe(a)) - (qr(b) - qr(a))) <= tol * (b - a): continue
                judge(f"t={ts[i]:g} {n if n.startswith('i(') else f'v({n})'}", eng[i], got, tol, max(swing, 1e-3))
        if dig is not None:  # the same transitions in the same order, each within dt of ngspice's
            def settled(seq):
                # An undetermined state (U, or u strength) held less than dt is a sampling artifact: it appears only
                # when an analog point lands inside an adc_bridge's threshold band, which each simulator's steps decide.
                # Within 2 dt of tstop one run's next event may fall after the end: not judged.
                keep = [e for j, e in enumerate(seq) if e[0] <= tstop - 2 * dt and
                        not (("U" in e[1] or e[1][1] == "u") and j + 1 < len(seq) and seq[j + 1][0] - e[0] < dt)]
                return transitions(keep)
            for k, n in enumerate(r["dnodes"]):
                got = settled(transitions((t, v) for t, idx, v in r["devents"] if idx == k))
                exp = dig.get(n)
                exp = exp and settled(exp)
                if exp is None: bad.append(f"ngspice has no event node {n}"); continue
                if [v for _, v in got] != [v for _, v in exp]:
                    bad.append(f"{n}: {' '.join(f'{v}@{t:.4g}' for t, v in got[:6])} vs ngspice {' '.join(f'{v}@{t:.4g}' for t, v in exp[:6])}")
                    continue
                for (te, v), (tr, _) in zip(got, exp):
                    worst_abs = max(worst_abs, abs(te - tr) / dt * 1e-12)  # (reported in ps-per-step units, not volts)
                    if abs(te - tr) > dt * (1 + 1e-9): bad.append(f"{n} -> {v} at {te:.6g} vs ngspice {tr:.6g}")
        if not r["conv"] and not bad: bad.append(f"engine reports not converged: {r['err']}")
    return ("FAIL" if bad else "PASS"), worst_abs, worst_rel, "; ".join(bad[:3]) + (f" (+{len(bad) - 3} more)" if len(bad) > 3 else "")

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-k", default="", help="only netlists whose name contains this substring")
    ap.add_argument("--modes", default="cpu,gpu")
    ap.add_argument("--engine", default=ENGINE)
    ap.add_argument("--ngspice", default="ngspice")
    ap.add_argument("--timeout", type=int, default=300)
    for k, v in DEFAULTS.items(): ap.add_argument("--" + k.replace("_", "-"), type=float, default=v)
    ap.add_argument("-v", action="store_true", help="print mismatch details")
    args = ap.parse_args()
    if not os.access(args.engine, os.X_OK):
        sys.exit(f"engine not found: {args.engine}  (cmake -S . -B build && cmake --build build --target acutesim_engine_dump)")
    args.tmp = tempfile.mkdtemp(prefix="ngdiff_")
    atexit.register(shutil.rmtree, args.tmp, True)
    if os.path.isdir(os.path.join(HERE, "netlists", "lib")):  # .include/.lib targets, beside the temp netlists
        shutil.copytree(os.path.join(HERE, "netlists", "lib"), os.path.join(args.tmp, "lib"))
    pdk = os.path.join(ROOT, "datasets", "pdk")  # downloaded PDKs (git-ignored), as pdk/<name>/...
    if os.path.isdir(pdk): os.symlink(pdk, os.path.join(args.tmp, "pdk"))
    fails = 0
    for path in sorted(glob.glob(os.path.join(HERE, "netlists", "*.cir"))):
        name = os.path.basename(path)
        if args.k not in name: continue
        text = re.sub(r"(?ims)^\s*\.control\b.*?^\s*\.endc\b[^\n]*\n?", "", open(path).read())  # a script, not circuit
        an = analysis(text)
        if len(temps(text)) > 1 and an[0] not in ("op", "dc"): an = ("unsupported", ".temp list with a transient/AC/noise run")
        if steps(text) and an[0] not in ("op", "dc"): an = ("unsupported", ".step with a transient/AC/noise run: tests/cli_check.py runs it per step")
        if an[0] == "unsupported":
            print(f"{'SKIPPED':<19} {name} ({an[1]})"); continue
        full = with_includes(text, os.path.dirname(path))
        missing = [l for l in needed_osdi(full) if not os.path.exists(os.path.join(OSDI_DIR, l))]
        missing += sorted({p for p in re.findall(r"\bpdk/[\w.-]+", full) if not os.path.exists(os.path.join(ROOT, "datasets", p))})
        if missing:
            print(f"{'SKIPPED':<19} {name} (missing {', '.join(missing)}: datasets/osdi from tests/build_osdi.sh, datasets/pdk see ENGINE_HANDOFF)"); continue
        nm = node_map(text) if an[0] not in ("noise", "pz", "sens") else {}
        dn = ()
        if re.search(r"^\s*a\S*\s", text, re.M | re.I) and an[0] == "tran":
            # XSPICE: which A ports are analog depends on the model, so take the engine's own node lists
            pre = run_engine(args, text, "cpu", "tran", an[1], an[2])
            nm, dn = {n.lower(): k + 1 for k, n in enumerate(pre["nodes"]) if n != "-"}, tuple(pre["dnodes"])
        try: ref = run_ngspice(args, text, an, list(nm), branch_names(text) if an[0] not in ("noise", "pz", "sens") else (), dn)
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            print(f"{'NGSPICE-ERROR':<19} {name} {e}"); fails += 1; continue
        for mode in args.modes.split(","):
            st, wa, wr, note = compare(args, text, an, mode, nm, ref)
            fails += st == "FAIL"
            print(f"{st:<19} {name:<32} [{mode}] worst abs={wa:.3g} V  rel={wr:.3g}" + (f"  {note}" if note and (args.v or st != "PASS") else ""), flush=True)
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
