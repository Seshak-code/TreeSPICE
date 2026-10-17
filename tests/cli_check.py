#!/usr/bin/env python3
"""treespice CLI check against ngspice: the .print tables of x_print_*.cir (.dc, .ac) match ngspice's, the raw
files treespice writes (binary and ascii) load in ngspice and hold the same vectors, and the .meas values of
x_meas_*.cir match ngspice's (within "* meastol REL ABS": the simulators' time points differ; "* expect NAME VALUE"
checks a measure ngspice lacks, as DERIV; .step runs are compared step by step with ngspice on the substituted netlist).
    python3 tests/cli_check.py [--treespice build/treespice] [--ngspice ngspice]"""
import argparse, glob, os, re, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET = os.path.join(ROOT, "tests", "ngspice_diff", "netlists")

def table(out):
    """The first .print table in a log -> (header words after Index, rows of floats; a complex cell is two floats)."""
    lines, rows, head = out.splitlines(), [], None
    for k, ln in enumerate(lines):
        if ln.startswith("Index"): head = ln.split()[1:]; continue
        if head is None or ln.startswith("-"): continue
        t = ln.replace(",", " ").split()
        if not t or not t[0].isdigit(): 
            if rows: break
            continue
        rows.append([float(x) for x in t[1:]])
    return head, rows

def measures(out):
    """{name: value} of the "name = value" lines of a log, and the names of the measures that failed."""
    vals, failed = {}, set()
    for ln in out.splitlines():
        if m := re.match(r"(\w+)\s+=\s+(\S+)", ln): vals[m[1]] = float(m[2])
        if m := re.match(r"Error: measure\s+(\w+)\s", ln): failed.add(m[1])
    return vals, failed

def step_blocks(out):
    """A .step run's log -> one (values, failed) per "Sweep:" block (the measures follow each block's table)."""
    parts = re.split(r"^Sweep:.*$", out, flags=re.M)[1:]
    return [measures(p) for p in parts]

def close(a, b): return abs(a - b) <= 1e-9 + 2e-3 * max(abs(a), abs(b))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--treespice", default=os.path.join(ROOT, "build", "treespice"))
    ap.add_argument("--ngspice", default="ngspice")
    a = ap.parse_args()
    bad, tmp = 0, tempfile.mkdtemp(prefix="clichk_")
    def fail(msg):
        nonlocal bad; bad += 1; print("FAIL", msg)
    for net in sorted(glob.glob(os.path.join(NET, "x_print_*.cir"))):
        name = os.path.basename(net)
        ts = subprocess.run([a.treespice, net], capture_output=True, text=True)
        if ts.returncode: fail(f"{name}: treespice exit {ts.returncode}: {ts.stderr.strip()[:200]}"); continue
        th, tr = table(ts.stdout)
        if not tr: fail(f"{name}: no table"); continue
        if "tran" not in name and "dcnest" not in name:  # tran steps differ between the simulators: raw check only
            ng = subprocess.run([a.ngspice, "-b", net], capture_output=True, text=True).stdout
            # ngspice prints one table per output column when a column is complex: keep to real-valued columns
            nh, nr = table(ng)
            if len(nr) != len(tr): fail(f"{name}: {len(tr)} rows vs ngspice {len(nr)}")
            elif name == "x_print_dc.cir":
                for r1, r2 in zip(tr, nr):
                    if not all(close(x, y) for x, y in zip(r1, r2)): fail(f"{name}: {r1} vs ngspice {r2}")
                print("PASS", name, "table")
            else:  # ac: ngspice's .print splits a wide table; its wrdata of the same expressions gives every column
                specs = re.search(r"^\.print\s+ac\s+(.*)$", open(net).read(), re.I | re.M).group(1).split()
                fn = {"vdb": "db", "vm": "mag", "vp": "ph", "vr": "real", "vi": "imag"}
                ex = [f"{fn[m.group(1)]}(v({m.group(2)}))" if (m := re.fullmatch(r"(v[dmpri]b?)\((\w+)\)", x)) and m.group(1) in fn else x
                      for x in specs]
                ctl, dat = os.path.join(tmp, "ac.cir"), os.path.join(tmp, "ac.dat")
                body = [l for l in open(net).read().splitlines() if not l.lower().startswith((".print", ".end"))]
                open(ctl, "w").write("\n".join(body + [".control", "ac dec 2 100 10k", f"wrdata {dat} " + " ".join(ex), "quit", ".endc", ".end"]) + "\n")
                subprocess.run([a.ngspice, "-b", ctl], capture_output=True)
                cells = []
                for l in open(dat):  # per expression: scale, then re[, im] (plain v()/i() are complex)
                    v, c, k = [float(x) for x in l.split()], [], 0
                    for x in ex:
                        n = 2 if re.match(r"[vi]\(", x) else 1
                        c += ([v[0]] if not c else []) + v[k + 1:k + 1 + n]
                        k += 1 + n
                    cells.append(c)
                if len(cells) != len(tr) or not all(len(c) == len(r) and all(close(x, y) for x, y in zip(c, r)) for c, r in zip(cells, tr)):
                    fail(f"{name}: table differs from ngspice: {tr[0]} vs {cells[0]}")
                else: print("PASS", name, f"table ({len(ex)} columns)")
        # raw: load in ngspice, compare against treespice's own table
        for ascii_ in (False, True):
            raw = os.path.join(tmp, name + (".a" if ascii_ else ".b") + ".raw")
            subprocess.run([a.treespice, "-r", raw] + (["--ascii"] if ascii_ else []) + [net], capture_output=True, check=True)
            vec = "v(out)"
            ctl = os.path.join(tmp, "ld.cir")
            dat = os.path.join(tmp, "ld.dat")
            open(ctl, "w").write(f"x\n.control\nload {raw}\nwrdata {dat} {vec}\nquit\n.endc\n.end\n")
            subprocess.run([a.ngspice, "-b", ctl], capture_output=True)
            if not os.path.exists(dat): fail(f"{name}: ngspice could not load {raw}"); continue
            got = [[float(x) for x in l.split()] for l in open(dat) if l.strip()]
            os.remove(dat)
            ci = th.index(vec) if vec in th else None
            if ci is None: fail(f"{name}: no {vec} in table"); continue
            # a printed column's cells: scale first, then per column (complex = 2 cells)
            width = lambda h: 2 if "ac" in name and h in ("v(out)",) else 1
            off = 1 + sum(width(h) for h in th[1:ci])
            if "dcnest" in name: got = got[:len(tr)]  # one plot holds every outer value; the table is the first
            if len(got) != len(tr): fail(f"{name}: raw has {len(got)} points, table {len(tr)}")
            elif not all(close(g[1], r[off]) for g, r in zip(got, tr)): fail(f"{name}: raw v(out) differs from the table")
            else: print("PASS", name, "ascii raw" if ascii_ else "binary raw")
    for net in sorted(glob.glob(os.path.join(NET, "x_meas_*.cir"))):
        name, text = os.path.basename(net), open(net).read()
        rel, ab = [float(x) for x in re.search(r"^\* meastol (\S+) (\S+)", text, re.M).groups()]
        expect = {m[1]: float(m[2]) for m in re.finditer(r"^\* expect (\w+) (\S+)", text, re.M)}
        ts = subprocess.run([a.treespice, net], capture_output=True, text=True)
        if ts.returncode: fail(f"{name}: treespice exit {ts.returncode}: {ts.stderr.strip()[:200]}"); continue
        step = re.search(r"^\.step\s+param\s+(\w+)\s+list\s+(.*)$", text, re.I | re.M)
        if step:  # ngspice 47 has no .step: one run per value with the parameter set
            want = []
            for v in step.group(2).split():
                src = os.path.join(tmp, name + "." + v + ".cir")
                open(src, "w").write(re.sub(r"^\.step.*$", "", text, flags=re.I | re.M).replace(".end", f".param {step.group(1)}={v}\n.end"))
                want.append(measures(subprocess.run([a.ngspice, "-b", src], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout))
            got = step_blocks(ts.stdout)
        else:
            want, got = [measures(subprocess.run([a.ngspice, "-b", net], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout)], [measures(ts.stdout)]
        if len(got) != len(want): fail(f"{name}: {len(got)} step blocks, ngspice runs {len(want)}"); continue
        n, bad0 = 0, bad
        for k, ((wv, wf), (gv, gf)) in enumerate(zip(want, got)):
            at = f"{name}{f' step {k}' if step else ''}"
            for m in sorted(set(wv) | wf | set(gv) | gf):
                n += 1
                if m in expect: ok = m in gv and abs(gv[m] - expect[m]) <= ab + rel * abs(expect[m])
                elif m in wf: ok = m in gf and m not in gv
                elif m in wv: ok = m in gv and abs(gv[m] - wv[m]) <= ab + rel * abs(wv[m])
                else: ok = False  # a measure ngspice did not run or reject
                if not ok: fail(f"{at}: {m} = {gv.get(m, 'failed' if m in gf else 'missing')} vs ngspice {wv.get(m, 'failed' if m in wf else 'missing')}")
        if bad == bad0: print("PASS", name, f"{n} measures")
    # expressions: v(a)*2 + 1 and v(a,b) equal their pieces (ngspice's .print cannot do arithmetic)
    for kind, card, cells in (("dc", ".dc V1 0 5 1", 1), ("ac", ".ac dec 1 100 1k", 2)):
        net = os.path.join(tmp, "expr.cir")
        open(net, "w").write(f"e\nV1 in 0 DC 1 AC 1\nR1 in out 1k\nR2 out 0 2k\nC1 out 0 1u\n{card}\n"
                             f".print {kind} v(out) v(out)*2+1 v(in,out) v(in)-v(out)\n.end\n")
        out = subprocess.run([a.treespice, net], capture_output=True, text=True).stdout
        _, rows = table(out)
        for r in rows:
            v, e, d1, d2 = (r[1:1 + cells], r[1 + cells:1 + 2 * cells], r[1 + 2 * cells:1 + 3 * cells], r[1 + 3 * cells:])
            ok = all(close(2 * x + (1 if k == 0 else 0), y) for k, (x, y) in enumerate(zip(v, e))) and d1 == d2
            if not ok: fail(f"expr {kind}: {r}")
        print("PASS" if rows else "FAIL", "expr", kind); bad += not rows
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
