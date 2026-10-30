#!/usr/bin/env python3
"""Reference cross-check: TreeSPICE against deterministic SymPy and mpmath computations of the same mathematics.

Each case builds its reference without the engine's code: an MNA system assembled here from the netlist (exact
rationals in SymPy), solved exactly or at 50 digits in mpmath; device currents from the published model equations;
transients replayed from the integration formulas on the engine's own accepted time points. Every comparison has a
bound derived from the arithmetic involved (a condition number times the unit roundoff, a Newton tolerance), printed
beside the measured error, so a PASS means the engine computed what the mathematics says to within rounding.

    ~/eda/venv-refcheck/bin/python tests/refcheck/refcheck.py [--build build] [area ...]   # areas: dae solver device

Needs sympy 1.14 and mpmath 1.3 (python3 -m venv ~/eda/venv-refcheck && ~/eda/venv-refcheck/bin/pip install
sympy==1.14.0 mpmath==1.3.0), and the engine's treespice, acutesim_engine_dump and solve_triplets in --build.
"""
import argparse, math, os, random, re, struct, subprocess, sys, tempfile
from fractions import Fraction

try:
    import mpmath as mp
    import sympy as sp
except ImportError:
    sys.exit("refcheck needs sympy and mpmath: ~/eda/venv-refcheck/bin/python " + " ".join(sys.argv)
             + "  (python3 -m venv ~/eda/venv-refcheck && ~/eda/venv-refcheck/bin/pip install sympy==1.14.0 mpmath==1.3.0)")

mp.mp.dps = 50
U = 2.0 ** -53                                   # unit roundoff of IEEE double
K_B, Q_E, T_NOM = mp.mpf("1.380649e-23"), mp.mpf("1.602176634e-19"), mp.mpf("300.15")  # SI 2019 exact; 27 C
VT = K_B * T_NOM / Q_E
GMIN = mp.mpf("1e-12")                           # SPICE's junction GMIN default
TIGHT = ".options reltol=1e-12 vntol=1e-15 abstol=1e-21\n"  # Newton error below the comparison's resolution
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUILD = os.path.join(ROOT, "build")
results = []                                     # (area, case, passed)


def report(area, case, what, err, bound):
    ok = bool(err <= bound)
    results.append((area, case, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {case:28s} {what:44s} err {float(err):.3e}  bound {float(bound):.3e}")


# --------------------------------------------------------------------------------------------- engine I/O

def read_raw(path):
    """ngspice binary raw file(s): a list of plots {name, vars: [names], cols: {name: [values]}} at full precision"""
    data, pos, plots = open(path, "rb").read(), 0, []
    while pos < len(data):
        hdr, names = {}, []
        while True:
            end = data.index(b"\n", pos)
            line, pos = data[pos:end].decode(), end + 1
            if line.startswith(("Binary:", "Values:")): break
            if line.startswith("\t"): names.append(line.split("\t")[2].lower())
            elif ":" in line: k, v = line.split(":", 1); hdr[k.strip()] = v.strip()
        if line.startswith("Values:"): raise RuntimeError("refcheck reads binary raw files only")
        cplx, nv, npts = "complex" in hdr["Flags"], int(hdr["No. Variables"]), int(hdr["No. Points"])
        width = 2 if cplx else 1
        vals = struct.unpack_from(f"<{nv * npts * width}d", data, pos)
        pos += 8 * nv * npts * width
        cols = {n: [] for n in names}
        for p in range(npts):
            for j, n in enumerate(names):
                k = (p * nv + j) * width
                cols[n].append(complex(vals[k], vals[k + 1]) if cplx else vals[k])
        plots.append({"name": hdr["Plotname"], "vars": names, "cols": cols})
    return plots


def treespice(deck):
    with tempfile.TemporaryDirectory() as d:
        cir, raw = os.path.join(d, "c.cir"), os.path.join(d, "c.raw")
        open(cir, "w").write(deck)
        p = subprocess.run([os.path.join(BUILD, "treespice"), "--cpu", cir, "-r", raw], capture_output=True, text=True, timeout=300)
        if p.returncode != 0 or not os.path.exists(raw): raise RuntimeError(f"treespice failed: {p.stdout}{p.stderr}")
        return read_raw(raw)


def dump(deck, mode):
    with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False) as f: f.write(deck)
    try:
        p = subprocess.run([os.path.join(BUILD, "acutesim_engine_dump"), f.name, "cpu", mode], capture_output=True, text=True, timeout=300)
    finally: os.unlink(f.name)
    return p.stdout


# --------------------------------------------------------------------------------------------- an independent MNA

SUFFIX = {"f": "1e-15", "p": "1e-12", "n": "1e-9", "u": "1e-6", "m": "1e-3", "k": "1e3", "meg": "1e6", "g": "1e9", "t": "1e12"}


def num(tok):
    """A SPICE number as an exact rational (the decimal text, not its binary rounding)"""
    m = re.fullmatch(r"([-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?)(meg|[fpnumkgt])?[a-z]*", tok.lower())
    if not m: raise ValueError(f"not a number: {tok}")
    return Fraction(m.group(1)) * (Fraction(SUFFIX[m.group(2)]) if m.group(2) else 1)


class Mna:
    """G x + C x' = s(t) for R, C, L, K, V, I, E, F, G, H cards; x = node voltages (first appearance), then the
    branch currents of V, L, E, H in card order (a V/L/E/H current flows into its + terminal, as ngspice's i(name))"""

    def __init__(self, deck):
        cards = [l.split() for l in re.sub(r"\(|\)|,", " ", deck).lower().splitlines()[1:] if l.strip() and l[0] not in "*."]
        self.nodes, self.branches, self.src = [], [], {}
        for c in cards:
            if c[0][0] == "k": continue  # K names two inductors, not nodes
            for nd in c[1:3] + (c[3:5] if c[0][0] in "eg" else []):
                if nd != "0" and nd not in self.nodes: self.nodes.append(nd)
            if c[0][0] in "vleh": self.branches.append(c[0])
        n = len(self.nodes) + len(self.branches)
        self.n, self.G, self.C = n, sp.zeros(n, n), sp.zeros(n, n)
        self.s_dc, self.s_ac, self.wave = sp.zeros(n, 1), sp.zeros(n, 1), {}
        ix = lambda nd: None if nd == "0" else self.nodes.index(nd)
        br = lambda name: len(self.nodes) + self.branches.index(name)
        def add(M, r, c, v):
            if r is not None and c is not None: M[r, c] += v
        inductance = {}
        for c in cards:
            if c[0][0] == "k": continue  # coupled below, once every inductor is known
            kind, a, b = c[0][0], ix(c[1]), ix(c[2])
            if kind in "rc":
                v = sp.Rational(num(c[3]))
                M, g = (self.G, 1 / v) if kind == "r" else (self.C, v)
                add(M, a, a, g); add(M, b, b, g); add(M, a, b, -g); add(M, b, a, -g)
            elif kind in "vleh":
                k = br(c[0])
                add(self.G, a, k, 1); add(self.G, b, k, -1); add(self.G, k, a, 1); add(self.G, k, b, -1)
                if kind == "l": inductance[c[0]] = sp.Rational(num(c[3])); self.C[k, k] -= inductance[c[0]]
                elif kind == "e": add(self.G, k, ix(c[3]), -sp.Rational(num(c[5]))); add(self.G, k, ix(c[4]), sp.Rational(num(c[5])))
                elif kind == "h": self.G[k, br(c[3])] -= sp.Rational(num(c[4]))
                else: self._source(c, [(k, 1)])
            elif kind == "i":  # SPICE: its current flows from n+ through the source to n-, so it leaves n+
                self._source(c, [(r, sg) for r, sg in ((a, -1), (b, 1)) if r is not None])
            elif kind == "g":
                gm = sp.Rational(num(c[5]))
                for r, sg in ((a, 1), (b, -1)):
                    add(self.G, r, ix(c[3]), sg * gm); add(self.G, r, ix(c[4]), -sg * gm)
            elif kind == "f":
                k = br(c[3]); g = sp.Rational(num(c[4]))
                add(self.G, a, k, g); add(self.G, b, k, -g)
        for c in cards:  # K couples two inductors declared anywhere in the deck
            if c[0][0] == "k":
                m = sp.Rational(num(c[3])) * sp.sqrt(inductance[c[1]] * inductance[c[2]])
                self.C[br(c[1]), br(c[2])] -= m; self.C[br(c[2]), br(c[1])] -= m

    def _source(self, c, rows):
        """a V or I card's DC, AC and PULSE values onto the right-hand side rows given (row, sign)"""
        rest, dc, ac, wave, i = c[3:], Fraction(0), Fraction(0), None, 0
        while i < len(rest):
            t = rest[i]
            if t == "dc": dc = num(rest[i + 1]); i += 2
            elif t == "ac": ac = num(rest[i + 1]); i += 2
            elif t == "pulse": wave = [num(x) for x in rest[i + 1:i + 8]]; i += 8
            else: dc = num(t); i += 1
        if wave: dc = wave[0]
        for r, sg in rows:
            self.s_dc[r] += sg * sp.Rational(dc)
            self.s_ac[r] += sg * sp.Rational(ac)
            if wave: self.wave[r] = (sg, wave)

    def s(self, t):
        """the excitation at time t (mpmath), PULSE as ngspice evaluates it"""
        out = mp.matrix([mp.mpf(sp.Rational(x).p) / sp.Rational(x).q for x in self.s_dc])
        for r, (sg, w) in self.wave.items():
            v1, v2, td, tr, tf, pw, per = [mp.mpf(x.numerator) / x.denominator for x in w + [0] * (7 - len(w))][:7]
            tt = t - td
            if per > 0 and tt > 0: tt = tt - per * mp.floor(tt / per)
            v = v1 if t <= td else v1 + (v2 - v1) * tt / tr if tt < tr else v2 if tt < tr + pw else \
                v2 + (v1 - v2) * (tt - tr - pw) / tf if tt < tr + pw + tf else v1
            out[r] = sg * v
        return out

    def corners(self, stop):
        out = set()
        for _, w in self.wave.values():
            td, tr, tf, pw, per = [float(x) for x in (w + [0] * 7)[2:7]]
            base = 0.0
            while base <= stop:
                for c in (td, td + tr, td + tr + pw, td + tr + pw + tf): out.add(base + c)
                if per <= 0: break
                base += per
        return sorted(out)

    def names(self):
        return [f"v({x})" for x in self.nodes] + [f"i({x})" for x in self.branches]


def mpm(M):
    """a SymPy matrix (rationals, or exact irrationals such as K's sqrt(L1 L2)) at the working precision"""
    def one(v):
        return mp.mpf(v.p) / v.q if v.is_Rational else mp.mpf(str(sp.N(v, mp.mp.dps + 10)))
    return mp.matrix([[one(v) for v in M.row(i)] for i in range(M.rows)])


def cond_inf(A):
    """condition number of A with its rows equilibrated (Bauer-Skeel): the scale-invariant one, since MNA rows mix
    amperes (KCL) and volts (branch equations) and the plain kappa(A) overstates the error by their ratio"""
    n = A.rows
    D = [1 / max(abs(A[i, j]) for j in range(A.cols)) for i in range(n)]
    S = mp.matrix([[D[i] * A[i, j] for j in range(A.cols)] for i in range(n)])
    return mp.mnorm(S, "inf") * mp.mnorm(mp.inverse(S), "inf")


def maxabs(v): return max(abs(x) for x in v)


# --------------------------------------------------------------------------------------------- analyses vs exact DAEs

def dae_dc_exact():
    """DC operating point against the exact rational solution of an MNA assembled here: a bridge, controlled sources
    of all four kinds (unsymmetric, zero diagonals), and an inductor (a short at DC)"""
    for name, deck in (("bridge", "* bridge\nV1 1 0 DC 10\nR1 1 2 1k\nR2 1 3 2k\nR3 2 3 3k\nR4 2 0 4k\nR5 3 0 5k\n.op\n.end\n"),
                       ("controlled", "* EFGH\nV1 in 0 DC 1\nR1 in a 1k\nE1 b 0 a 0 100\nR2 b c 10k\nG1 0 c a 0 1m\nVS c d DC 0\nR3 d 0 1k\n"
                                      "F1 0 e VS 50\nR4 e 0 220\nH1 f 0 VS 1k\nR5 f g 47\nL1 g 0 1u\nR6 a 0 3.3k\n.op\n.end\n")):
        m = Mna(deck)
        x = m.G.LUsolve(m.s_dc)                          # exact rationals
        eng = treespice(deck)[0]["cols"]
        xm = [mp.mpf(v.p) / v.q for v in x]
        err = max(abs(mp.mpf(eng[nm][0]) - xv) for nm, xv in zip(m.names(), xm))
        kappa = cond_inf(mpm(m.G))
        report("dae", "dc_" + name, f"x vs exact rational (n={m.n}, kappa={float(kappa):.1e})", err, 10 * m.n * U * kappa * maxabs(xm))


def dae_ac_exact():
    """AC phasors against (G + jwC)^-1 s solved at 50 digits: a ladder with coupled inductors (K) and a VCCS"""
    deck = ("* ladder\nV1 in 0 DC 0 AC 1\nR1 in a 50\nL1 a b 10u\nC1 b 0 1n\nL2 b c 22u\nK1 L1 L2 0.6\nR2 c 0 75\n"
            "G1 0 d b 0 10m\nR3 d 0 1k\nC2 d 0 47p\n.ac dec 10 1k 1g\n.end\n")
    m = Mna(deck)
    eng = treespice(deck)[0]["cols"]
    G, C, s = mpm(m.G), mpm(m.C), mpm(m.s_ac)
    worst = (mp.mpf(-1), mp.mpf(0), mp.mpf(1))  # (err/bound, err, bound) at the worst frequency
    for k, f in enumerate(eng["frequency"]):
        w = 2 * mp.pi * mp.mpf(f.real)
        Y = G + mp.mpc(0, 1) * w * C
        x = mp.lu_solve(Y, s)
        e = max(abs(mp.mpc(eng[nm][k]) - x[i]) for i, nm in enumerate(m.names()) if nm.startswith("v("))
        b = 10 * m.n * U * cond_inf(Y) * maxabs(x)
        worst = max(worst, (e / b, e, b))
    report("dae", "ac_ladder_K", f"phasors, {len(eng['frequency'])} frequencies (worst of err/bound)", worst[1], worst[2])


def dae_pz_exact():
    """Poles of a fifth-order RLC filter: the roots of det(G + sC) in exact rational arithmetic, found by
    mpmath.polyroots at 50 digits, against the engine's .pz (QZ in double)"""
    deck = ("* LC ladder\nV1 in 0 DC 0 AC 1\nR1 in a 1k\nC1 a 0 1n\nL1 a b 100u\nC2 b 0 2n\nL2 b c 100u\nC3 c 0 1n\nR2 c 0 1k\n"
            ".pz in 0 c 0 vol pz\n.end\n")
    m = Mna(deck)
    s = sp.symbols("s")
    den = sp.Poly(sp.expand((m.G + s * m.C).det(method="berkowitz")), s)
    ref = mp.polyroots([mp.mpf(sp.Rational(c).p) / sp.Rational(c).q for c in den.all_coeffs()], maxsteps=200, extraprec=200)
    poles = [complex(float(a), float(b)) for a, b in re.findall(r"POLE\s+(\S+)\s+(\S+)", dump(deck, "pz"))]
    if len(poles) != len(ref):
        report("dae", "pz_ladder", f"pole count {len(poles)} vs {len(ref)}", 1, 0); return
    err = max(min(abs(mp.mpc(p) - r) / abs(r) for p in poles) for r in ref)
    # a simple eigenvalue's relative error is ~ its condition times u; these poles are well separated
    report("dae", "pz_ladder", f"{len(ref)} poles, relative", err, mp.mpf(1e-9))


def replay(m, ts, method_of):
    """the engine's transient re-run in 50 digits on its own accepted time points: (G + aC/h) x1 = s1 + (aC/h) x0 + [d0],
    d = C x' the reactive term carried between steps (a = 1 backward Euler, 2 trapezoidal with d0 added); Gear 2 at
    variable step: C x1' = C (a0 x1 + a1 x0 + a2 x-1), the slope of the parabola through the last three points"""
    G, C = mpm(m.G), mpm(m.C)
    x = mp.lu_solve(G, m.s(mp.mpf(0)))                   # the DC operating point at t = 0
    d = mp.matrix(m.n, 1)
    xs = [x]
    xm, hp = x, None                                     # the point before x and the step to x (Gear 2)
    for k in range(1, len(ts)):
        t0, t1 = mp.mpf(ts[k - 1]), mp.mpf(ts[k])
        meth = method_of(k)
        if meth == "cold":                                 # 100 backward-Euler micro-steps (the cold-start sweep)
            hs = (t1 - t0) / 100
            for j in range(1, 101):
                xn = mp.lu_solve(G + C / hs, m.s(t0 + j * hs) + (C / hs) * x)
                d, xm, x, hp = (C / hs) * (xn - x), x, xn, hs
        elif meth == "be":
            h = t1 - t0
            xn = mp.lu_solve(G + C / h, m.s(t1) + (C / h) * x)
            d, xm, x, hp = (C / h) * (xn - x), x, xn, h
        elif meth == "gear":
            h = t1 - t0
            a0, a1, a2 = 1 / h + 1 / (h + hp), -(h + hp) / (h * hp), h / (hp * (h + hp))
            xn = mp.lu_solve(G + a0 * C, m.s(t1) - C * (a1 * x + a2 * xm))
            d, xm, x, hp = C * (a0 * xn + a1 * x + a2 * xm), x, xn, h
        else:
            h = t1 - t0
            xn = mp.lu_solve(G + 2 * C / h, m.s(t1) + (2 * C / h) * x + d)
            d, xm, x, hp = (2 * C / h) * (xn - x) - d, x, xn, h
        xs.append(x)
    return xs


def dae_tran_replay():
    """Transient against the integration formulas themselves: SPICE's companion models (backward Euler on the cold
    start, the run's second step and every step after a corner, trapezoidal otherwise) replayed at 50 digits on the
    engine's accepted points. Cases: the Chapter 1 RC, a series RLC (inductor companion), a capacitor across a
    voltage source (a loop of capacitors and a source: index two, the source current is C dV/dt); each again with
    .options method=gear (variable-step Gear 2 in place of trapezoidal)"""
    decks = (("rc", "* RC\nV1 in 0 PULSE(0 1 0 1n 1n 10u 20u)\nR1 in out 1k\nC1 out 0 1n\n.tran 0.1u 5u\n.end\n"),
             ("rlc", "* RLC\nV1 in 0 PULSE(0 1 10n 2n 2n 400n 1u)\nR1 in a 20\nL1 a b 1u\nC1 b 0 1n\nR2 b 0 10k\n.tran 5n 2u\n.end\n"),
             ("cv_loop", "* C across V\nV1 in 0 PULSE(0 1 0 10n 10n 100n 300n)\nC1 in 0 1n\nR1 in 0 1k\n.tran 2n 600n\n.end\n"))
    for name, deck in decks + tuple((n + "_gear", d.replace(".tran", ".options method=gear\n.tran")) for n, d in decks):
        m = Mna(deck)
        eng = treespice(deck)[0]["cols"]
        ts = eng["time"]
        corners = m.corners(ts[-1])
        hmax = max(b - a for a, b in zip(ts, ts[1:]))
        landed = lambda t: any(abs(t - c) <= 1e-9 * hmax for c in corners if c > 0)
        second = "gear" if name.endswith("_gear") else "trap"
        method_of = lambda k: "cold" if k == 1 else "be" if k == 2 or landed(ts[k - 1]) else second
        xs = replay(m, ts, method_of)
        names = [nm for nm in m.names() if nm in eng]
        err = max(abs(mp.mpf(eng[nm][k]) - xs[k][m.names().index(nm)]) for k in range(len(ts)) for nm in names)
        h = mp.mpf(min(b - a for a, b in zip(ts, ts[1:])))
        kappa = max(cond_inf(mpm(m.G) + a * mpm(m.C) / mp.mpf(hh)) for a in (1, 2) for hh in (h, hmax))
        scale = max(maxabs([xs[k][m.names().index(nm)] for k in range(len(ts))]) for nm in names)
        # each step rounds at ~kappa*u; TR and BE are A-stable, so N steps accumulate at most linearly
        report("dae", "tran_" + name, f"{len(ts)} points, {','.join(names)}", err, 10 * len(ts) * U * kappa * scale)


def dae_noise_rc():
    """Thermal noise of R seen through C: S_v = 4kTR / (1 + (wRC)^2) at 300.15 K, against the engine's .noise"""
    deck = "* RC noise\nV1 in 0 DC 0 AC 1\nR1 in out 1k\nC1 out 0 1n\n.noise v(out) V1 dec 10 1k 1g\n.end\n"
    out = dump(deck, "noise")
    rows = [list(map(float, l.split())) for l in out.splitlines() if re.match(r"^[0-9.eE+-]+\s+[0-9.eE+-]+\s+[0-9.eE+-]+\s*$", l)]
    R, C = mp.mpf(1000), mp.mpf("1e-9")
    err = mp.mpf(0)
    for f, on, _ in rows:
        w = 2 * mp.pi * f
        ref = mp.sqrt(4 * K_B * T_NOM * R / (1 + (w * R * C) ** 2))  # V/sqrt(Hz), as ngspice's onoise
        err = max(err, abs(on - ref) / ref)
    report("dae", "noise_rc", f"onoise, {len(rows)} frequencies, relative", err, mp.mpf(1e-12) if rows else mp.mpf(-1))


# --------------------------------------------------------------------------------------------- the contraction solver

def mna_like(rng, n, nsrc, nvcvs, gmin_path=False):
    """a circuit-shaped matrix: a random connected conductance graph (1e-12..1e3 S), voltage-source border rows (zero
    diagonals) and VCVS rows with gain 1e5; gmin_path makes one node reach ground only through 1e-12 S"""
    nodes = n - nsrc - nvcvs
    e = {}
    def g(a, b, v):
        for i, j, s in ((a, a, 1), (b, b, 1), (a, b, -1), (b, a, -1)):
            if i is not None and j is not None: e[(i, j)] = e.get((i, j), 0.0) + s * v
    for k in range(1, nodes):  # a spanning tree, then extra branches
        g(k, rng.randrange(k), 10 ** rng.uniform(-6, 3))
    for _ in range(2 * nodes):
        a, b = rng.randrange(nodes), rng.randrange(nodes)
        if a != b: g(a, b, 10 ** rng.uniform(-12, 3))
    for k in range(nodes): g(k, None, 1e-12 if gmin_path and k == nodes - 1 else 10 ** rng.uniform(-9, -3) if rng.random() < 0.2 else 0.0)
    row = nodes
    driven = rng.sample(range(nodes), nsrc + nvcvs)  # one voltage constraint per node: two would be a source loop
    for a in driven[:nsrc]:
        e[(a, row)] = e.get((a, row), 0.0) + 1.0; e[(row, a)] = e.get((row, a), 0.0) + 1.0; row += 1
    for a in driven[nsrc:]:
        c = rng.choice([k for k in range(nodes) if k != a])
        e[(a, row)] = e.get((a, row), 0.0) + 1.0; e[(row, a)] = e.get((row, a), 0.0) + 1.0
        e[(row, c)] = e.get((row, c), 0.0) - 1e5; row += 1
    return e


def solve_engine(n, e, b, refine):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(f"{n} {len(e)}\n" + "".join(f"{i} {j} {v:.17g}\n" for (i, j), v in e.items()) + "".join(f"{v:.17g}\n" for v in b))
    try:
        p = subprocess.run([os.path.join(BUILD, "solve_triplets"), f.name, str(refine)], capture_output=True, text=True, timeout=300)
    finally: os.unlink(f.name)
    lines = p.stdout.split("\n")
    ok, rank_def = int(lines[0].split()[1]), int(lines[0].split()[3])
    return ok, rank_def, [float(x) for x in lines[1:] if x.strip()]


def solver_cases():
    """The contraction solver against mpmath's exact solve of the same double-precision matrix. Pass: componentwise
    backward error (Oettli-Prager) of a few u with the engine's two refinement steps, and forward error within
    Skeel's bound cond(A,x)*omega. Refinement 0 (the factorization alone) is reported for its growth."""
    rng = random.Random(20261026)
    cases = [("grid_unsym", None), ("mna_80", (80, 4, 3, False)), ("mna_120_gmin", (120, 6, 4, True))]
    for name, spec in cases:
        if spec is None:  # a 10 x 10 grid, unsymmetric (convection-like: -1 one way, -1.3 the other)
            k, e = 10, {}
            for i in range(k * k):
                e[(i, i)] = 4.0 + 0.01 * i
                for j in (i - 1, i + 1, i - k, i + k):
                    if 0 <= j < k * k and (abs(i - j) == k or i // k == j // k): e[(i, j)] = -1.0 - (0.3 if j > i else 0.0)
            n = k * k
        else:
            n, ns, nv, gp = spec; e = mna_like(rng, n, ns, nv, gp)
        b = [rng.uniform(-1, 1) for _ in range(n)]
        A = mp.matrix(n, n)
        for (i, j), v in e.items(): A[i, j] = mp.mpf(v)
        bm = mp.matrix([mp.mpf(v) for v in b])
        x = mp.lu_solve(A, bm)                                        # exact to 50 digits
        Ainv = mp.inverse(A)
        absA = mp.matrix([[abs(A[i, j]) for j in range(n)] for i in range(n)])
        absAinv = mp.matrix([[abs(Ainv[i, j]) for j in range(n)] for i in range(n)])
        skeel = mp.norm(absAinv * (absA * mp.matrix([abs(v) for v in x])), mp.inf) / mp.norm(x, mp.inf)
        for refine in (0, 2):
            ok, rank_def, xe = solve_engine(n, e, b, refine)
            if not ok or len(xe) != n:
                report("solver", f"{name}_r{refine}", "solve failed", 1, 0); continue
            xem = mp.matrix([mp.mpf(v) for v in xe])
            r = bm - A * xem                                          # exact residual
            den = absA * mp.matrix([abs(v) for v in xem]) + mp.matrix([abs(v) for v in bm])
            omega = max(abs(r[i]) / den[i] for i in range(n) if den[i] > 0)
            fwd = mp.norm(xem - x, mp.inf) / mp.norm(x, mp.inf)
            if refine == 2:
                report("solver", f"{name}_r2", f"backward error omega (n={n}, skeel={float(skeel):.1e})", omega, 4 * U)
                report("solver", f"{name}_r2", "forward error vs Skeel bound", fwd, 2 * skeel * max(omega, mp.mpf(U)) + 4 * U)
            else:
                print(f"  [info] {name+'_r0':28s} factorization alone: omega {float(omega):.2e}, forward {float(fwd):.2e}")
    # a loop of voltage sources is structurally singular: the solver must say so (fail closed), not return numbers
    e = {(0, 0): 1e-3, (0, 1): 1.0, (1, 0): 1.0, (0, 2): 1.0, (2, 0): 1.0}
    ok, rank_def, xe = solve_engine(3, e, [0.0, 1.0, 2.0], 2)
    report("solver", "vs_loop_singular", "refused (ok=0 or rank deficient)", 0 if (not ok or rank_def) else 1, 0)


# --------------------------------------------------------------------------------------------- device equations

def junction(v, IS, nvt, spice=True):
    """Shockley's I = IS (e^(v/nvt) - 1) [@sze2007], or with spice=True as SPICE evaluates it: below -3 nvt the
    exponential is replaced by -IS (1 + (3 nvt / (e v))^3) (ngspice dioload/bjtload), smooth to second order there"""
    if not spice or v >= -3 * nvt: return IS * (sp.exp(v / nvt) - 1)
    return -IS * (1 + (3 * nvt / (sp.E * v)) ** 3)


def paper_gap(case, what, spice_val, paper_val):
    """how far SPICE's model departs from the paper's equation at this point (information, not a check)"""
    gap = abs(spice_val - paper_val) / max(abs(paper_val), mp.mpf("1e-30"))
    print(f"  [info] {case:28s} SPICE's {what} vs the paper's: relative {float(gap):.2e}")


def device_diode():
    """Shockley diode with SPICE's junction GMIN, I = IS (exp(v/(N Vt)) - 1) + GMIN v, and its small-signal admittance
    gd + jw (Cj + TT gd), with Cj from the depletion law and SPICE's linear extension above FC*VJ [@sze2007, ngspice]"""
    IS, N, TT, CJO, VJ, M, FC = mp.mpf("1e-14"), mp.mpf("1.5"), mp.mpf("1e-9"), mp.mpf("2e-12"), mp.mpf("0.8"), mp.mpf("0.4"), mp.mpf("0.5")
    v = sp.symbols("v")
    i_expr = sp.Float(IS, 50) * (sp.exp(v / (sp.Float(N, 50) * sp.Float(VT, 50))) - 1) + sp.Float(GMIN, 50) * v
    gd_expr = sp.diff(i_expr, v)
    for vd in ("-2", "0.3", "0.65"):
        f = 1e3
        deck = (TIGHT + f"* diode\nV1 a 0 DC {vd} AC 1\nD1 a 0 dm\n"
                f".model dm D is=1e-14 n=1.5 tt=1n cjo=2p vj=0.8 m=0.4 fc=0.5\n.ac lin 1 {f:g} {f:g}\n.end\n")
        cols = treespice(deck)
        ac = next(p for p in cols if p["name"].lower().startswith("ac"))["cols"]
        V = mp.mpf(vd)
        gd = mp.mpf(str(gd_expr.subs(v, sp.Float(V, 50)).evalf(50)))
        cj = CJO * (1 - V / VJ) ** (-M) if V < FC * VJ else CJO / (1 - FC) ** (1 + M) * (1 - FC * (1 + M) + M * V / VJ)
        y_ref = mp.mpc(gd, 2 * mp.pi * f * (cj + TT * gd))
        y_eng = -mp.mpc(ac["i(v1)"][0])                                  # i(v1) flows into V1's + terminal
        report("device", f"diode_y_{vd}V", "admittance gd + jwC (relative)", abs(y_eng - y_ref) / abs(y_ref), mp.mpf(1e-9))
        op = treespice(TIGHT + f"* diode op\nV1 a 0 DC {vd}\nD1 a 0 dm\n.model dm D is=1e-14 n=1.5\n.op\n.end\n")[0]["cols"]
        Vs, nvt = sp.Float(V, 50), sp.Float(N * VT, 50)
        i_ref = mp.mpf(str((junction(Vs, sp.Float(IS, 50), nvt) + sp.Float(GMIN, 50) * Vs).evalf(50)))
        report("device", f"diode_i_{vd}V", "DC current, SPICE's law (relative)", abs(-mp.mpf(op["i(v1)"][0]) - i_ref) / abs(i_ref), mp.mpf(1e-10))
        paper_gap(f"diode_i_{vd}V", "junction current", i_ref, mp.mpf(str(i_expr.subs(v, Vs).evalf(50))))


def device_mos1():
    """Shichman-Hodges (MOS level 1) [@shichman1968] as SPICE writes it: beta = KP W/L, Ids = beta (vgs-VTO-vds/2) vds
    (1+LAMBDA vds) in the linear region, beta/2 (vgs-VTO)^2 (1+LAMBDA vds) in saturation, 0 below threshold; plus the
    drain-bulk junction (IS 1e-14, GMIN). gm and gds from SymPy derivatives against the engine's AC transconductances"""
    KP, W, L, VTO, LAM, ISJ = mp.mpf("100e-6"), mp.mpf("10e-6"), mp.mpf("1e-6"), mp.mpf("0.7"), mp.mpf("0.02"), mp.mpf("1e-14")
    vg, vd = sp.symbols("vg vd")
    beta = sp.Float(KP * W / L, 50)
    def ids_expr(region):
        vgst = vg - sp.Float(VTO, 50)
        core = {"cut": sp.Integer(0), "lin": beta * (vgst - vd / 2) * vd, "sat": beta / 2 * vgst ** 2}[region]
        return core * (1 + sp.Float(LAM, 50) * vd) if region != "cut" else core
    jbd = lambda: -(sp.Float(ISJ, 50) * (sp.exp(-vd / sp.Float(VT, 50)) - 1) + sp.Float(GMIN, 50) * (-vd))  # into the drain
    model = ".model nm NMOS level=1 vto=0.7 kp=100u lambda=0.02\n"
    for vgs, vds, region in (("0.5", "1", "cut"), ("1.5", "0.3", "lin"), ("1.5", "2", "sat")):
        sub = {vg: sp.Float(mp.mpf(vgs), 50), vd: sp.Float(mp.mpf(vds), 50)}
        i_total = ids_expr(region) + jbd()
        i_ref = mp.mpf(str(i_total.subs(sub).evalf(50)))
        op = treespice(TIGHT + f"* mos op\nVD d 0 DC {vds}\nVG g 0 DC {vgs}\nM1 d g 0 0 nm W=10u L=1u\n{model}.op\n.end\n")[0]["cols"]
        i_eng = -mp.mpf(op["i(vd)"][0])
        report("device", f"mos1_ids_{region}", "drain current (relative, abs floor 1e-21 A)", abs(i_eng - i_ref) / max(abs(i_ref), mp.mpf("1e-9")), mp.mpf(1e-10))
        for port, sym in (("VG", vg), ("VD", vd)):
            g_ref = mp.mpf(str(sp.diff(i_total, sym).subs(sub).evalf(50)))
            ac_g, ac_d = ("AC 1", "") if port == "VG" else ("", "AC 1")
            cols = treespice(TIGHT + f"* mos ac\nVD d 0 DC {vds} {ac_d}\nVG g 0 DC {vgs} {ac_g}\nM1 d g 0 0 nm W=10u L=1u\n{model}.ac lin 1 1 1\n.end\n")
            ac = next(p for p in cols if p["name"].lower().startswith("ac"))["cols"]
            g_eng = -mp.mpc(ac["i(vd)"][0]).real
            report("device", f"mos1_{'gm' if port == 'VG' else 'gds'}_{region}", "small-signal (relative, floor 1e-15 S)",
                   abs(g_eng - g_ref) / max(abs(g_ref), mp.mpf("1e-6")), mp.mpf(1e-9))


def device_bjt():
    """Ebers-Moll in Gummel-Poon form [@ebers1954; @gummel1970] with qb = 1 (no VAF, IKF) and SPICE's GMIN on the
    non-ideal base terms: Ic = IS (e^vbe/Vt - e^vbc/Vt) - IS/BR (e^vbc/Vt - 1) - GMIN vbc, Ib = IS/BF (e^vbe/Vt - 1)
    + IS/BR (e^vbc/Vt - 1) + GMIN (vbe + vbc). dIc/dVb against the engine's AC"""
    IS, BF, BR = mp.mpf("1e-16"), mp.mpf(100), mp.mpf(2)
    vb, vc = sp.symbols("vb vc")
    vt, gmin = sp.Float(VT, 50), sp.Float(GMIN, 50)
    model = ".model qn NPN is=1e-16 bf=100 br=2\n"
    for vbs, vcs, region in (("0.7", "2", "active"), ("0.75", "0.1", "saturated")):
        sub = {vb: sp.Float(mp.mpf(vbs), 50), vc: sp.Float(mp.mpf(vcs), 50)}
        vbc_num = sub[vb] - sub[vc]
        spice = bool(vbc_num < -3 * vt)  # which branch of the junction law applies at this point (fixed, so diff() is exact)
        cbe = sp.Float(IS, 50) * (sp.exp(vb / vt) - 1)
        cbc = -sp.Float(IS, 50) * (1 + (3 * vt / (sp.E * (vb - vc))) ** 3) if spice else sp.Float(IS, 50) * (sp.exp((vb - vc) / vt) - 1)
        # SPICE's vertical NPN also has a collector-substrate junction (substrate at ground): with ISS = 0 it carries
        # only GMIN v_sub, v_sub = -vc, flowing from the substrate into the collector
        ic = (cbe - cbc) - cbc / sp.Float(BR, 50) - gmin * (vb - vc) - gmin * (-vc)
        ib = cbe / sp.Float(BF, 50) + gmin * vb + cbc / sp.Float(BR, 50) + gmin * (vb - vc)
        ic_paper = sp.Float(IS, 50) * (sp.exp(vb / vt) - sp.exp((vb - vc) / vt)) - sp.Float(IS / BR, 50) * (sp.exp((vb - vc) / vt) - 1)
        op = treespice(TIGHT + f"* bjt op\nVC c 0 DC {vcs}\nVB b 0 DC {vbs}\nQ1 c b 0 qn\n{model}.op\n.end\n")[0]["cols"]
        for nm, expr, col in (("ic", ic, "i(vc)"), ("ib", ib, "i(vb)")):
            ref = mp.mpf(str(expr.subs(sub).evalf(50)))
            report("device", f"bjt_{nm}_{region}", "terminal current, SPICE's model (relative)", abs(-mp.mpf(op[col][0]) - ref) / abs(ref), mp.mpf(1e-10))
        paper_gap(f"bjt_ic_{region}", "Ic (GMIN, substrate, reverse law)", mp.mpf(str(ic.subs(sub).evalf(50))), mp.mpf(str(ic_paper.subs(sub).evalf(50))))
        g_ref = mp.mpf(str(sp.diff(ic, vb).subs(sub).evalf(50)))
        cols = treespice(TIGHT + f"* bjt ac\nVC c 0 DC {vcs}\nVB b 0 DC {vbs} AC 1\nQ1 c b 0 qn\n{model}.ac lin 1 1 1\n.end\n")
        ac = next(p for p in cols if p["name"].lower().startswith("ac"))["cols"]
        report("device", f"bjt_dic_dvb_{region}", "dIc/dVb (relative)", abs(-mp.mpc(ac["i(vc)"][0]).real - g_ref) / abs(g_ref), mp.mpf(1e-9))


AREAS = {"dae": [dae_dc_exact, dae_ac_exact, dae_pz_exact, dae_tran_replay, dae_noise_rc],
         "solver": [solver_cases], "device": [device_diode, device_mos1, device_bjt]}


def main():
    global BUILD
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", default=BUILD, help="directory holding treespice, acutesim_engine_dump, solve_triplets")
    ap.add_argument("areas", nargs="*", help="any of " + ", ".join(AREAS) + " (default: all)")
    a = ap.parse_args()
    if set(a.areas) - set(AREAS): ap.error(f"unknown area(s): {', '.join(sorted(set(a.areas) - set(AREAS)))}")  # argparse can't combine choices with an empty nargs="*"
    BUILD = os.path.abspath(a.build)
    for area in a.areas or AREAS:
        print(f"{area}:")
        for fn in AREAS[area]:
            try: fn()
            except Exception as e:  # a case that cannot run is a failure, never a skip
                results.append((area, fn.__name__, False)); print(f"  [FAIL] {fn.__name__}: {type(e).__name__}: {e}")
    bad = [r for r in results if not r[2]]
    print(f"{len(results) - len(bad)} passed, {len(bad)} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
