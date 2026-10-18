#!/usr/bin/env python3
"""Generate a Verilog-A VBIC (ngspice 47's vbic, levels 4 and 9) from ngspice's own C:
vbic_4T_et_cf_t (vbictemp.c: TNOM -> device temperature) and the value statements of
vbic_4T_et_cf_fj (vbicload.c: currents and charges; its derivative statements are dropped,
OpenVAF differentiates), wired as VBICload stamps them.
usage: vbic_gen.py <ngspice-47/src/spicelib/devices/vbic> <out.va>"""
import json, re, sys

def vbic_table(D):
    defs = open(D + "/vbicdefs.h").read()
    m = re.search(r"typedef struct sVBICmodel\s*\{(.*?)\}\s*VBICmodel;", defs, re.S)
    body = m.group(1)
    fields = re.findall(r"^\s*double\s+(VBIC\w+)\s*;", body, re.M)
    start = fields.index("VBICtnom")
    pf = fields[start:]
    # parameter names: IOP("name", VBIC_MOD_X, IF_REAL, ...)
    parm = open(D + "/vbic.c").read()
    names = {}
    for nm, idn in re.findall(r'IOP[A-Z]*\(\s*"(\w+)"\s*,\s*(VBIC_MOD_\w+)', parm):
        names.setdefault(idn, []).append(nm)
    mpar = open(D + "/vbicmpar.c").read()
    id2field = {}
    for idn, fld in re.findall(r"case\s+(VBIC_MOD_\w+)\s*:\s*\n\s*mods?->(VBIC\w+)\s*=", mpar):
        id2field.setdefault(fld, idn)
    setup = open(D + "/vbicsetup.c").read()
    defaults = dict(re.findall(r"if\s*\(\s*!\s*model->(VBIC\w+)Given\s*\)\s*\{?\s*model->\1\s*=\s*([^;]+);", setup))
    out = []
    for k, f in enumerate(pf):
        idn = id2field.get(f)
        out.append((k, f, names.get(idn, []), defaults.get(f, "?")))
    return out

D, OUT = sys.argv[1:3]
table = vbic_table(D)
src = open(D + "/vbicload.c").read()
tsrc = open(D + "/vbictemp.c").read()


def body(text, name):
    """Statements of the C function `name` (its last definition), declarations dropped."""
    i = text.rindex("int " + name + "(")
    i = text.index("{", text.index(")\n{", i))
    depth, j = 0, i
    while True:
        if text[j] == "{": depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0: break
        j += 1
    lines = [l.strip() for l in text[i + 1:j].splitlines()]
    return [l for l in lines if l and not l.startswith("double ") and not l.startswith("/*") and not l.startswith("return")]


def va(lines, keep=lambda lhs: True, rename=None):
    """C statements -> Verilog-A; statements whose target fails `keep` are dropped."""
    out, assigned = [], set()
    def cond(c):
        c = re.sub(r"\(\*(\w+)\)", r"\1", c)
        return rename(c, expr=True) if rename else c
    for l in lines:
        if l.startswith("if("):
            out.append("if (" + cond(l[3:-2]) + ") begin"); continue
        if l == "}else{": out.append("end else begin"); continue
        if l.startswith("}else if(") and l.endswith("){"): out.append("end else if (" + cond(l[9:-2]) + ") begin"); continue
        if l == "}": out.append("end"); continue
        l = re.sub(r"^\(\*(\w+)\)", r"\1", l)              # (*Itzr)=... -> Itzr=...
        m = re.match(r"^\*?(\w+(?:\[\d+\])?)\s*=\s*(.*);$", l)
        if not m: raise SystemExit("unhandled C: " + l)
        lhs, rhs = m.groups()
        if not keep(lhs): continue
        rhs = re.sub(r"\(\*(\w+)\)", r"\1", rhs)        # (*SCALE) -> SCALE
        rhs = re.sub(r"\blog\(", "ln(", rhs)            # C log is natural: VA ln
        if rename: lhs, rhs = rename(lhs), rename(rhs, expr=True)
        out.append(f"{lhs} = {rhs};")
        assigned.add(lhs)
    return out, assigned


# ── parameters: p[k] <-> ngspice names (the model struct's order from VBICtnom) ──
pname = {}
for k, field, names, dflt in table[:108]:
    pname[k] = names[0] if names else f"p{k}"

# The model function reads p[k] (temperature-mapped where vbictemp maps them): local reals p_k.
def prename(s, expr=False):
    return re.sub(r"\bp\[(\d+)\]", r"p_\1", s)

# vbic_4T_et_cf_t: pnom[k] are the parameters, p[k] the mapped locals; TAMB the device temperature [C]
def trename(s, expr=False):
    s = re.sub(r"\bpnom\[(\d+)\]", lambda m: pname[int(m.group(1))], s)
    s = re.sub(r"\bp\[(\d+)\]", r"p_\1", s)
    return s.replace("*TAMB", "TAMB").replace("(*TAMB)", "TAMB")

tlines, tvars = va(body(tsrc, "vbic_4T_et_cf_t"), rename=trename)
# the model function: values only (derivative targets carry an underscore: Ibe_Vbei, xvar1_rT)
flines, fvars = va(body(src, "vbic_4T_et_cf_fj"), keep=lambda lhs: "_" not in lhs, rename=prename)

reserved = {"type"}
params = []
for k, field, names, dflt in table[:108]:
    n = pname[k]
    d = str(dflt).strip().rstrip("f")
    if d == "?": d = "0.0"
    params.append(f"    parameter real {n} = {d};")
# a p_k the model function reads that neither the temperature function nor VBICload sets is the parameter itself
read = set(re.findall(r"\bp_(\d+)\b", " ".join(flines)))
set_ = set(re.findall(r"^(p_\d+) =", "\n".join(tlines), re.M)) | {"p_0"}
unset = sorted(int(k) for k in read if f"p_{k}" not in set_)
tlines += [f"p_{k} = {pname[k]};" for k in unset]
locals_ = sorted((tvars | fvars | {f"p_{k}" for k in range(108)}) - {pname[k] for k in range(108)})

va_text = f"""// VBIC as ngspice 47 runs it (levels 4 and 9), generated by tests/models/vbic_gen.py from ngspice's
// vbictemp.c and vbicload.c (copyright 1990 Regents of the University of California; model author 1995
// Colin McAndrew, Motorola; Spice3 implementation 2003 Dietmar Warning, DAnalyse; ngspice's modified BSD).
`include "disciplines.vams"
`include "constants.vams"
`define SELFHEAT ((selft < 0.0 || selft == 1.0) && rth > 0.0)  // vbicsetup.c

module vbic_ng(c, b, e, s, t);
    inout c, b, e, s, t;
    electrical c, b, e, s, t;
    electrical cx, ci, bx, bi, ei, bp, si, xf1, xf2;
{chr(10).join(params)}
    parameter real type = 1.0;                       // +1 NPN, -1 PNP
    parameter real selft = -1.0;                     // self-heating switch (-1: not given)
    (* type="instance" *) parameter real area = 1.0;
    (* type="instance" *) parameter real dtemp_i = 0.0;  // instance DTEMP [K]

    real {", ".join(locals_)};
    real Vbei, Vbex, Vbci, Vbcx, Vbep, Vrci, Vrbi, Vrbp, Vbcp, Vbe, Vbc, Vrcx, Vrbx, Vre, Vrs, Vrth, Vcei, Vcep;
    real Vxf1, Vxf2, SCALE, gmin, Itxf, Iciei, TAMB;

    analog begin
        gmin = $simparam("gmin", 1e-12);
        TAMB = $temperature - `P_CELSIUS0 + dtemp_i;
        // vbictemp.c: the parameters at the device temperature
{chr(10).join("        " + l for l in tlines)}
        // vbicload.c: p[0] is the device temperature, the mapped values replace the nominal ones
        p_0 = TAMB + {pname[105]};
        if ($param_given(rth) && p_84 < 1e-12) p_84 = 1e-12;     // vbicsetup.c: cth floor when rth is given
        SCALE = area;

        Vbei = type * V(bi, ei); Vbex = type * V(bx, ei); Vbci = type * V(bi, ci); Vbcx = type * V(bi, cx);
        Vbep = type * V(bx, bp); Vrci = type * V(cx, ci); Vrbi = type * V(bx, bi); Vrbp = type * V(bp, cx);
        Vbcp = type * V(si, bp); Vbe = type * V(b, e); Vbc = type * V(b, c); Vrcx = type * V(c, cx);
        Vrbx = type * V(b, bx); Vre = type * V(e, ei); Vrs = type * V(s, si);
        Vrth = `SELFHEAT ? V(t) : 0.0;
        Vcei = Vbei - Vbci; Vcep = Vbep - Vbcp;
        Vxf1 = V(xf1); Vxf2 = V(xf2);

{chr(10).join("        " + l for l in flines)}

        // excess phase (TD): Itzf through a 2-pole delay network on xf1, xf2 (vbicload.c)
        if (td > 0.0) begin  // vbicsetup.c: excess phase
            I(xf1) <+ Vxf2 - Itzf + ddt(td * Vxf1);
            I(xf2) <+ Vxf2 - Vxf1 + ddt(td * Vxf2 / 3.0);
            Itxf = Vxf2;
        end else begin
            I(xf1) <+ Vxf1;
            I(xf2) <+ Vxf2;
            Itxf = Itzf;
        end
        Iciei = Itxf - Itzr;

        I(bi, ei) <+ type * (Ibe + gmin * Vbei + ddt(Qbe));
        I(bx, ei) <+ type * (Ibex + gmin * Vbex + ddt(Qbex));
        I(ci, ei) <+ type * Iciei;
        I(bi, ci) <+ type * (Ibc + gmin * Vbci + ddt(Qbc));
        I(bi, cx) <+ type * ddt(Qbcx);
        I(bx, bp) <+ type * (Ibep + gmin * Vbep + ddt(Qbep));
        I(cx, ci) <+ type * (Irci + gmin * (Vrci + Vbci + Vbcx));
        I(bx, bi) <+ type * Irbi;
        I(bp, cx) <+ type * Irbp;
        I(si, bp) <+ type * (Ibcp + gmin * Vbcp + ddt(Qbcp));
        I(bx, si) <+ type * Iccp;
        I(b, e) <+ type * ddt(Qbeo);
        I(b, c) <+ type * ddt(Qbco);
        if (rcx > 0.0) I(c, cx) <+ type * Ircx; else V(c, cx) <+ 0.0;
        if (rbx > 0.0) I(b, bx) <+ type * Irbx; else V(b, bx) <+ 0.0;
        if (re > 0.0) I(e, ei) <+ type * Ire; else V(e, ei) <+ 0.0;
        if (rs > 0.0) I(s, si) <+ type * Irs; else V(s, si) <+ 0.0;
        // Ith is minus the dissipated power: VBICload injects -Ith into the thermal node. Without self-heating
        // nothing in the model loads it (ngspice: v(t) = 0); a GMIN keeps a floating t from leaving AC singular.
        if (`SELFHEAT) I(t) <+ Irth + Ith + ddt(Qcth); else I(t) <+ gmin * V(t);
    end
endmodule
"""
open(OUT, "w").write(va_text)
print(f"{OUT}: {len(tlines)} temperature, {len(flines)} model statements, {len(params)} parameters; p read unset: {unset}")
