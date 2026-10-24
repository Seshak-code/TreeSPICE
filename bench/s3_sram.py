#!/usr/bin/env python3
"""S3 gate netlist (docs/scale-s3.md): a 6T SRAM written as nested subcircuits, a write then a read of one word.

    python3 bench/s3_sram.py --banks B --rows R --cols C [--width W] [--hier] > sram.cir

Cells (6T, 2 fF storage nodes) start at q = 1 by the cell body's .ic. Bank 0 is its own definition: its word line 0
pulses twice (write at 0.2-1.2 ns, read at 2.0-3.0 ns) and its first W columns' drivers pull bl low during the write
(writing 0), then return to the precharge level; every bit line hangs on 1.8 V through 200 ohm and 20 fF. The other
banks hold: word lines at 0, bit lines precharged. B * R * C cells; .options hier runs the quotient circuit.
"""
import argparse
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--banks", type=int, required=True)
ap.add_argument("--rows", type=int, required=True)
ap.add_argument("--cols", type=int, required=True)
ap.add_argument("--width", type=int, default=8)
ap.add_argument("--hier", action="store_true")
a = ap.parse_args()
C, R, W = a.cols, a.rows, min(a.width, a.cols)
out = sys.stdout
w = out.write

w(f"* S3 gate: 6T SRAM, {a.banks} banks x {R} rows x {C} columns = {a.banks * R * C} cells\n")
w(".subckt cell bl blb wl vdd\n"
  "M1 q qb 0 0 nm W=1u L=0.18u\nM2 q qb vdd vdd pm W=1u L=0.18u\n"
  "M3 qb q 0 0 nm W=1u L=0.18u\nM4 qb q vdd vdd pm W=1u L=0.18u\n"
  "M5 bl wl q 0 nm W=1u L=0.18u\nM6 blb wl qb 0 nm W=1u L=0.18u\n"
  "Cq q 0 2f\nCqb qb 0 2f\n.ic v(q)=1.8 v(qb)=0\n.ends\n")
bls = " ".join(f"bl{c}" for c in range(C)) + " " + " ".join(f"blb{c}" for c in range(C))
w(f".subckt row wl vdd {bls}\n")
for c in range(C):
    w(f"XC{c} bl{c} blb{c} wl vdd cell\n")
w(".ends\n")


def bank(name, written):
    w(f".subckt {name} vdd\n")
    for r in range(R):
        w(f"XR{r} wl{r} vdd {bls} row\n")
    for r in range(R):
        if written and r == 0:
            w("VW0 wl0 0 PULSE(0 1.8 0.2n 0.05n 0.05n 1n 1.8n)\n")  # write, then read
        else:
            w(f"VW{r} wl{r} 0 0\n")
    for c in range(C):
        for side in ("bl", "blb"):
            drive = written and c < W and side == "bl"
            src = "PWL(0 1.8 0.15n 1.8 0.2n 0 1.2n 0 1.3n 1.8)" if drive else "1.8"
            w(f"VP{side}{c} p{side}{c} 0 {src}\nRP{side}{c} p{side}{c} {side}{c} 200\nCB{side}{c} {side}{c} 0 20f\n")
    w(".ends\n")


bank("bankw", True)
if a.banks > 1:
    bank("bank", False)
w(".model nm nmos level=1 vto=0.5 kp=200u cgso=1e-10 cgdo=1e-10\n"
  ".model pm pmos level=1 vto=-0.5 kp=80u cgso=1e-10 cgdo=1e-10\nVDD vdd 0 1.8\nXB0 vdd bankw\n")
for b in range(1, a.banks):
    w(f"XB{b} vdd bank\n")
probes = ["xb0.xr0.xc0.q", "xb0.xr0.xc0.qb", f"xb0.xr0.xc{C - 1}.q", "xb0.xr1.xc0.q"]
if a.banks > 1:
    probes.append(f"xb{a.banks - 1}.xr{R - 1}.xc{C - 1}.q")
w(".save " + " ".join(f"v({p})" for p in probes) + " i(vdd)\n")
if a.hier:
    w(".options hier\n")
w(".tran 0.02n 3.2n\n.end\n")
