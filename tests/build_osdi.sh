#!/bin/sh
# Compile the Verilog-A models the ngspice harness loads through OSDI into datasets/osdi/
# (git-ignored, like the other datasets), plus IHP SG13G2's PSP103 when datasets/pdk/ihp
# holds that PDK. Needs OpenVAF-reloaded and its source tree,
# whose integration_tests/ hold the reference .va models:
#   OPENVAF_DIR=/path/to/OpenVAF tests/build_osdi.sh
set -eu
: "${OPENVAF_DIR:?set OPENVAF_DIR to an OpenVAF(-reloaded) checkout}"
OPENVAF="${OPENVAF:-$OPENVAF_DIR/target/release/openvaf-r}"
out="${OSDI_OUT:-$(cd "$(dirname "$0")/.." && pwd)/datasets/osdi}"
mkdir -p "$out"
# bsim4.va (Cogenda's port) differs from ngspice's b4set.c in three defaults: wk2 and wgamma1 are
# never set to 0 (the "wk1 = 0" line is duplicated where wk2 belongs), so a card that gives k2 or
# gamma1 without its W term bins against the -12345789 "not given" marker (QA3 BUG-1: 3-5x Id on
# most sky130 bins); and TNOM defaults to 25 C where ngspice uses 27 (BUG-2). It also defines
# __XYCE_VAMS__, whose I(a,b) <+ V(a,b)*g gate and body resistor branches OpenVAF compiles next to
# the else-branch's V(b,bi) <+ 0 collapses: rbodymod=1 had no effect (forward body current 4x
# native ngspice at Vb = 0.9 V); the V <+ I/g form matches native to 0.1 %. Patched copy:
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
perl -0pe 's/(if \(!BSIM4wk1Given\)\s*BSIM4wk1 = 0\.0;\s*)if \(!BSIM4wk1Given\)\s*BSIM4wk1 = 0\.0;/$1if (!BSIM4wk2Given) BSIM4wk2 = 0.0;\n            if (!BSIM4wgamma1Given) BSIM4wgamma1 = 0.0;/ or die "bsim4.va: wk2 default patch no longer applies\n";
           s/`define DEFAULT_TNOM 25\b/`define DEFAULT_TNOM 27/ or die "bsim4.va: TNOM patch no longer applies\n";
           s/^`define __XYCE_VAMS__[^\n]*$//m or die "bsim4.va: __XYCE_VAMS__ patch no longer applies\n"' \
  "$OPENVAF_DIR/integration_tests/BSIM4/bsim4.va" > "$tmp/bsim4.va"
"$OPENVAF" "$tmp/bsim4.va" -o "$out/bsim4.osdi"
"$OPENVAF" "$OPENVAF_DIR/integration_tests/BSIMCMG/bsimcmg.va" -o "$out/bsimcmg.osdi"
# HICUM/L2 2.4.0, the version of ngspice's native HICUM (BJT LEVEL 8); unpatched, it matches native to 4e-5
"$OPENVAF" "$OPENVAF_DIR/integration_tests/HICUML2/hicuml2.va" -o "$out/hicum2.osdi"
# BSIM-SOI 4.6 (ngspice's native B4SOI, LEVEL 10/58, is 4.4). bsimsoi.va adds the overlap charges to qgate
# (for the QG/CGG outputs) and then stamps ddt(qgate) from gate to body, beside its own gate-drain and gate-source
# overlap branches: the overlap capacitance counted twice and pumped into a floating body (Gemini's finding; native
# b4soild.c loads only the intrinsic charge there). Gate to body takes the intrinsic qgi:
perl -0pe 's/I\(gi,b\)\s*<\+\s*B4SOItype \* ddt\(qgate\);/I(gi,b)  <+ B4SOItype * ddt(qgi);/ or die "bsimsoi.va: qgate patch no longer applies\n"' \
  "$OPENVAF_DIR/integration_tests/BSIMSOI/bsimsoi.va" > "$tmp/bsimsoi.va"
"$OPENVAF" "$tmp/bsimsoi.va" -o "$out/bsimsoi.osdi"
# EKV 2.6 has no native ngspice model: ngspice runs it through OSDI only. ekv.va defaults TNOM to 25 C
# (ngspice: 27), and declares L, W, M, NS, AS, AD, PS and PD as model parameters, so an instance cannot give them.
perl -0pe 's/`define DEFAULT_TNOM 25\b/`define DEFAULT_TNOM 27/ or die "ekv.va: TNOM patch no longer applies\n";
           s/^(\s*)parameter real (L|W|M|NS|AS|AD|PS|PD) =/$1(* type="instance" *) parameter real $2 =/mg == 8 or die "ekv.va: instance patch no longer applies\n"' \
  "$OPENVAF_DIR/integration_tests/EKV/ekv.va" > "$tmp/ekv.va"
"$OPENVAF" "$tmp/ekv.va" -o "$out/ekv.osdi"
# IHP SG13G2's own PSP103 (module psp103va), when its PDK files are in datasets/pdk/ihp
ihp="$out/../pdk/ihp/ihp-sg13g2/libs.tech/verilog-a/psp103/psp103.va"
[ -f "$ihp" ] && "$OPENVAF" "$ihp" -o "$out/psp103.osdi"
# bsim3.va (Cogenda's port of BSIM3v3.3) differs from ngspice's b3*.c in five places; the patched copy matches
# the native model at 27, -40 and 125 C (DC, AC, noise, PMOS) to 1e-3:
#  - TNOM defaults to 25 C where ngspice uses 27;
#  - XPART defaults to 1.0 for version > 3.031 where b3set.c uses 0.0 (the 40/60 charge partition);
#  - the overlap charges qgdo / qgso use the bias-dependent local_cgdo / local_cgso in place of the CGDO / CGSO
#    parameters (b3ld.c), double counting CGDL / CGSL;
#  - Leff / Weff drop XL / XW (b3temp.c: L + XL - 2 dL), and the dL power laws use the drawn L, W;
#  - the channel thermal noise (NOIMOD 1, 3) uses hand-written Gm / Gds / Gmb whose "not calculated" terms are
#    zero (it came out 0.75 x native amplitude); ddx() of Ids gives the exact derivatives.
perl -0pe 's/`define DEFAULT_TNOM 25\b/`define DEFAULT_TNOM 27/ or die "bsim3.va: TNOM patch no longer applies\n";
           s/BSIM3xpart=1\.0;/BSIM3xpart=0.0;/ or die "bsim3.va: XPART patch no longer applies\n";
           s/(qgdo = [^;]*;)/($x=$1) =~ s{local_cgdo}{BSIM3cgdo}g; $x/ge;
           s/(qgso = [^;]*;)/($x=$1) =~ s{local_cgso}{BSIM3cgso}g; $x/ge;
           s/Lnew = l;\s*Wnew = w;\s*Ldrn = Lnew-xl;\s*Wdrn = Wnew-xw;/Lnew = l + xl;\n        Wnew = w + xw;\n        Ldrn = l;\n        Wdrn = w;/ or die "bsim3.va: XL patch no longer applies\n";
           s/pow\(Lnew, (BSIM3L[lw]n|BSIM3Wln)\)/pow(Ldrn, $1)/g;
           s/pow\(Wnew, (BSIM3Lwn|BSIM3Wwn)\)/pow(Wdrn, $1)/g;
           s/dbeta_dVg = CoxWovL \* dueff_dVg.*?Gm = Gm \* dVgsteff_dVg;\s*Gmb = Gmb \* dVbseff_dVb;/Gm = BSIM3type * ddx(Ids, V(g));\n                Gds = abs(ddx(Ids, V(di)));\n                Gmb = BSIM3type * ddx(Ids, V(b));/s or die "bsim3.va: noise patch no longer applies\n";
           s/8 \* `P_K \* T \* \(Gm \+ Gds \+ Gmb\) \/ 3/8 * `P_K * T * abs(Gm + Gds + Gmb) \/ 3/ or die "bsim3.va: noise sign patch no longer applies\n"' \
  "$OPENVAF_DIR/integration_tests/BSIM3/bsim3.va" > "$tmp/bsim3.va"
"$OPENVAF" "$tmp/bsim3.va" -o "$out/bsim3.osdi"
