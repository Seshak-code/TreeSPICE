// test_engine_fixes.cpp — Regression checks for engine review fixes 1-4.
// Drives the public ISimulationEngine API (both CPU and default/GPU modes):
//   1. NaN results must NOT report converged (fail-closed convergence)
//   2. Transient must terminate and match the analytic RC answer
//   3. Parsed diodes get SPICE defaults and honour a trailing .model card
//   4. Inductors are shorts at DC
// Run: ./build/test_engine_fixes

#include "engine_api/isimulation_engine.h"
#include "engine_api/simulation_session.h"
#include "engine_api/simulation_dto.h"
#include "engine_api/engine_callbacks.h"
#include "physics/circuitsim.h"
#include "netlist/spice_parser.h"
#include "netlist/numparam.h"
#include "netlist/hier_partition.h"

#include <array>
#include <cmath>
#include <chrono>
#include <complex>
#include <cstdio>
#include <string>
#include <filesystem>
#include <fstream>

using namespace acutesim;
using namespace acutesim::compute::orchestration;

static int g_fail = 0;
static void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

static SimulationResponseDTO run(PrecisionMode mode, const std::string& netlist,
                                 double tranStep = 0, double tranStop = 0) {
    auto engine = ISimulationEngine::create();
    engine->setPrecisionMode(mode);
    auto session = engine->createSession();
    SimulationRequestDTO req;
    req.source.payload = netlist;
    EngineCallbacks cb;
    if (tranStop > 0) {
        req.transient.stepSizeS = tranStep;
        req.transient.stopTimeS = tranStop;
        return session->runTransient(req, cb, nullptr);
    }
    return session->runDC(req, cb);
}

// V(node) of the last waveform point; node numbers are 1-based.
static double V(const SimulationResponseDTO& r, int node) {
    if (r.waveformPoints.empty() || (int)r.waveformPoints.back().nodeVoltages.size() < node)
        return NAN;
    return r.waveformPoints.back().nodeVoltages[node - 1];
}

static bool near(double a, double b, double tol) { return std::abs(a - b) <= tol; }

int main() {
    for (PrecisionMode mode : {PrecisionMode::Native_F64, PrecisionMode::Mixed}) {
        std::printf("\n== mode %s ==\n", mode == PrecisionMode::Native_F64 ? "CPU" : "default (GPU if available)");

        // Fix 4: inductor is a short at DC (1 mOhm model -> ~5 uV error)
        auto r = run(mode, "* series RL\nV1 1 0 DC 5\nL1 1 2 1m\nR1 2 0 1k\n.op\n");
        check(r.converged && near(V(r, 2), 5.0, 1e-4), "series RL: V(2) = 5 V at DC");
        r = run(mode, "* shunt RL\nV1 1 0 DC 5\nR1 1 2 1k\nL1 2 0 1m\n.op\n");
        check(r.converged && near(V(r, 2), 0.0, 1e-4), "shunt RL: V(2) = 0 V at DC");

        // Fix 3: default diode (IS=1e-14) and trailing .model card (IS=1e-12)
        r = run(mode, "* diode default\nV1 1 0 DC 5\nR1 1 2 1k\nD1 2 0 NOCARD\n.model NOCARD D\n.op\n");
        double vd = V(r, 2), id = (5.0 - vd) / 1e3;
        check(r.converged && near(vd, PhysicsConstants::V_THERMAL_NOM * std::log(id / 1e-14 + 1), 1e-4),
              "diode with SPICE defaults matches Shockley (V=" + std::to_string(vd) + ")");
        r = run(mode, "* diode card after\nV1 1 0 DC 5\nR1 1 2 1k\nD1 2 0 DB\n.model DB D(IS=1e-12)\n.op\n");
        vd = V(r, 2); id = (5.0 - vd) / 1e3;
        check(r.converged && near(vd, PhysicsConstants::V_THERMAL_NOM * std::log(id / 1e-12 + 1), 1e-4),
              "trailing .model IS=1e-12 is applied (V=" + std::to_string(vd) + ")");

        // Junction exponent cap: at 1 A (Is=1e-14) the old cap of 30 made the
        // diode linear above 0.107 A and gave 0.9915 V; Shockley gives 0.8335 V.
        r = run(mode, "* 1A diode\nI1 0 1 DC 1\nD1 1 0 DM\n.model DM D(IS=1e-14)\n.op\n");
        check(r.converged && near(V(r, 1), PhysicsConstants::V_THERMAL_NOM * std::log(1.0 / 1e-14 + 1), 2e-3),
              "1 A diode follows Shockley, not the old linear clamp (V=" + std::to_string(V(r, 1)) + ")");

        // Diode breakdown: BV is adjusted (ngspice diotemp) so that exactly IBV
        // flows at -BV; forcing IBV in reverse must give BV across the diode.
        r = run(mode, "* zener knee\nI1 0 1 DC 5m\nD1 0 1 DZ\n.model DZ D(IS=1e-14 BV=5.1 IBV=5m)\n.op\n");
        check(r.converged && near(V(r, 1), 5.1, 2e-3),
              "diode breakdown passes IBV at BV (V=" + std::to_string(V(r, 1)) + ")");

        // Gummel-Poon Early effect: Ic = IS (e^(Vbe/Vt) - 1) (1 - Vbc/VAF) in forward
        // active (q1 = 1/(1 - Vbc/VAF); the reverse terms are ~1e-15 A here).
        r = run(mode, "* early\nVCC 1 0 DC 5\nRS 1 2 1\nVBE 3 0 DC 0.7\nQ1 2 3 0 QE\n"
                      ".model QE NPN(IS=1e-15 BF=100 VAF=50)\n.op\n");
        {
            const double ic = 5.0 - V(r, 2), vbc = 0.7 - V(r, 2);  // V(): nodes in order of appearance
            const double expect = 1e-15 * (std::exp(0.7 / PhysicsConstants::V_THERMAL_NOM) - 1.0) * (1.0 - vbc / 50.0);
            check(r.converged && near(ic, expect, 1e-3 * expect),
                  "BJT Early effect matches q1 = 1/(1 - Vbc/VAF) (Ic=" + std::to_string(ic) + ")");
        }

        // Fix 1: a NaN solution must not be reported as converged (N=0 -> /0)
        r = run(mode, "* nan diode\nV1 1 0 DC 5\nR1 1 2 1k\nD1 2 0 DZ\n.model DZ D(N=0)\n.op\n");
        check(!r.converged, "NaN-producing circuit reports NOT converged");

        // GPU acceptance: a device the GPU loop doesn't stamp (E source) must
        // not yield a wrong answer flagged converged. Right answer or honest
        // failure both pass, so this survives VCVS support being added later.
        r = run(mode, "* noninverting opamp\nV1 1 0 DC 5\nRB 1 0 10k\nE1 2 0 1 3 999k\n"
                      "R1 2 3 20k\nR2 3 0 10k\n.op\n");
        check(!r.converged || near(V(r, 2), 15.0, 1e-2),
              "E-source op-amp: no false convergence (converged=" +
              std::to_string(r.converged) + ", V(out)=" + std::to_string(V(r, 2)) + ")");

        // E/G/J on the default path (were silently dropped). Gain 999k also
        // exercises the scale-aware KCL test (raw residual ~1e-5 A from rounding).
        r = run(mode, "* noninverting opamp\nV1 1 0 DC 5\nRB 1 0 10k\nE1 2 0 1 3 999k\n"
                      "R1 2 3 20k\nR2 3 0 10k\n.op\n");
        check(r.converged && near(V(r, 2), 15.0, 1e-3),
              "E-source op-amp converges to 15 V (V=" + std::to_string(V(r, 2)) + ")");
        r = run(mode, "* G source\nV1 1 0 DC 2\nRB 1 0 1k\nG1 2 0 1 0 1m\nR2 2 0 1k\n.op\n");
        check(r.converged && near(V(r, 2), -2.0, 1e-3), "VCCS: 1 mS * 2 V into 1k gives -2 V");
        // Shichman-Hodges sat: Id = BETA*(Vgs-VTO)^2 = 1e-3*(−1+2)^2 = 1 mA -> V(d) = 10 - 1 = 9 V
        r = run(mode, "* njf\nVDD 1 0 DC 10\nVG 2 0 DC -1\nRD 1 3 1k\nJ1 3 2 0 JN\n"
                      ".model JN NJF(BETA=1e-3 VTO=-2)\n.op\n");
        check(r.converged && near(V(r, 3), 9.0, 1e-3), "N-JFET saturation current matches hand calc");
        r = run(mode, "* pjf mirror\nVDD 1 0 DC -10\nVG 2 0 DC 1\nRD 1 3 1k\nJ1 3 2 0 JP\n"
                      ".model JP PJF(BETA=1e-3 VTO=-2)\n.op\n");
        check(r.converged && near(V(r, 3), -9.0, 1e-3), "P-JFET is the exact mirror (was mis-linearised)");

        // UIC: op-amp integrator has no DC operating point (open loop); SPICE
        // skips it with UIC. Also drives the Trap->Gear2 switch, whose SoA stamp
        // used to be Backward Euler and made every later step fail.
        r = run(mode, "* integrator\nVIN 1 0 SIN(0 15 60)\nR1 1 2 10k\nC1 2 3 150u\n"
                      "E1 3 0 0 2 999k\n.tran 1e-4 30e-3 uic\n", 1e-4, 30e-3);
        {
            double w = 2 * 3.141592653589793 * 60, t = 30e-3;
            double ideal = -(15.0 / (1.5 * w)) * (1 - std::cos(w * t));
            check(r.converged && near(V(r, 3), ideal, 1e-4),
                  "UIC integrator runs to 30 ms and matches -(1/RC)*integral (V=" +
                  std::to_string(V(r, 3)) + ", ideal " + std::to_string(ideal) + ")");
        }

        // Subcircuits: hierarchical and flattened netlists must agree exactly
        // (instance internals are numbered after the top-level nodes, so only
        // top-level nodes 1..N are compared). Covers: .subckt defined after use,
        // nesting, instance capacitor history and instance LTE step control.
        {
            auto same = [&](const char* what, const std::string& hier, const std::string& flat,
                            int nodes, double step, double stop) {
                auto rh = run(mode, hier, step, stop), rf = run(mode, flat, step, stop);
                double worst = (rh.waveformPoints.empty() || rf.waveformPoints.empty()) ? INFINITY : 0.0;
                for (int n = 1; n <= nodes && worst < INFINITY; ++n)
                    worst = std::max(worst, std::abs(V(rh, n) - V(rf, n)));
                check(rh.converged && rf.converged && worst < 1e-9,
                      std::string(what) + " (max |hier-flat| = " + std::to_string(worst) + " V)");
            };
            same("nested subcircuits == flattened",
                 "* h\nV1 1 0 DC 5\nXP 1 2 PAIR\n.subckt PAIR p q\nX1 p n DIV\nX2 n q DIV\nRL q 0 47k\n.ends\n"
                 ".subckt DIV a b\nR1 a mid 1k\nR2 mid b 1k\nR3 b 0 10k\n.ends\n.op\n",
                 "* f\nV1 1 0 DC 5\nRL 2 0 47k\nR1a 1 m1 1k\nR2a m1 n 1k\nR3a n 0 10k\n"
                 "R1b n m2 1k\nR2b m2 2 1k\nR3b 2 0 10k\n.op\n", 2, 0, 0);
            same("RC ladder in subcircuits == flattened (transient)",
                 "* h\nV1 1 0 PULSE(0 5 0 1u 1u 1m 2m)\nX1 1 2 RC\nX2 2 3 RC\nX3 3 4 RC\n"
                 ".subckt RC in out\nR1 in out 1k\nC1 out 0 100n\n.ends\n",
                 "* f\nV1 1 0 PULSE(0 5 0 1u 1u 1m 2m)\nR1 1 2 1k\nC1 2 0 100n\nR2 2 3 1k\n"
                 "C2 3 0 100n\nR3 3 4 1k\nC3 4 0 100n\n", 4, 1e-5, 2e-3);
            r = run(mode, "* e\nV1 1 0 1\nX1 1 0 NOPE\nR1 1 0 1k\n.op\n");
            check(!r.converged && r.errorDetail.find("unknown subcircuit") != std::string::npos,
                  "unknown subcircuit is a parse error");
        }

        // Fix 2: RC transient at dt = tau/100 terminates and is accurate
        r = run(mode, "* rc\nV1 1 0 DC 5\nR1 1 2 1k\nC1 2 0 1u\n.tran 1e-5 1e-3 uic\n", 1e-5, 1e-3);
        double expect = 5.0 * (1.0 - std::exp(-1.0));
        check(r.converged && near(V(r, 2), expect, 1e-3) &&
              near(r.waveformPoints.back().timeS, 1e-3, 1e-12),
              "RC transient (uic) reaches t=1ms with V(2)=" + std::to_string(V(r, 2)) +
              " (analytic " + std::to_string(expect) + ")");

        // Phase 2.1: without UIC the transient starts at the DC operating point
        // (it used to start from zero state: the RC above charged from 0 V)
        r = run(mode, "* rc op\nV1 1 0 DC 5\nR1 1 2 1k\nC1 2 0 1u\n.tran 1e-5 1e-3\n", 1e-5, 1e-3);
        check(r.converged && near(r.waveformPoints.front().nodeVoltages[1], 5.0, 1e-6) && near(V(r, 2), 5.0, 1e-6),
              "RC without uic stays at its DC operating point 5 V (V(2)=" + std::to_string(V(r, 2)) + ")");
        r = run(mode, "* rl op\nV1 1 0 DC 1\nR1 1 2 1k\nL1 2 0 1m\n.tran 1e-8 1e-6\n", 1e-8, 1e-6);
        check(r.converged && std::abs(r.waveformPoints.front().nodeVoltages[1]) < 1e-5,
              "RL without uic starts with I_L = 1 mA (V(2)(0+)=" +
              std::to_string(r.waveformPoints.front().nodeVoltages[1]) + ", zero start gives 0.99)");
        // IC= with UIC, and .ic (forced during the DC bias, or initial with UIC):
        // all four give v(t) = 5 - 3 exp(-t/RC)
        const double expectIC = 5.0 - 3.0 * std::exp(-1.0);
        for (const char* nl : {"* c ic\nV1 1 0 DC 5\nR1 1 2 1k\nC1 2 0 1u IC=2\n.tran 1e-5 1e-3 uic\n",
                               "* .ic\nV1 1 0 DC 5\nR1 1 2 1k\nC1 2 0 1u\n.ic v(2)=2\n.tran 1e-5 1e-3\n",
                               "* .ic uic\nV1 1 0 DC 5\nR1 1 2 1k\nC1 2 0 1u\n.ic v(2)=2\n.tran 1e-5 1e-3 uic\n"}) {
            r = run(mode, nl, 1e-5, 1e-3);
            check(r.converged && near(V(r, 2), expectIC, 1e-3),
                  std::string(nl).substr(2, std::string(nl).find('\n') - 2) + ": V(2)(1ms)=" +
                  std::to_string(V(r, 2)) + " (analytic " + std::to_string(expectIC) + ")");
        }
        r = run(mode, "* l ic\nV1 1 0 DC 1\nR1 1 2 1k\nL1 2 0 1m IC=1m\n.tran 1e-8 1e-6 uic\n", 1e-8, 1e-6);
        check(r.converged && std::abs(r.waveformPoints.front().nodeVoltages[1]) < 1e-5,
              "L IC=1m with uic starts in steady state (V(2)(0+)=" +
              std::to_string(r.waveformPoints.front().nodeVoltages[1]) + ")");
        r = run(mode, "* bad ic\nV1 1 0 5\nR1 1 0 1k\n.ic v(nope)=1\n.tran 1u 1m\n", 1e-6, 1e-3);
        check(!r.converged && r.errorDetail.find("unknown node") != std::string::npos,
              ".ic on an unknown node is a parse error");

        // Waveform sources on the default path (were stuck at 0 V / DC)
        r = run(mode, "* sin\nV1 1 0 SIN(0 1 1k)\nR1 1 0 1k\n", 1e-5, 2.5e-4);
        check(r.converged && near(V(r, 1), 1.0, 2e-3),
              "SIN source peaks at 1 V at t=T/4 (V=" + std::to_string(V(r, 1)) + ")");
        r = run(mode, "* pulse rc\nV1 1 0 PULSE(0 5 0 1n 1n 1 2)\nR1 1 2 1k\nC1 2 0 1u\n", 1e-5, 1e-3);
        check(r.converged && near(V(r, 2), expect, 2e-3),
              "PULSE-driven RC matches analytic (V=" + std::to_string(V(r, 2)) + ")");

        // Fix 5: parser
        r = run(mode, "* ac then sin\nV1 1 0 AC 1 SIN(0 1 1k)\nR1 1 0 1k\n.op\n");
        check(r.converged && near(V(r, 1), 0.0, 1e-4),
              "AC + SIN source: DC value is SIN's t=0 value, not 1000 V");
        r = run(mode, "* param\n.param RV=1k\nV1 1 0 DC 5\nR1 1 2 {RV}\nR2 2 0 1k\n.op\n");
        check(r.converged && near(V(r, 2), 2.5, 1e-3), ".param {RV} is substituted");
        r = run(mode, "* late pnp model\nVCC 1 0 DC 5\nRE 1 2 1k\nVB 3 0 DC 3\nQ1 4 3 2 QP\n"
                      "RC 4 0 1k\n.model QP PNP(IS=1e-15 BF=100)\n.op\n");
        // Hand: Ie~(5-Ve)/1k, Ic=Ie*100/101, Vbe=PhysicsConstants::V_THERMAL_NOM*ln(Ic/1e-15)=0.7205 -> Ve=3.7205
        // (an NPN would not conduct at all; ignoring IS from the card gives 3.662)
        check(r.converged && near(V(r, 2), 3.7205, 2e-3),
              ".model after the device still makes it a PNP (V(e)=" + std::to_string(V(r, 2)) + ")");
        r = run(mode, "* unknown\nV1 1 0 5\n9Z1 1 0 5\n.op\n");  // no element letter (Z is a MESFET)
        check(!r.converged && r.errorDetail.find("unknown element") != std::string::npos,
              "unknown element is a parse error: " + r.errorDetail);
        // MNA branch currents: F/H read V1's branch current (1e-6 tolerance: every
        // node carries a 1e-12 S GMIN, a 1e-9 relative effect here). SPICE convention:
        // I(V1) flows + -> source -> -, so V1 driving 1k gives I(V1) = -1 mA.
        r = run(mode, "* cccs\nV1 1 0 1\nR1 1 0 1k\nF1 0 2 V1 10\nR2 2 0 1k\n.op\n");
        check(r.converged && near(V(r, 2), -10.0, 1e-6), "F source: 10 * I(V1) into 1k = -10 V (V=" + std::to_string(V(r, 2)) + ")");
        r = run(mode, "* ccvs\nV1 1 0 1\nR1 1 0 1k\nH1 2 0 V1 1k\nR2 2 0 1k\n.op\n");
        check(r.converged && near(V(r, 2), -1.0, 1e-6), "H source: 1k * I(V1) = -1 V (V=" + std::to_string(V(r, 2)) + ")");
        r = run(mode, "* exact source\nV1 1 0 DC 5\nR1 1 0 1\n.op\n");
        check(r.converged && near(V(r, 1), 5.0, 1e-12), "V source exact at 5 A load (was 1 mOhm Norton: 5 mV low)");

        // MOSFET level 1, linear region: Ids = K(2*Vov*Vds - Vds^2)(1 + lambda*Vds)
        // (lambda was only applied in saturation). Drain held near 0.3 V via 1 ohm.
        r = run(mode, "* nmos linear\nVD 1 0 DC 0.3\nR1 1 2 1\nM1 2 3 0 0 NM W=10u L=1u\nVG 3 0 DC 2\n"
                      ".model NM NMOS(VTO=0.7 KP=200u LAMBDA=0.02)\n.op\n");
        {
            double K = 0.5 * 200e-6 * 10.0, vov = 2.0 - 0.7, vd = 0.3;
            for (int it = 0; it < 100; ++it)
                vd = 0.3 - 1.0 * K * (2 * vov * vd - vd * vd) * (1 + 0.02 * vd);
            check(r.converged && near(V(r, 2), vd, 1e-9),
                  "NMOS linear region includes lambda (Vd=" + std::to_string(V(r, 2)) + ", hand " + std::to_string(vd) + ")");
        }

        // Transient inductor (failed to converge before: history never written back)
        // RL step, tau = L/R = 1 us: v_R(5 us) = 1 - e^-5
        r = run(mode, "* rl step\nV1 1 0 DC 1\nL1 1 2 1m\nR1 2 0 1k\n.tran 1e-8 5e-6 uic\n", 1e-8, 5e-6);
        check(r.converged && near(V(r, 2), 1.0 - std::exp(-5.0), 1e-4),
              "RL step transient matches 1 - e^-t/tau (V=" + std::to_string(V(r, 2)) + ")");
        // Mutual inductance, open secondary: v2/v1 = M/L1 = k = 0.99
        r = run(mode, "* xfmr\nV1 1 0 SIN(0 1 1k)\nR1 1 2 1\nL1 2 0 1m\nL2 3 0 1m\nK1 L1 L2 0.99\n"
                      "R2 3 0 1MEG\n.tran 1e-6 2.5e-4\n", 1e-6, 2.5e-4);
        check(r.converged && std::abs(V(r, 2)) > 0.1 && near(V(r, 3) / V(r, 2), 0.99, 1e-3),
              "K coupling: open-circuit v2/v1 = 0.99 (ratio=" + std::to_string(V(r, 3) / V(r, 2)) + ")");

        // A .tran step 10x the period (QA3 BUG-6): steps started at tstep and stepped a 10 MHz LC
        // tank a period at a time, collapsing it to 4e-81. Now ngspice's step control: the
        // envelope over the 20th period stays at the exact exp(-t/2RC) = 0.990 (ngspice 45
        // itself drifts out of phase here; the envelope is what both should keep)
        r = run(mode, "* lc\nL1 1 0 253.3n\nC1 1 0 1n IC=1\nR1 1 0 100k\n.tran 1u 2u uic\n", 1e-6, 2e-6);
        double env = 0.0;
        for (const auto& p : r.waveformPoints) if (p.timeS > 1.9e-6) env = std::max(env, std::abs(p.nodeVoltages[0]));
        check(r.converged && near(env, std::exp(-2e-6 / (2 * 100e3 * 1e-9)), 0.03),
              "LC tank with tstep = 10 periods keeps its envelope (" + std::to_string(env) + ")");

        // Fix 2: an invalid timestep is an error, not an infinite loop; 0 means automatic
        r = run(mode, "* rc\nV1 1 0 DC 5\nR1 1 2 1k\nC1 2 0 1u\n", -1e-6, 1e-3);
        check(!r.converged && !r.errorDetail.empty(), "negative timestep returns an error");
        r = run(mode, "* rc\nV1 1 0 DC 5\nR1 1 2 1k\nC1 2 0 1u\n.tran 0 1m uic\n", 0.0, 1e-3);
        check(r.converged && near(r.waveformPoints.back().timeS, 1e-3, 1e-12) && near(V(r, 2), 5.0 * (1.0 - std::exp(-1.0)), 5e-3),
              ".tran 0 tstop runs with an automatic step (V(2)=" + std::to_string(V(r, 2)) + ")");
    }

    // Phase 2.3: native .dc (netlist card or DCConfigDTO); unsupported forms are errors
    for (PrecisionMode mode : {PrecisionMode::Native_F64, PrecisionMode::Mixed}) {
        auto r = run(mode, "* dc sweep\nV1 1 0 DC 0\nR1 1 2 1k\nR2 2 0 3k\n.dc V1 0 2 0.5\n");
        bool ok = r.converged && r.waveformPoints.size() == 5;
        for (size_t k = 0; ok && k < 5; ++k)
            ok = near(r.waveformPoints[k].timeS, 0.5 * k, 1e-12) &&
                 near(r.waveformPoints[k].nodeVoltages[1], 0.375 * k, 1e-6);
        check(ok, ".dc V1 0 2 0.5: 5 points, V(2) = 0.75*V1");
        auto engine = ISimulationEngine::create();
        engine->setPrecisionMode(mode);
        SimulationRequestDTO req;
        req.source.payload = "* dto sweep\nI1 0 1 DC 0\nR1 1 0 2k\n.op\n";
        req.dc = {"i1", 0.0, 1e-3, 1e-3};
        EngineCallbacks cb;
        r = engine->createSession()->runDC(req, cb);
        check(r.converged && r.waveformPoints.size() == 2 && near(r.waveformPoints[1].nodeVoltages[0], 2.0, 1e-6),
              "DCConfigDTO sweep of a current source: V(1) = 2 V at 1 mA");
        r = run(mode, "* nested\nV1 1 0 1\nV2 2 0 1\nR1 1 2 1k\n.dc V1 0 1 0.5 V2 0 1 0.5\n");
        check(r.converged && r.waveformPoints.size() == 9 && r.outerNames.size() == 1 &&
                  near(r.waveformPoints[4].timeS, 0.5, 1e-12) && near(r.waveformPoints[4].outer[0], 0.5, 1e-12) &&
                  near(r.waveformPoints[8].timeS, 1.0, 1e-12) && near(r.waveformPoints[8].outer[0], 1.0, 1e-12),
              "nested .dc: 3 x 3 points, the first source innermost");
        {   // a corner temperature is the netlist's .temp; a .temp list is for .op/.dc only
            const std::string ckt = "* corner\nV1 1 0 DC 1\nR1 1 2 1k\nD1 2 0 DM\n.model DM D(IS=1e-14 EG=1.11 XTI=3)\n";
            auto eng = ISimulationEngine::create();
            eng->setPrecisionMode(mode);
            SimulationRequestDTO cr;
            cr.source.payload = ckt;
            cr.corner.temperature = 85.0;
            EngineCallbacks ccb;
            auto a = eng->createSession()->runDC(cr, ccb);
            auto b = run(mode, ckt + ".temp 85\n");
            check(a.converged && b.converged && near(a.waveformPoints[0].nodeVoltages[1], b.waveformPoints[0].nodeVoltages[1], 1e-12) &&
                      !near(a.waveformPoints[0].nodeVoltages[1], run(mode, ckt).waveformPoints[0].nodeVoltages[1], 1e-3),
                  "corner temperature 85 equals .temp 85");
            cr.corner.temperature = 27.0;
            cr.source.payload = ckt + ".temp 10 20\n";
            auto t = eng->createSession()->runTransient(cr, ccb, nullptr);
            check(!t.converged && t.errorDetail.find(".temp list") != std::string::npos, ".temp list with .tran is an error");
        }
        r = run(mode, "* missing\nV1 1 0 1\nR1 1 0 1k\n.dc V9 0 1 0.5\n");
        check(!r.converged && r.errorDetail.find("not found") != std::string::npos, ".dc of an unknown source is an error");
    }

    // Phase 2.4: AC (G + jwC from the DC Jacobian) and noise through the session
    for (PrecisionMode mode : {PrecisionMode::Native_F64, PrecisionMode::Mixed}) {
        auto engine = ISimulationEngine::create();
        engine->setPrecisionMode(mode);
        EngineCallbacks cb;
        SimulationRequestDTO req;
        req.source.payload = "* rc corner\nV1 1 0 AC 1\nR1 1 2 1k\nC1 2 0 1u\n.ac lin 1 159.15494309189535 159.15494309189535\n";
        auto r = engine->createSession()->runAC(req, cb);
        bool ok = r.converged && r.waveformPoints.size() == 1;
        const double re = ok ? r.waveformPoints[0].nodeVoltages[1] : NAN, im = ok ? r.waveformPoints[0].nodeVoltagesImag[1] : NAN;
        check(ok && near(std::hypot(re, im), std::sqrt(0.5), 1e-9) && near(std::atan2(im, re), -M_PI / 4, 1e-9),
              "AC: RC at f = 1/(2 pi RC) gives 1/sqrt(2) at -45 deg");
        req.source.payload = "* no sweep\nV1 1 0 AC 1\nR1 1 0 1k\n";
        r = engine->createSession()->runAC(req, cb);
        check(!r.converged && !r.errorDetail.empty(), "AC without a sweep is an error");
        req.source.payload = "* divider noise\nV1 1 0 DC 0 AC 1\nR1 1 2 1k\nR2 2 0 1k\n.noise v(2) V1 lin 2 1 2\n";
        r = engine->createSession()->runNoise(req, cb);
        const double expect = std::sqrt(4 * 1.380649e-23 * 300.15 * 500.0);  // 1k || 1k thermal
        ok = r.converged && r.extended.noise.onoise.size() == 2;
        check(ok && near(r.extended.noise.onoise[0], expect, 1e-6 * expect) &&
                  near(r.extended.noise.inoise[0], 2 * expect, 2e-6 * expect) &&
                  near(r.extended.noise.onoiseTotal, expect, 1e-6 * expect),  // white: total over 1 Hz
              "noise: 1k/1k divider gives sqrt(4kT*500) out, x2 input-referred, white total");
        // MOS LEVEL 6: ngspice's mos6 has no AC load (its .ac sees no device), so the oracle is the DC slope, which
        // run_diff holds to ngspice (x_mos6_*): the 10 Hz gain equals the central difference of v(3) over VG
        const auto mos6 = [](double vg, const char* an) {
            return "* mos6 cs\nVDD 1 0 DC 5\nVG 2 0 DC " + std::to_string(vg) + " AC 1\nRD 1 3 10k\nM1 3 2 0 0 NM W=20u L=2u m=2\n"
                   ".model NM NMOS(LEVEL=6 VTO=0.8 KV=2 NV=0.7 KC=2e-5 NC=1.3 GAMMA=0.5 PHI=0.65 TOX=50n LAMBDA=0.02 SIGMA=0.01 "
                   "CGSO=4e-10 CGDO=4e-10 CJ=3e-4)\n" + std::string(an);
        };
        req.source.payload = mos6(1.3, ".ac lin 1 10 10\n");
        r = engine->createSession()->runAC(req, cb);
        const double h = 1e-4, slope = (V(run(mode, mos6(1.3 + h, "")), 3) - V(run(mode, mos6(1.3 - h, "")), 3)) / (2 * h);
        ok = r.converged && r.waveformPoints.size() == 1;
        check(ok && near(r.waveformPoints[0].nodeVoltages[2], slope, 1e-6 * std::abs(slope)) && std::abs(slope) > 1.0,
              "MOS LEVEL 6: 10 Hz AC gain equals the DC slope dv(3)/dVG");
        // BJT quasi-saturation in AC: ngspice 47's Irci partial in vbc takes d ln((1+Kbci)/(1+Kbcx)) as
        // Kbci'(1+Kbcx)/(1+Kbci)^2 (not Kbci'/(1+Kbci)), so its AC gain (-4.141 here) is not its own DC slope
        // (-4.0175); the oracle is the DC slope: the 10 Hz gain equals the central difference of v(c) over VIN
        {
            const auto qs = [](double vin, const char* an) {
                return "* bjt qs ac\nvin in 0 DC " + std::to_string(vin) + " AC 1\nrb in b 100\nvcc vcc 0 DC 3\nrl vcc c 500\n"
                       "q1 c b 0 qs\n.model qs npn is=1e-15 bf=150 br=3 vaf=80 rb=20 rc=2 re=0.5 rco=80 vo=1.5 gamma=1e-9 "
                       "qco=5e-12 cjc=1p cje=2p tf=0.2n tr=2n\n" + std::string(an);
            };
            req.source.payload = qs(0.79, ".ac lin 1 10 10\n");
            const auto ra = engine->createSession()->runAC(req, cb);
            const double hq = 1e-5, sq = (V(run(mode, qs(0.79 + hq, "")), 4) - V(run(mode, qs(0.79 - hq, "")), 4)) / (2 * hq);
            const bool okq = ra.converged && ra.waveformPoints.size() == 1;
            check(okq && near(ra.waveformPoints[0].nodeVoltages[3], sq, 1e-5 * std::abs(sq)) && near(sq, -4.0175, 2e-3),
                  "BJT quasi-saturation: 10 Hz AC gain equals the DC slope dv(c)/dVIN (" +
                      (okq ? std::to_string(ra.waveformPoints[0].nodeVoltages[3]) : ra.errorDetail) + " vs " + std::to_string(sq) + ")");
        }
        // Diode RSW: the separate sidewall's transient charge is its depletion charge (ngspice 47 stores the
        // capacitance as that charge, so its sidewall current is d(C)/dt: 10 pF behind 1 ohm nearly vanishes, v(2)
        // -4.02 V at 3 ns): a microohm RSW follows the common sidewall's -0.6965 V (ngspice's own, without RSW)
        {
            const auto d = [](const char* rsw) {
                return "* diode rsw tran\nV1 1 0 PULSE(0 -5 1n 1n 1n 50n 100n)\nR1 1 2 1k\nD1 2 0 ds PJ=1\nR2 1 3 1k\nD2 3 0 dc PJ=1\n"
                       ".model ds D(IS=1e-14 CJSW=10p MJSW=0.01 JSW=1e-15 " + std::string(rsw) + ")\n"
                       ".model dc D(IS=1e-14 CJSW=10p MJSW=0.01 JSW=1e-15)\n.tran 0.1n 3n\n";
            };
            const auto rt = run(mode, d("RSW=1e-6"), 1e-10, 3e-9);
            check(rt.converged && near(V(rt, 2), V(rt, 3), 1e-4) && near(V(rt, 3), -0.6965, 5e-3),
                  "diode RSW: the sidewall's transient charge is its depletion charge (" + std::to_string(V(rt, 2)) + " vs " +
                      std::to_string(V(rt, 3)) + " V)");
        }
        // Diode RSW with the bottom's characteristic in AC: the sidewall current follows the bottom's vd, which
        // ngspice 47's AC load stamps as a conductance at A_sw behind RSW (0.295 here against its own DC slope
        // 0.205): the 10 Hz gain equals the central difference of v(2) over VB
        {
            const auto dr = [](double vb, const char* an) {
                return "* diode rsw ac\nVB 1 0 DC " + std::to_string(vb) + " AC 1\nR3 1 2 300\nD3 2 0 dc PJ=50u\n"
                       ".model dc D(IS=1e-14 CJO=1p JSW=1e-9 CJSW=3e-7 RSW=100)\n" + std::string(an);
            };
            req.source.payload = dr(0.68, ".ac lin 1 10 10\n");
            const auto ra = engine->createSession()->runAC(req, cb);
            const double hd = 1e-5, sd = (V(run(mode, dr(0.68 + hd, "")), 2) - V(run(mode, dr(0.68 - hd, "")), 2)) / (2 * hd);
            const bool okd = ra.converged && ra.waveformPoints.size() == 1;
            check(okd && near(ra.waveformPoints[0].nodeVoltages[1], sd, 1e-5 * std::abs(sd)) && near(sd, 0.205, 2e-3),
                  "diode RSW (bottom characteristic): 10 Hz AC gain equals the DC slope dv(2)/dVB (" +
                      (okd ? std::to_string(ra.waveformPoints[0].nodeVoltages[1]) : ra.errorDetail) + " vs " + std::to_string(sd) + ")");
        }
        // Diode self-heating in AC. ngspice 47's AC leaves CTH0 out (dioacld stamps the thermal capacitor's
        // operating-point current, 0) and the power's charge-current term its transient has, and its dI/dT misses
        // the IKF knee (0.0396 against its DC slope 0.0426 here): (1) without junction charge the thermal loop is
        // one pole, H0 / H(f) - 1 = j f / fp, H0 the 1 mHz gain = the DC slope of v(tj), IKF in;
        // (2) with TT, v(tj) at 25.1 kHz is ngspice's own transient's 17.1606 - 0.00029j (a 1 mV sine, fitted),
        // where its AC gives +0.0075j
        {
            const auto dh = [](double vd, const char* tt, const char* an) {
                return "* diode heat ac\nVD a 0 DC " + std::to_string(vd) + " AC 1\nR1 a k 10\nD1 k 0 tj DH PJ=1 THERMAL\n"
                       ".model DH D(IS=1e-12 N=1.2 RS=0.5 RTH0=200 TRS1=3e-3 JSW=1e-12 " + std::string(tt) + ")\n" + std::string(an);
            };
            auto ac = [&](const std::string& nl) {
                req.source.payload = nl;
                return engine->createSession()->runAC(req, cb);
            };
            const auto r1 = ac(dh(2.5, "CTH0=1e-5 IKF=0.5", ".ac lin 1 100 100\n")), r4 = ac(dh(2.5, "CTH0=1e-5 IKF=0.5", ".ac lin 1 400 400\n"));
            const auto r0 = ac(dh(2.5, "CTH0=1e-5 IKF=0.5", ".ac lin 1 1e-3 1e-3\n"));
            const double hv = 1e-5, sh = (V(run(mode, dh(2.5 + hv, "CTH0=1e-5 IKF=0.5", "")), 3) - V(run(mode, dh(2.5 - hv, "CTH0=1e-5 IKF=0.5", "")), 3)) / (2 * hv);
            const bool okh = r0.converged && r1.converged && r4.converged && !r0.waveformPoints.empty() &&
                             !r1.waveformPoints.empty() && !r4.waveformPoints.empty();
            auto H = [](const SimulationResponseDTO& r) {
                return std::complex<double>(r.waveformPoints[0].nodeVoltages[2], r.waveformPoints[0].nodeVoltagesImag[2]);
            };
            const std::complex<double> h0 = okh ? H(r0) : NAN, e1 = okh ? h0 / H(r1) - 1.0 : NAN, e4 = okh ? h0 / H(r4) - 1.0 : NAN;
            check(okh && near(h0.real(), sh, 1e-5 * std::abs(sh)) && std::abs(e1.real()) < 1e-4 * std::abs(e1) &&
                      near(e4.imag() / e1.imag(), 4.0, 1e-4) && e1.imag() > 0.0,
                  "diode self-heating AC: the thermal loop's one pole at RTH0 || CTH0 (fp " +
                      (okh ? std::to_string(100.0 / e1.imag()) : std::string("?")) + " Hz), its DC slope " + std::to_string(sh));
            const auto rt = ac(dh(2.5, "CTH0=1e-13 TT=5n", ".ac lin 1 25118.9 25118.9\n"));
            const bool okt = rt.converged && !rt.waveformPoints.empty();
            check(okt && near(H(rt).real(), 17.16060, 2e-5) && near(H(rt).imag(), -2.8999e-4, 2e-6),
                  "diode self-heating AC with TT: v(tj) is ngspice's transient's (" +
                      (okt ? std::to_string(H(rt).real()) + std::to_string(H(rt).imag()) + "j" : rt.errorDetail) + ")");
        }
        // Diode LEVEL 3: ngspice 47 multiplies M into the W/L AREA and PJ and again where they scale the device
        // (M^2), and takes its XOM / XOI default 1e4 in metres (no overlap capacitance) where a given value is in
        // Angstrom: here M=2 is two M=1 diodes in parallel, and the default is the 1e4 Angstrom of XOM=1e4
        {
            const std::string m3 = ".option scale=1e-6\n.model d3 D(LEVEL=3 IS=1e-4 JSW=1e-10 CJO=2e-4 LM=50 WM=40 XM=1e-7)\n";
            req.source.payload = "* level3 m\nVB 1 0 DC 0.7 AC 1\nR1 1 2 100\nD1 2 0 d3 W=20 L=10 M=2\nR2 1 3 100\nD2 3 0 d3 W=20 L=10\n"
                                 "D3 3 0 d3 W=20 L=10\n" + m3 + ".ac lin 1 100Meg 100Meg\n";
            const auto rm = engine->createSession()->runAC(req, cb);
            const bool okm = rm.converged && !rm.waveformPoints.empty();
            const auto& pm = okm ? rm.waveformPoints[0] : WaveformDataPoint{};
            check(okm && near(pm.nodeVoltages[1], pm.nodeVoltages[2], 1e-10) && near(pm.nodeVoltagesImag[1], pm.nodeVoltagesImag[2], 1e-10) && std::abs(pm.nodeVoltagesImag[1]) > 1e-4,
                  "diode LEVEL 3: W/L with M=2 is two M=1 diodes in parallel (" +
                      (okm ? std::to_string(pm.nodeVoltages[1]) + " vs " + std::to_string(pm.nodeVoltages[2]) : rm.errorDetail) + ")");
            auto ac3 = [&](const char* xom) {
                req.source.payload = "* level3 xom\nVB 1 0 DC -1 AC 1\nR1 1 2 1k\nD1 2 0 d3 W=20 L=10\n"
                                     ".option scale=1e-6\n.model d3 D(LEVEL=3 IS=1e-4 LM=50 WM=40 " + std::string(xom) + ")\n.ac lin 1 1G 1G\n";
                const auto r = engine->createSession()->runAC(req, cb);
                return r.converged && !r.waveformPoints.empty() ? r.waveformPoints[0].nodeVoltagesImag[1] : NAN;
            };
            const double dflt = ac3(""), given = ac3("XOM=1e4"), none = ac3("XOM=1e4 LM=0");
            check(near(dflt, given, 1e-15) && std::abs(dflt - none) > 1e-3,
                  "diode LEVEL 3: the XOM default is 1e4 Angstrom (" + std::to_string(dflt) + ", " + std::to_string(given) +
                      ", no overlap " + std::to_string(none) + ")");
        }
        // XSPICE aswitch in AC is the resistance its control sets (ngspice 47's AC load negates the conductance:
        // a 1 V source over 1k and the switch read 1.46 V there): the same divider as a resistor of that value
        {
            const std::string sw = "* aswitch ac\nVS s 0 DC 1 AC 1\nR1 s x 1k\nVC c 0 DC 0.5\na1 c %gd(x 0) sw\n"
                                   ".model sw aswitch(cntl_off=0 cntl_on=1 r_off=100k r_on=100)\nR2 s y 1k\nR3 y 0 3162.2776601683795\n"
                                   ".ac lin 1 1k 1k\n";
            req.source.payload = sw;
            const auto ra = engine->createSession()->runAC(req, cb);
            const bool ok2 = ra.converged && ra.waveformPoints.size() == 1;
            check(ok2 && near(ra.waveformPoints[0].nodeVoltages[1], ra.waveformPoints[0].nodeVoltages[3], 1e-9) &&
                      near(ra.waveformPoints[0].nodeVoltages[1], 0.7597469, 1e-6),
                  "XSPICE aswitch in AC is its resistance (" + (ok2 ? std::to_string(ra.waveformPoints[0].nodeVoltages[1]) : ra.errorDetail) + " V)");
        }
        // XSPICE pswitch in AC is its resistance too (ngspice 47 stamps -1/r, as aswitch, and leaves out the control
        // port's own): log, mid-way between cntl_off and cntl_on, r = sqrt(r_on r_off), the aswitch case's divider
        {
            req.source.payload = "* pswitch ac\nVS s 0 DC 1 AC 1\nR1 s x 1k\nVC c 0 DC 0.5\na1 %g c %gd(x 0) sw\n"
                                 ".model sw pswitch(cntl_off=0 cntl_on=1 r_off=100k r_on=100)\n.ac lin 1 1k 1k\n";
            const auto ra = engine->createSession()->runAC(req, cb);
            const bool ok2 = ra.converged && ra.waveformPoints.size() == 1;
            check(ok2 && near(ra.waveformPoints[0].nodeVoltages[1], 0.7597469, 1e-6),
                  "XSPICE pswitch in AC is its resistance (" + (ok2 ? std::to_string(ra.waveformPoints[0].nodeVoltages[1]) : ra.errorDetail) + " V)");
        }
        // XSPICE lcouple on a linear core in AC is the inductor N^2 mu A / length = 100 uH it is in a transient
        // (ngspice 47's winding gain is +jwN against its transient's -N d/dt on the core current: a negative
        // inductor in AC): re jwl/(1+jwl) = 0.283043 at 1 kHz over 1 ohm, as the L element beside it
        {
            req.source.payload = "* lcouple ac\nVS s 0 DC 0 AC 1\nR1 s x 1\na1 x 0 mm 0 lc\n.model lc lcouple(num_turns=10)\n"
                                 "a2 mm 0 cr\n.model cr core(h_array=[-1000 1000] b_array=[-1 1] area=1e-4 length=0.1)\n"
                                 "R2 s y 1\nL2 y 0 1e-4\n.ac lin 1 1k 1k\n";
            const auto ra = engine->createSession()->runAC(req, cb);
            const bool ok2 = ra.converged && ra.waveformPoints.size() == 1;
            check(ok2 && near(ra.waveformPoints[0].nodeVoltages[1], ra.waveformPoints[0].nodeVoltages[2], 1e-9) &&
                      near(ra.waveformPoints[0].nodeVoltages[1], 0.283043, 1e-5),
                  "XSPICE lcouple in AC is its inductor (" + (ok2 ? std::to_string(ra.waveformPoints[0].nodeVoltages[1]) : ra.errorDetail) + " V)");
        }
        // XSPICE inductoric in AC is an inductor's admittance 1/(jwl) (ngspice 47 gives jwl: a shunt one read as the
        // series divider, 0.8467 V here): the same as an L element, re jwl/(1+jwl) = 0.283043 at 100 kHz over 1 ohm
        {
            req.source.payload = "* inductoric ac\nVS s 0 DC 0 AC 1\nR1 s x 1\na1 x 0 ind1\n.model ind1 inductoric(l=1u)\n"
                                 "R2 s y 1\nL2 y 0 1u\n.ac lin 1 100k 100k\n";
            const auto ra = engine->createSession()->runAC(req, cb);
            const bool ok2 = ra.converged && ra.waveformPoints.size() == 1;
            check(ok2 && near(ra.waveformPoints[0].nodeVoltages[1], ra.waveformPoints[0].nodeVoltages[2], 1e-9) &&
                      near(ra.waveformPoints[0].nodeVoltages[1], 0.283043, 1e-5),
                  "XSPICE inductoric in AC is an inductor (" + (ok2 ? std::to_string(ra.waveformPoints[0].nodeVoltages[1]) : ra.errorDetail) + " V)");
        }
        req.source.payload = mos6(1.3, ".noise v(3) VG lin 1 10 10\n");
        r = engine->createSession()->runNoise(req, cb);
        check(!r.converged && r.errorDetail.find("no noise model") != std::string::npos, "MOS LEVEL 6 .noise is an error");
    }

    // No node gmin: a node with no DC path is not silently 0 (the contraction
    // solver once left a root front's pivot-less rows at x = 0 and "converged":
    // series caps gave V(2)=0); a linear circuit converging on the direct attempt
    // must not report GMIN stepping
    for (PrecisionMode mode : {PrecisionMode::Native_F64, PrecisionMode::Mixed}) {
        // ngspice 47 resolves it with its transient OP (from uncharged capacitors):
        // the series caps divide, V(2) = 2.5 V
        auto r = run(mode, "* series caps\nV1 1 0 DC 5\nC1 1 2 1u\nC2 2 0 1u\nR1 1 0 1k\n.op\n");
        check(r.converged && near(V(r, 2), 2.5, 1e-4),
              "node without a DC path: transient OP gives the capacitive divider V(2) = 2.5 (" + std::to_string(V(r, 2)) + ")");
        // ...settled or not: 1 mA into a 1 uF cap-only node is I*10us/C = 0.01 V, as ngspice 47
        r = run(mode, "* cap-only node\nI1 0 2 DC 1m\nC1 2 0 1u\nV1 1 0 DC 1\nR1 1 0 1k\n.op\n");
        check(r.converged && near(V(r, 1), 0.01, 1e-6),  // node "2" is listed first: index 1
              "current into a cap-only node: ngspice's transient OP value 0.01 V (" + std::to_string(V(r, 1)) + ")");
        // A pivot cancelled to rounding residue is singular: 1 mA into a floating 100-resistor
        // chain split over several fronts "converged" at 1e14 V (BUG-6)
        std::string net = "* floating chain\n.options gmin=0 gminsteps=0\nV1 1 0 DC 1\nR0 1 0 1k\nI1 0 f0 DC 1m\n";
        for (int i = 0; i < 100; ++i) net += "RF" + std::to_string(i) + " f" + std::to_string(i) + " f" + std::to_string(i + 1) + " 1k\n";
        r = run(mode, net + ".op\n");
        check(!r.converged && r.errorDetail.find("no DC path") != std::string::npos,
              "current into a floating 100-resistor chain: singular (" + r.errorDetail.substr(0, 60) + ")");
        // ...and a floating star, whose cancelled pivot is fill (its own A entry is 0): only
        // the absolute bound (ngspice's PIVTOL) catches it (BUG-6, Gemini round 3)
        net = "* floating star\n.options gmin=0 gminsteps=0\nV1 1 0 DC 1\nR0 1 0 1k\nI1 0 c DC 1m\n";
        for (int i = 0; i < 10; ++i) net += "RS" + std::to_string(i) + " c s" + std::to_string(i) + " 1k\n";
        r = run(mode, net + ".op\n");
        check(!r.converged && r.errorDetail.find("no DC path") != std::string::npos,
              "current into a floating 10-spoke star: singular (" + r.errorDetail.substr(0, 60) + ")");
        // A floating node behind a 1 ms RC: ngspice's 10 us transient OP leaves it mid-charge
        r = run(mode, "* rc float\nV1 1 0 DC 10\nR1 1 2 1k\nC1 2 3 1u\nC2 3 4 1u\nR2 4 0 1k\n.op\n");
        check(r.converged && near(V(r, 2), 5.049751, 1e-4) && near(V(r, 3), 5.0, 1e-4) && near(V(r, 4), 4.950249, 1e-4),
              "floating node behind a 1 ms RC: ngspice's transient OP values 5.05 / 5.00 / 4.95 V");
    }
    {
        CircuitSim sim;
        TensorNetlist d;
        d.numGlobalNodes = 2;
        d.addVoltageSource(1, 0, 1.0); d.addResistor("R1", 1, 2, 1e3); d.addResistor("R2", 2, 0, 1e3);
        SolverStep s = sim.solveDC(d);
        check(s.stats.converged && !s.stats.gminSteppingUsed, "linear divider: no GMIN stepping reported");
    }

    // Source stepping (ngspice gillespie_src): with gmin stepping off, a 100-stage RTL
    // chain that plain Newton cannot solve (50 stages it can) converges by ramping the
    // sources; ngspice 45 aborts it with gminsteps=0 and matches with gmin stepping on
    {
        std::string net = "* rtl\n.options gminsteps=0\nVCC 1 0 DC 5\nVIN 2 0 DC 0\n";
        for (int i = 0; i < 100; ++i)
            net += "RB" + std::to_string(i) + " " + std::to_string(2 + i) + " b" + std::to_string(i) + " 10k\nQ" +
                   std::to_string(i) + " " + std::to_string(3 + i) + " b" + std::to_string(i) + " 0 NP\nRC" +
                   std::to_string(i) + " 1 " + std::to_string(3 + i) + " 1k\n";
        net += ".model NP NPN(IS=1e-15 BF=100)\n.op\n.end\n";
        auto p = SPICEParser().parse(net);
        CircuitSim sim;
        SolverStep s = sim.solveDC(p.netlist);
        check(p.success && s.stats.converged && s.stats.sourceSteppingUsed,
              "DC source stepping solves a 100-stage RTL chain with gminsteps=0 (converged " +
                  std::to_string(s.stats.converged) + ", source stepping " + std::to_string(s.stats.sourceSteppingUsed) + ")");
    }

    // Phase 3 MOS1: SPICE card defaults (KP 2e-5, VTO 0, W = L = 100u) give
    // Id = 0.5*2e-5*1^2 = 10 uA at Vgs = 1 V (saturation); M=3 triples it
    for (PrecisionMode mode : {PrecisionMode::Native_F64, PrecisionMode::Mixed}) {
        auto r = run(mode, "* mos defaults\nVDD 1 0 DC 5\nVG 2 0 DC 1\nRD 1 3 1k\nM1 3 2 0 0 NM\n.model NM NMOS\n.op\n");
        check(r.converged && near(V(r, 3), 5.0 - 1e3 * 1e-5, 1e-6),
              "MOS1 card defaults: V(drain) = 4.99 (" + std::to_string(V(r, 3)) + ")");
        r = run(mode, "* mos M\nVDD 1 0 DC 5\nVG 2 0 DC 1\nRD 1 3 1k\nM1 3 2 0 0 NM M=3\n.model NM NMOS\n.op\n");
        check(r.converged && near(V(r, 3), 5.0 - 3e3 * 1e-5, 1e-6),
              "MOS1 M=3 triples the current: V(drain) = 4.97 (" + std::to_string(V(r, 3)) + ")");
        r = run(mode, "* mos NF\nVDD 1 0 DC 5\nM1 1 1 0 0 NM NF=2\n.model NM NMOS\n.op\n");
        check(!r.converged && r.errorDetail.find("NF") != std::string::npos, "MOS1 NF=2 is an explicit error");
        // .lib section missing and an .include cycle are explicit errors (absolute paths: no baseDir here)
        {
            const std::string dir = std::filesystem::temp_directory_path().string();
            std::ofstream(dir + "/tf_corner.lib") << ".lib tt\n.model nm nmos\n.endl tt\n";
            std::ofstream(dir + "/tf_cycle.inc") << ".include " + dir + "/tf_cycle.inc\n";
            r = run(mode, "* lib\n.lib " + dir + "/tf_corner.lib ss\nV1 1 0 DC 1\nR1 1 0 1k\n.op\n");
            check(!r.converged && r.errorDetail.find("no section ss") != std::string::npos,
                  ".lib: a missing corner section is an explicit error (" + r.errorDetail + ")");
            r = run(mode, "* inc\n.include " + dir + "/tf_cycle.inc\nV1 1 0 DC 1\nR1 1 0 1k\n.op\n");
            check(!r.converged && r.errorDetail.find("cycle") != std::string::npos,
                  ".include: a cycle is an explicit error (" + r.errorDetail + ")");
        }
        // QA phase 3 (BUG-2..5): nonsense parameters, OFF/IC and mismatched cards are explicit errors
        const std::pair<const char*, const char*> bad[] = {
            {"V1 1 0 DC 3\nR1 1 2 1k\nQ1 2 2 0 QM\n.model QM NPN VAF=-5\n", "VAF must be >= 0"},
            {"V1 1 0 DC 3\nR1 1 2 1k\nM1 2 2 0 0 MM\n.model MM NMOS GAMMA=1 PHI=0\n", "PHI must be > 0"},
            {"V1 1 0 DC 1\nR1 1 2 1k\nD1 2 0 DM\n.model DM D BV=0\n", "BV must be > 0"},
            {"V1 1 0 DC 1\nR1 1 2 1k\nD1 2 0 DM\n.model DM D N=0\n", "N must be > 0"},
            {"V1 1 0 DC 1\nR1 1 2 1k\nM1 2 2 0 0 NM OFF\n.model NM NMOS LEVEL=54\n", "OFF and IC on a LEVEL 54 device are not supported yet"},
            {"V1 1 0 DC 1\nR1 1 2 1k\nM1 2 1 0 VM IC=1,1\n.model VM VDMOS\n", "OFF and IC on a VDMOS device are not supported yet"},
            {"V1 1 0 DC 1\nR1 1 2 1k\nM1 2 1 0 VM THERMAL\n.model VM VDMOS(RTHJC=1)\n", "or 5 (d g s tj tc) with THERMAL"},
            {"V1 1 0 DC 1\nR1 1 2 1k\nZ1 2 1 0 MZ\n.model MZ NMF(LEVEL=2)\n", "MESA) is not supported yet"},
            {"V1 1 0 DC 1\nR1 1 2 1k\nZ1 2 1 0 MZ\n.model MZ NJF\n", "not an NMF/PMF model"},
            {"V1 1 0 DC 1\nR1 1 2 1k\nM1 2 1 0 3 4 VM THERMAL\n.model VM VDMOS\n", "THERMAL needs RTHJC"},
            {"V1 1 0 DC 1\nR1 1 2 1k\nM1 2 1 0 VM W=1u\n.model VM VDMOS\n", "a VDMOS takes M, TEMP, DTEMP and THERMAL only"},
            {"V1 1 0 DC 1\nR1 1 2 1k\nQ1 2 2 0 NM\n.model NM NMOS\n", "is NMOS, not a NPN/PNP model"},
            {"V1 1 0 DC 1\nR1 1 2 1k\nM1 2 2 0 NM W=10u\n.model NM NMOS\n", "expected 4 nodes"},
            {"V1 1 0 DC 1\nR1 1 0 1k\na1 d1 y inv\n.model inv d_inverter\n", "supported in .tran only"},
            {"V1 1 0 DC 1\nR1 1 0 1k\na1 1 2 sg\n.model sg seegen\n", "SEEGEN is not supported yet"},
            {"V1 1 0 DC 1\nR1 1 2 1k\nD1 2 0 DV\n.model DV D(TT=1n VP=0.5 RSW=10)\n", "VP with RSW is not supported yet"},
            {"V1 1 0 DC 1\nR1 1 0 1k\na1 [q] src\n.model src d_source(input_file=\"no_such_dsource.txt\")\n", "cannot open d_source input_file"},
            {"V1 1 0 DC 1\nR1 1 0 1k\na1 [i] c NULL [q] sm\n.model sm d_state(state_file=\"no_such_state.txt\")\n", "cannot open d_state state_file"},
            {"V1 1 0 DC 1\nR1 1 0 1k\na1 d1 d2 d3 tri\n.model tri d_tristate(inertial_delay=true)\n", "inertial_delay on a d_tristate is not supported yet"},
            {"V1 1 0 DC 1\nR1 1 0 1k\na1 d1 y inv\n.model inv d_inverter(delay=1n)\n", "DELAY is not a D_INVERTER parameter"},
            {"V1 1 0 DC 1\nR1 1 2 1k\nM1 2 2 0 0 0 NM\n.model NM NMOS\n", "expected 4 nodes (d g s b) for a LEVEL 1"},
            {"V1 1 0 DC 1\nR1 1 2 1k\nM1 2 2 0 0 SN\n.model SN NMOS LEVEL=57\n", "(B3SOI) is not supported"},
            // QA round 2 and the parameter front end
            {"V1 1 0 DC 1\nR1 1 2 1k\nD1 2 0 DM\n.model DM D\n.model DM D IS=1e-15\n", "defined twice"},
            {"V1 1 0 DC {nosuch*2}\nR1 1 0 1k\n", "unknown parameter NOSUCH"},
            {".param a={b+1} b={a}\nV1 1 0 DC {a}\nR1 1 0 1k\n", "depends on itself"},
            {".if (1)\nV1 1 0 DC 1\nR1 1 0 1k\n", ".if without .endif"},
            {".subckt rr a b r=1k\nR1 a b {r}\n.ends\nV1 1 0 DC 1\nX1 1 0 rr q=2\n", "has no parameter Q"},
        };
        {  // Gummel-Poon AREA scales both junctions once: ngspice 47 takes the BC diode's IS times AREA twice (bjttemp:
           // tSatCur already holds AREA, then *= AREAB, which defaults to AREA; its IBC path and ISC scale once), so
           // in saturation its AREA = 2 collector current is -50.76 mA where twice AREA = 1's is -52.46 mA
            const std::string q = "* bjt area\nvb b 0 0.8\nvc c 0 0.1\nq1 c b 0 qm area=2\nq2 c2 b 0 qm\nvc2 c2 0 0.1\n"
                                  ".model qm npn is=1e-15 bf=100 br=2\n";
            const auto a = run(mode, q + ".op\n");
            bool ok = a.converged && !a.waveformPoints.empty() && a.waveformPoints.back().branchCurrents.size() >= 3;
            const double i1 = ok ? a.waveformPoints.back().branchCurrents[1] : 0.0, i2 = ok ? a.waveformPoints.back().branchCurrents[2] : 0.0;
            check(ok && near(i1, 2.0 * i2, 1e-9 * std::abs(i2)) && near(i2, -0.0262321676, 1e-6),
                  "BJT AREA = 2 is twice AREA = 1 in saturation (" + std::to_string(i1) + " vs " + std::to_string(i2) + " A)");
        }
        {  // ngspice's xfer warns in a transient and gives its DC gain: an error here
            const auto t = run(mode, "* xfer tran\nV1 1 0 DC 1\na1 1 2 xf\n.model xf xfer(table=[0 0 0 1k -3 -45])\nR1 2 0 1k\n", 1e-6, 1e-5);
            check(!t.converged && t.errorDetail.find("xfer code model has no transient") != std::string::npos,
                  "explicit error 'xfer code model has no transient' (" + t.errorDetail + ")");
        }
        for (const auto &[net, want] : bad) {
            r = run(mode, std::string("* bad\n") + net + ".op\n");
            check(!r.converged && r.errorDetail.find(want) != std::string::npos,
                  std::string("explicit error '") + want + "' (" + r.errorDetail + ")");
        }
        // OSDI (N devices) fails closed: missing library, no model card, card type not a loaded module
        r = run(mode, "* osdi lib\n.osdi no_such_model.osdi\nV1 1 0 DC 1\nR1 1 0 1k\n.op\n");
        check(!r.converged && r.errorDetail.find("not found") != std::string::npos,
              "OSDI: a missing .osdi library is an explicit error (" + r.errorDetail + ")");
        r = run(mode, "* osdi card\nV1 1 0 DC 1\nN1 1 0 0 0 nomodel\n.op\n");
        check(!r.converged && r.errorDetail.find(".model") != std::string::npos,
              "OSDI: an N device without a .model card is an explicit error (" + r.errorDetail + ")");
        r = run(mode, "* osdi module\nV1 1 0 DC 1\nN1 1 0 0 0 nch\n.model nch bsim4va type=1\n.op\n");
        check(!r.converged && r.errorDetail.find("OSDI module") != std::string::npos,
              "OSDI: a card whose type is no loaded module is an explicit error (" + r.errorDetail + ")");
        r = run(mode, "* mos LD\nVDD 1 0 DC 5\nM1 1 1 0 0 NM L=1u\n.model NM NMOS LD=0.5u\n.op\n");
        check(!r.converged && r.errorDetail.find("LD") != std::string::npos, "MOS1 Leff <= 0 is an explicit error");
        // TOX without KP: KP = UO * 1e-4 * 3.9 eps0 / TOX (ngspice), here 500 cm^2/Vs and 20 nm
        const double kp = 500e-4 * 3.9 * 8.854214871e-12 / 20e-9;
        r = run(mode, "* mos tox\nVDD 1 0 DC 5\nVG 2 0 DC 1\nRD 1 3 1k\nM1 3 2 0 0 NM\n.model NM NMOS TOX=20n UO=500\n.op\n");
        check(r.converged && near(V(r, 3), 5.0 - 1e3 * 0.5 * kp, 1e-6),
              "MOS1 KP from UO and TOX: V(drain) = " + std::to_string(5.0 - 0.5e3 * kp) + " (" + std::to_string(V(r, 3)) + ")");
        r = run(mode, "* mos mj\nVDD 1 0 DC 5\nM1 1 1 0 0 NM\n.model NM NMOS CJ=1e-4 MJ=1\n.op\n");
        check(!r.converged && r.errorDetail.find("MJ") != std::string::npos, "MOS1 MJ >= 1 is an explicit error");
    }

    // The CSR pattern cache must not trust the canonical topology hash: it ignores
    // sources and node numbering, so a reused CircuitSim kept a stale pattern and
    // silently dropped the moved source's stamps.
    {
        CircuitSim sim;
        TensorNetlist a, b;
        a.numGlobalNodes = b.numGlobalNodes = 2;
        a.addVoltageSource(1, 0, 1.0); a.addResistor("R1", 1, 2, 1e3); a.addResistor("R2", 2, 0, 1e3);
        b.addVoltageSource(2, 0, 1.0); b.addResistor("R1", 1, 2, 1e3); b.addResistor("R2", 2, 0, 1e3);
        sim.solveDC(a);
        SolverStep s = sim.solveDC(b);
        check(s.stats.converged && s.nodeVoltages.size() == 2 && near(s.nodeVoltages[0], 1.0, 1e-9) &&
              near(s.nodeVoltages[1], 1.0, 1e-9),
              "reused CircuitSim: moved source gives V1 = V2 = 1 (got " +
              (s.nodeVoltages.size() == 2 ? std::to_string(s.nodeVoltages[0]) + ", " + std::to_string(s.nodeVoltages[1]) : "none") + ")");
    }

    // Linear-time canonicalRelabel must reproduce the original (std::map, O(n^2)
    // BFS) hash: golden DB entries and cached traces store it. Value from the old code.
    {
        std::vector<GraphCanonical::CanonicalEdge> e;
        for (auto x : {std::array<int, 3>{0, 3, 0}, {3, 7, 1}, {3, 7, 0}, {7, 7, 2}, {7, 0, 3},
                       {5, 9, 0}, {9, 5, 4}, {2, 0, 5}})
            e.emplace_back(x[0], x[1], x[2]);
        check(GraphCanonical::computeTopologyHash(e, 0) == 3977363823923051078ULL,
              "topology hash unchanged by the linear-time relabel");
    }

    {   // .param expression semantics as ngspice 47 evaluates them (QA3 BUG-9)
        auto none = [](const std::string&, double&) { return false; };
        int bad = 0;
        for (const auto& [e, want] : std::vector<std::pair<const char*, double>>{
                 {"2**3**2", 64}, {"2^3^2", 64}, {"-2**2", -4}, {"2**-1", 0.5}, {"nint(2.5)", 2}, {"nint(-2.5)", -2},
                 {"nint(3.5)", 4}, {"pwr(-2,3)", 8}, {"pow(-2,3)", -8}, {"5mil", 5e-3}, {"sqr(3)", 9}, {"2<>3", 1},
                 {"-7%3", -1}, {"0?2:0?4:5", 5}, {"log(100)", std::log(100.0)}, {"temper", 27}, {"atanh(0.5)", std::atanh(0.5)},
                 {"limit(5,0,1)", 1}, {"limit(-5,0,1)", 0}, {"limit(5,2)", 5}})
            if (std::abs(numparam::eval(e, none) - want) > 1e-12 * std::abs(want)) { std::printf("    %s\n", e); ++bad; }
        for (const char* e : {"1/0", "ln(0)", "1e400", "sqrt(-1)", "10%0", "min(3,1,2)"}) {
            bool threw = false;
            try { numparam::eval(e, none); } catch (const std::exception&) { threw = true; }
            if (!threw) { std::printf("    %s accepted\n", e); ++bad; }
        }
        check(bad == 0, ".param expressions: ngspice associativity, rounding, pwr, mil, errors for non-finite results");
    }

    {   // Monte Carlo: samples differ, the same seed reproduces, the spread follows the distribution
        auto mcRun = [](uint32_t seed, const char* dist, double sigma, double tol) {
            auto engine = ISimulationEngine::create();
            auto session = engine->createSession();
            SimulationRequestDTO req;
            req.source.payload = "mc\nV1 1 0 10\nR1 1 2 1k\nR2 2 0 1k\n.end\n";
            MonteCarloConfigDTO mc;
            mc.sampleCount = 400; mc.seed = seed;
            mc.variations = {{"R1", "resistance", sigma, dist, 0.0, tol}};
            std::vector<double> v;
            for (const auto& p : session->runMonteCarlo(req, mc).waveformPoints) v.push_back(p.nodeVoltages[1]);
            return v;
        };
        const auto a = mcRun(7, "Gaussian", 0.05, 0), b = mcRun(7, "Gaussian", 0.05, 0), c = mcRun(8, "Gaussian", 0.05, 0);
        const auto u = mcRun(7, "Uniform", 0, 0.10);
        double lo = 1e9, hi = -1e9, mean = 0;
        for (double x : u) { lo = std::min(lo, x); hi = std::max(hi, x); mean += x / u.size(); }
        check(a.size() == 400 && a != std::vector<double>(400, a[0]), "Monte Carlo samples are not all identical");
        check(a == b && a != c, "Monte Carlo: same seed reproduces, another seed differs");
        // V2 = 10*R2/(R1+R2), R1 = 1k*(1 +/- 10%) -> V2 in [4.762, 5.263]
        check(lo >= 4.76 && hi <= 5.27 && hi - lo > 0.4 && near(mean, 5.0, 0.05), "Monte Carlo: uniform stays within tolerance and fills it");
        {   // a .dc TEMP sweep re-parses the netlist per temperature: each point keeps the sample's variation
            auto engine = ISimulationEngine::create();
            auto session = engine->createSession();
            SimulationRequestDTO req;
            req.source.payload = "mc\nV1 1 0 10\nR1 1 2 1k\nR2 2 0 1k\n.dc TEMP 0 100 100\n.end\n";
            req.mc.variations = {{"R1", "resistance", 0.05, "Gaussian", 0.0, 0.0}};
            req.mcSeed = 3;
            const auto w = session->runDC(req).waveformPoints;
            check(w.size() == 2 && w[0].nodeVoltages[1] == w[1].nodeVoltages[1] && std::abs(w[0].nodeVoltages[1] - 5.0) > 1e-6,
                  "Monte Carlo variation survives the .dc TEMP re-parse");
        }
        {   // S3.0: an SRAM written as nested subcircuits (cell -> row of n -> bank of n rows -> n banks) parses without
            // flattening into its hierarchy: O(definitions), 1e9 cells included; a hierarchical name resolves through
            // it; the flat path expands the same hierarchy into as many instances
            auto sram = [](int n) {
                std::string t = "* nested sram\n.subckt cell bl blb wl vdd\nM1 q qb 0 0 nm W=1u L=0.18u\nM2 q qb vdd vdd pm W=1u L=0.18u\n"
                                "M3 qb q 0 0 nm W=1u L=0.18u\nM4 qb q vdd vdd pm W=1u L=0.18u\nM5 bl wl q 0 nm W=0.5u L=0.18u\n"
                                "M6 blb wl qb 0 nm W=0.5u L=0.18u\n.ends\n.subckt row bl blb wl vdd\n";
                for (int i = 0; i < n; ++i) t += "XC" + std::to_string(i) + " bl blb wl vdd cell\n";
                t += ".ends\n.subckt bank bl blb vdd\n";
                for (int i = 0; i < n; ++i) t += "XR" + std::to_string(i) + " bl blb wl" + std::to_string(i) + " vdd row\n";
                t += ".ends\n.model nm nmos level=1 vto=0.5 kp=200u\n.model pm pmos level=1 vto=-0.5 kp=80u\nVDD vdd 0 1.8\n";
                for (int i = 0; i < n; ++i) t += "XB" + std::to_string(i) + " bl" + std::to_string(i) + " blb" + std::to_string(i) + " vdd bank\n";
                return t + ".op\n.end\n";
            };
            SPICEParser hp;
            hp.setFlatten(false);
            const auto t0 = std::chrono::steady_clock::now();
            const auto big = hp.parse(sram(1000));
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::vector<const HierInstance*> chain;
            int local = 0;
            const bool found = big.success && big.netlist.hier.resolve("XB7.XR3.XC999.Q", chain, local);
            check(big.success && big.netlist.instances.empty() && big.netlist.hier.expandedInstances() == 1e9 + 1e6 + 1e3 &&
                      found && chain.size() == 3 && chain.back()->def == "CELL" && local > 4 && secs < 5.0,
                  "S3.0: a 1e9-cell nested SRAM parses unflattened (" + std::to_string(secs) + " s), x1.x2.x3.q resolves" +
                      (big.success ? "" : ": " + big.errorMessage));
            SPICEParser fp, hp2;
            hp2.setFlatten(false);
            const auto flat = fp.parse(sram(6)), h6 = hp2.parse(sram(6));
            check(flat.success && h6.success && (double)flat.netlist.instances.size() == h6.netlist.hier.expandedInstances() &&
                      flat.netlist.instances.size() == 6 * 6 * 6 + 6 * 6 + 6 && flat.netlist.hier.top.empty(),
                  "S3.0: the flat path expands the hierarchy into its instance count (" + std::to_string(flat.netlist.instances.size()) + ")");
            SPICEParser xp;
            xp.setFlatten(false);
            const auto xs = xp.parse("* a in subckt\n.subckt buf a y\naa a y g\n.model g gain(gain=2)\n.ends\nV1 i 0 1\nX1 i o buf\nR1 o 0 1k\n.op\n.end\n");
            check(!xs.success && xs.errorMessage.find("need the flattened netlist") != std::string::npos,
                  "S3.0: XSPICE devices in a subcircuit need the flattened netlist (" + xs.errorMessage + ")");
        }
        {   // S3.1: the partition of an unflattened netlist is sound: a flat DC solve of the same circuit (unique: an
            // array of inverter cells) gives every net of a class one voltage. Bank 0 sees its own driver: two bank,
            // row and cell contexts; a B source is refused, not partitioned
            auto arr = [](bool b) {
                std::string t = "* array\n.subckt cell in vdd\nM1 mid in 0 0 nm W=1u L=0.18u\nM2 mid in vdd vdd pm W=2u L=0.18u\n"
                                "M3 out mid 0 0 nm W=1u L=0.18u\nM4 out mid vdd vdd pm W=2u L=0.18u\nR1 out 0 50k\n.ends\n"
                                ".subckt row in vdd\n";
                for (int i = 0; i < 4; ++i) t += "XC" + std::to_string(i) + " in vdd cell\n";
                t += ".ends\n.subckt bank in vdd\n";
                for (int i = 0; i < 4; ++i) t += "XR" + std::to_string(i) + " in vdd row\n";
                t += ".ends\n.model nm nmos level=1 vto=0.5 kp=200u\n.model pm pmos level=1 vto=-0.5 kp=80u\nVDD vdd 0 1.8\n"
                     "VIN in 0 0.7\nVIN2 in2 0 1.1\nR9 in 0 1meg\nR10 in2 0 1meg\nXB0 in2 vdd bank\n";
                for (int i = 1; i < 4; ++i) t += "XB" + std::to_string(i) + " in vdd bank\n";
                // tight tolerances: flat and quotient are separate Newton runs, equal to roundoff only when converged
                return t + (b ? "B1 x 0 V=v(in)\nR11 x 0 1k\n" : "") + ".options reltol=1e-10 vntol=1e-13 abstol=1e-16\n.op\n.end\n";
            };
            const auto flat = run(PrecisionMode::Native_F64, arr(false));
            SPICEParser hp;
            hp.setFlatten(false);
            const auto h = hp.parse(arr(false));
            bool ok = flat.converged && h.success && !flat.waveformPoints.empty();
            hierpart::Partition part;
            if (ok) part = hierpart::partition(h.netlist);
            std::map<std::pair<int, int>, double> classV;  // (context, color) -> the first flat voltage seen
            double worst = 0.0;
            size_t mapped = 0;
            for (size_t i = 0; ok && i < flat.nodeNames.size(); ++i) {
                std::string name = flat.nodeNames[i];
                for (auto& ch : name) ch = (char)std::toupper((unsigned char)ch);
                if (name.empty()) continue;
                int ctx = 0, local = -1;
                if (name.find('.') == std::string::npos) {
                    const auto it = h.nodeIndex.find(name);
                    if (it == h.nodeIndex.end()) continue;
                    local = it->second;
                } else {
                    std::vector<const HierInstance*> chain;
                    if (!h.netlist.hier.resolve(name, chain, local)) { ok = false; break; }
                    const std::vector<HierInstance>* level = &h.netlist.hier.top;
                    for (const auto* x : chain) {
                        const size_t k = (size_t)(x - level->data());
                        ctx = part.ctx[ctx].child[k];
                        const auto kids = h.netlist.hier.children.find(x->def);
                        level = kids == h.netlist.hier.children.end() ? nullptr : &kids->second;
                    }
                }
                const auto key = std::make_pair(ctx, part.ctx[ctx].net[local]);
                const double v = flat.waveformPoints.back().nodeVoltages[i];
                if (auto it = classV.find(key); it != classV.end()) worst = std::max(worst, std::abs(it->second - v));
                else classV[key] = v;
                ++mapped;
            }
            check(ok && mapped > 100 && worst < 1e-9 && part.ctx.size() == 7,
                  "S3.1: every net of a class has one flat voltage (worst " + std::to_string(worst) + " V over " +
                      std::to_string(mapped) + " nets; " + std::to_string(part.ctx.size()) + " contexts, " +
                      std::to_string(classV.size()) + " classes)");
            // S3.2: the quotient circuit solves to the flat one's voltages: every flat node against its class's node
            hierpart::Partition qp;
            const auto qn = hierpart::quotient(h.netlist, &qp);
            CircuitSim qsim;
            auto qnl = qn.nl;
            const auto qs = qsim.solveDC(qnl);
            double qworst = 0.0;
            size_t qmapped = 0;
            for (size_t i = 0; qs.stats.converged && i < flat.nodeNames.size(); ++i) {
                std::string name = flat.nodeNames[i];
                for (auto& ch : name) ch = (char)std::toupper((unsigned char)ch);
                const int qnode = name.empty() ? -1 : hierpart::nodeFor(h.netlist, h.nodeIndex, qp, qn, name);
                if (qnode < 0) continue;
                const double qv = qnode == 0 ? 0.0 : qs.nodeVoltages[qnode - 1];
                qworst = std::max(qworst, std::abs(qv - flat.waveformPoints.back().nodeVoltages[i]));
                ++qmapped;
            }
            check(qs.stats.converged && qmapped > 100 && qworst < 1e-9,
                  "S3.2: the quotient circuit (" + std::to_string(qn.nl.numGlobalNodes) + " nodes, " + std::to_string(qn.elementClasses) +
                      " elements) gives every flat node its voltage (worst " + [&] { char b[32]; std::snprintf(b, 32, "%.3g", qworst); return std::string(b); }() + " V over " +
                      std::to_string(qmapped) + ")");
            SPICEParser bp;
            bp.setFlatten(false);
            const auto hb = bp.parse(arr(true));
            std::string err;
            try { hierpart::partition(hb.netlist); } catch (const std::exception& e) { err = e.what(); }
            check(hb.success && err.find("not supported unflattened") != std::string::npos,
                  "S3.1: a coupling the refinement does not model is refused (" + err + ")");
        }
        {   // S3.2 transient: a 6T SRAM writes a 0 into bank 0's row 0 (its word line pulses, its bit lines driven), every
            // cell initialised to 1 by the body's .ic; the quotient circuit runs the flat one's transient, node for
            // node at the same accepted steps. Column 0 (written) shares blb with two columns on a pulled-up bl: a
            // cell's colors must not merge with a neighbour's whose bl differs (qb is two hops from it)
            std::string t = "* sram write\n.subckt cell bl blb wl vdd\nM1 q qb 0 0 nm W=1u L=0.18u\nM2 q qb vdd vdd pm W=1u L=0.18u\n"
                            "M3 qb q 0 0 nm W=1u L=0.18u\nM4 qb q vdd vdd pm W=1u L=0.18u\nM5 bl wl q 0 nm W=1u L=0.18u\n"
                            "M6 blb wl qb 0 nm W=1u L=0.18u\nCq q 0 2f\nCqb qb 0 2f\n.ic v(q)=1.8 v(qb)=0\n.ends\n"
                            ".subckt row bl blb wl vdd cbl\nXC0 bl blb wl vdd cell\nXC1 cbl blb wl vdd cell\nXC2 cbl blb wl vdd cell\n.ends\n"
                            ".subckt bank bl blb vdd\nXR0 bl blb wl0 vdd cbl row\nXR1 bl blb wl1 vdd cbl row\nXR2 bl blb wl2 vdd cbl row\n"
                            "V0 wl0 0 0\nV1 wl1 0 0\nV2 wl2 0 0\nRC cbl vdd 1k\nCC cbl 0 10f\n.ends\n"
                            ".subckt bankw bl blb vdd\nXR0 bl blb wl0 vdd cbl row\nXR1 bl blb wl1 vdd cbl row\nXR2 bl blb wl2 vdd cbl row\n"
                            "V0 wl0 0 PULSE(0 1.8 0.2n 0.05n 0.05n 1n 4n)\nV1 wl1 0 0\nV2 wl2 0 0\nRC cbl vdd 1k\nCC cbl 0 10f\n.ends\n"
                            ".model nm nmos level=1 vto=0.5 kp=200u cgso=1e-10 cgdo=1e-10\n.model pm pmos level=1 vto=-0.5 kp=80u cgso=1e-10 cgdo=1e-10\n"
                            "VDD vdd 0 1.8\nVBL bl0 0 0\nVBLB blb0 0 1.8\nXB0 bl0 blb0 vdd bankw\n";
            for (int i = 1; i < 3; ++i)
                t += "VBL" + std::to_string(i) + " bl" + std::to_string(i) + " 0 1.8\nVBLB" + std::to_string(i) + " blb" + std::to_string(i) +
                     " 0 1.8\nXB" + std::to_string(i) + " bl" + std::to_string(i) + " blb" + std::to_string(i) + " vdd bank\n";
            t += ".tran 0.02n 1.5n\n.end\n";
            auto steps = [](TensorNetlist& nl, std::vector<std::pair<double, std::vector<double>>>& out) {
                CircuitSim sim;
                return sim.runTransient(nl, 0.02e-9, 1.5e-9, false, [&](const SolverStep& st) {
                    out.push_back({st.time, st.nodeVoltages});
                    return true;
                });
            };
            auto flatP = SPICEParser().parse(t);
            SPICEParser hp;
            hp.setFlatten(false);
            auto hierP = hp.parse(t);
            std::vector<std::pair<double, std::vector<double>>> fw, qw;
            std::string fe = flatP.success ? steps(flatP.netlist, fw) : flatP.errorMessage, qe = "parse";
            hierpart::Partition tp;
            hierpart::Quotient tq;
            if (hierP.success) {
                tq = hierpart::quotient(hierP.netlist, &tp);
                qe = steps(tq.nl, qw);
            }
            // flat node -> quotient node, by name
            std::vector<int> qOf(flatP.netlist.numGlobalNodes + 1, -1);
            for (const auto& [name, idx] : flatP.nodeIndex)
                if (idx >= 1 && idx <= flatP.netlist.numGlobalNodes) qOf[idx] = hierpart::nodeFor(hierP.netlist, hierP.nodeIndex, tp, tq, name);
            // the same accepted steps: step k of each at one time (to roundoff: the step control sees roundoff-level
            // differences in the voltages)
            double worst = 0.0, swing = 0.0, dt = 0.0;
            size_t common = 0;
            for (size_t k = 0; k < fw.size() && k < qw.size() && fe.empty() && qe.empty(); ++k) {
                dt = std::max(dt, std::abs(fw[k].first - qw[k].first) / std::max(fw[k].first, 1e-30));
                ++common;
                for (int n = 1; n <= flatP.netlist.numGlobalNodes; ++n)
                    if (qOf[n] > 0) {
                        worst = std::max(worst, std::abs(fw[k].second[n - 1] - qw[k].second[qOf[n] - 1]));
                        swing = std::max(swing, std::abs(fw[k].second[n - 1]));
                    }
            }
            char msg[256];
            std::snprintf(msg, sizeof msg, "S3.2 transient: the SRAM write's quotient (%d nodes for %d) runs flat's %zu steps (%zu; "
                          "times within %.2g), worst %.3g V", tq.nl.numGlobalNodes, flatP.netlist.numGlobalNodes, fw.size(), qw.size(), dt, worst);
            check(fe.empty() && qe.empty() && common == fw.size() && fw.size() == qw.size() && fw.size() > 20 && dt < 1e-12 && worst < 1e-9 && swing > 1.0,
                  std::string(msg) + (fe.empty() ? "" : " flat: " + fe) + (qe.empty() ? "" : " quotient: " + qe));
            // the same through the session: .options hier runs the quotient, its columns named by the outputs (a
            // hierarchical internal node and top-level ones); .op and .tran, flat against hierarchical
            auto session = [&](const std::string& net, bool tran) {
                auto engine = ISimulationEngine::create();
                SimulationRequestDTO q;
                q.source.payload = net;
                EngineCallbacks c;
                if (!tran) return engine->createSession()->runDC(q, c);
                q.transient.stepSizeS = 0.02e-9;
                q.transient.stopTimeS = 1.5e-9;
                return engine->createSession()->runTransient(q, c, nullptr);
            };
            const std::string saves = ".save v(bl0) v(blb1) v(xb0.xr0.xc0.q) v(xb2.xr1.xc2.qb) i(vbl1)\n";
            std::string flatNet = t, hierNet = t;
            flatNet.insert(flatNet.rfind(".tran"), saves);
            hierNet.insert(hierNet.rfind(".tran"), saves + ".options hier\n");
            for (bool tran : {false, true}) {
                const auto f = session(flatNet, tran), h = session(hierNet, tran);
                bool same = f.converged && h.converged && f.waveformPoints.size() == h.waveformPoints.size() && !f.waveformPoints.empty() &&
                            f.nodeNames.size() == 4 && f.branchNames.size() == 1;
                double w = 0.0;
                for (size_t c = 0; same && c < f.nodeNames.size(); ++c) {
                    const auto it = std::find(h.nodeNames.begin(), h.nodeNames.end(), f.nodeNames[c]);
                    same = it != h.nodeNames.end();
                    for (size_t k = 0; same && k < f.waveformPoints.size(); ++k)
                        w = std::max(w, std::abs(f.waveformPoints[k].nodeVoltages[c] - h.waveformPoints[k].nodeVoltages[it - h.nodeNames.begin()]));
                }
                same = same && h.branchNames == f.branchNames;
                for (size_t k = 0; same && k < f.waveformPoints.size(); ++k)
                    w = std::max(w, std::abs(f.waveformPoints[k].branchCurrents[0] - h.waveformPoints[k].branchCurrents[0]) * 1e3);
                check(same && w < 1e-9, std::string("S3.2 session: .options hier ") + (tran ? ".tran" : ".op") +
                          " reports the flat run's saved vectors (" + std::to_string(f.waveformPoints.size()) + " points, worst " +
                          [&] { char b2[32]; std::snprintf(b2, 32, "%.3g", w); return std::string(b2); }() + ")" + (h.converged ? "" : ": " + h.errorDetail));
            }
            {   // S3.3: an .ic by path holds one instance's cell (only its chain splits from its class); an unknown path is an error
                const std::string ic = ".ic v(xb2.xr1.xc2.q)=0 v(xb2.xr1.xc2.qb)=1.8\n";
                std::string fi = flatNet, hi = hierNet;
                fi.insert(fi.rfind(".tran"), ic);
                hi.insert(hi.rfind(".tran"), ic);
                const auto f = session(fi, true), h = session(hi, true);
                const auto col = [](const SimulationResponseDTO& r, const std::string& n) {
                    return (size_t)(std::find(r.nodeNames.begin(), r.nodeNames.end(), n) - r.nodeNames.begin());
                };
                bool same = f.converged && h.converged && f.waveformPoints.size() == h.waveformPoints.size() && !f.waveformPoints.empty();
                double w = 0.0;
                for (const auto& n : f.nodeNames) {
                    const size_t a = col(f, n), b = col(h, n);
                    same = same && b < h.nodeNames.size();
                    for (size_t k = 0; same && k < f.waveformPoints.size(); ++k)
                        w = std::max(w, std::abs(f.waveformPoints[k].nodeVoltages[a] - h.waveformPoints[k].nodeVoltages[b]));
                }
                const size_t qb = col(h, "xb2.xr1.xc2.qb");
                const bool held = same && qb < h.nodeNames.size() && h.waveformPoints.front().nodeVoltages[qb] > 1.7;
                std::string bad = hierNet;
                bad.insert(bad.rfind(".tran"), ".ic v(xb2.xr9.q)=0\n");
                const auto u = session(bad, true);
                check(same && held && w < 1e-9 && !u.converged && u.errorDetail.find("names unknown node 'XB2.XR9.Q'") != std::string::npos,
                      "S3.3: an .ic by path sets one instance's cell on the hierarchical path as flat (worst " +
                          [&] { char b2[32]; std::snprintf(b2, 32, "%.3g", w); return std::string(b2); }() + " V; " + u.errorDetail + ")");
            }
            {   // refused on the hierarchical path: .pz (the quotient lacks the modes symmetry hides), sweeping a source its
                // class shares (vbl1 = vbl2)
                std::string pzNet = hierNet;
                pzNet.replace(pzNet.rfind(".tran"), std::string::npos, ".pz bl0 0 blb0 0 vol pz\n.end\n");
                auto engine = ISimulationEngine::create();
                SimulationRequestDTO q;
                q.source.payload = pzNet;
                EngineCallbacks c;
                const auto ra = engine->createSession()->runPoleZero(q, c);
                std::string dcNet = hierNet;
                dcNet.replace(dcNet.rfind(".tran"), std::string::npos, ".dc VBL1 0 1.8 0.9\n.end\n");
                q.source.payload = dcNet;
                const auto rd = engine->createSession()->runDC(q, c);
                std::string okNet = hierNet;
                okNet.replace(okNet.rfind(".tran"), std::string::npos, ".dc VDD 1.6 1.8 0.1\n.end\n");
                q.source.payload = okNet;
                const auto ro = engine->createSession()->runDC(q, c);
                check(!ra.converged && ra.errorDetail.find("hierarchical path") != std::string::npos && !rd.converged &&
                          rd.errorDetail.find("stands for identical") != std::string::npos && ro.converged && ro.waveformPoints.size() == 3,
                      "S3.2 session: .pz and a shared source's sweep are refused, a unique source's .dc runs (" + ra.errorDetail + " | " +
                          rd.errorDetail + ")");
            }
        }
        {   // S3.3: .op, .ac, .noise and .tf on the quotient. Four identical BJT/JFET stages (classes of four I, L, Q and J,
            // whose multipliers carry the count) are summed into one node; every result equals the flat run's. Noise there
            // is each member's power (the class's multiplier) through each member's transfer (the adjoint times the
            // row's 1/class size). A stage's own output (a class of four nets) cannot be measured alone.
            const std::string amp = "* s3.3\n.subckt stage in out vcc\nQ1 c in e qn\nRC vcc c 5k\nRE e e2 500\nL1 e2 0 1u\nIB vcc c 10u\n"
                                    "J1 vcc c out jn\nRS out 0 2k\nC1 out 0 1p\n.ends\n"
                                    ".model qn npn bf=100 is=1e-15 rb=100 cje=1p cjc=0.5p tf=1n kf=1e-16\n"
                                    ".model jn njf vto=-2 beta=1e-3 lambda=0.01 cgs=1p cgd=1p kf=1e-17\n"
                                    "VCC vcc 0 5\nVIN in 0 DC 0.7 AC 1\nX1 in o1 vcc stage\nX2 in o2 vcc stage\nX3 in o3 vcc stage\n"
                                    "X4 in o4 vcc stage\nR1 o1 sum 10k\nR2 o2 sum 10k\nR3 o3 sum 10k\nR4 o4 sum 10k\nRL sum 0 10k\n"
                                    ".save v(sum) v(o2) v(x3.c) v(x1.e2) i(vcc)\n";
            auto run = [&](bool hier, const std::string& card, int kind) {  // 0 .op, 1 .ac, 2 .noise, 3 .tf
                auto engine = ISimulationEngine::create();
                SimulationRequestDTO q;
                q.source.payload = amp + (hier ? ".options hier\n" : "") + card + "\n.end\n";
                EngineCallbacks c;
                auto s = engine->createSession();
                return kind == 0 ? s->runDC(q, c) : kind == 1 ? s->runAC(q, c) : kind == 2 ? s->runNoise(q, c) : s->runTF(q, c);
            };
            auto rel = [](double a, double b) { return std::abs(a - b) / std::max(std::abs(a) + std::abs(b), 1e-300); };
            double worst = 0.0;
            bool ok = true;
            for (int kind : {0, 1}) {  // the saved vectors, by name (real and imaginary)
                const char* card = kind == 0 ? ".op" : ".ac dec 2 1k 1g";
                const auto f = run(false, card, kind), h = run(true, card, kind);
                ok = ok && f.converged && h.converged && f.waveformPoints.size() == h.waveformPoints.size() && f.nodeNames.size() == 4;
                for (size_t n = 0; ok && n < f.nodeNames.size(); ++n) {
                    const auto it = std::find(h.nodeNames.begin(), h.nodeNames.end(), f.nodeNames[n]);
                    ok = it != h.nodeNames.end();
                    const size_t hn = it - h.nodeNames.begin();
                    for (size_t k = 0; ok && k < f.waveformPoints.size(); ++k) {
                        const auto &a = f.waveformPoints[k], &b = h.waveformPoints[k];
                        worst = std::max(worst, rel(a.nodeVoltages[n], b.nodeVoltages[hn]));
                        if (kind == 1) worst = std::max(worst, rel(a.nodeVoltagesImag[n], b.nodeVoltagesImag[hn]));
                    }
                }
                for (size_t k = 0; ok && k < f.waveformPoints.size(); ++k)
                    worst = std::max(worst, rel(f.waveformPoints[k].branchCurrents.at(0), h.waveformPoints[k].branchCurrents.at(0)));
            }
            const auto fn = run(false, ".noise v(sum) vin dec 2 1k 1g", 2), hn = run(true, ".noise v(sum) vin dec 2 1k 1g", 2);
            ok = ok && fn.converged && hn.converged && fn.extended.noise.onoise.size() == hn.extended.noise.onoise.size();
            for (size_t k = 0; ok && k < fn.extended.noise.onoise.size(); ++k)
                worst = std::max({worst, rel(fn.extended.noise.onoise[k], hn.extended.noise.onoise[k]),
                                  rel(fn.extended.noise.inoise[k], hn.extended.noise.inoise[k])});
            if (ok) worst = std::max({worst, rel(fn.extended.noise.onoiseTotal, hn.extended.noise.onoiseTotal),
                                      rel(fn.extended.noise.inoiseTotal, hn.extended.noise.inoiseTotal)});
            const auto ft = run(false, ".tf v(sum) vin", 3), ht = run(true, ".tf v(sum) vin", 3);
            ok = ok && ft.converged && ht.converged && ft.extended.dcTf.values.size() == 3 && ht.extended.dcTf.names == ft.extended.dcTf.names;
            for (size_t k = 0; ok && k < 3; ++k) worst = std::max(worst, rel(ft.extended.dcTf.values[k], ht.extended.dcTf.values[k]));
            const auto refused = run(true, ".noise v(o1) vin dec 2 1k 1g", 2);
            char msg[160];
            std::snprintf(msg, sizeof msg, "S3.3: .op, .ac, .noise and .tf on the quotient equal flat (worst relative %.2g); a shared "
                          "noise output is refused", worst);
            check(ok && worst < 1e-9 && !refused.converged && refused.errorDetail.find("stands for 4 identical nets") != std::string::npos,
                  std::string(msg) + " (" + fn.errorDetail + hn.errorDetail + ft.errorDetail + ht.errorDetail + " | " + refused.errorDetail + ")");
        }
        {   // a corner runs only what the engine models: its supply scale; a process corner is the PDK's .lib section
            CircuitSim sim;
            TensorNetlist nl;
            acutesim::physics::CornerConfig c;
            c.name = "ff";
            const std::string ff = sim.applyCorner(nl, c);
            c.name = "TT";
            c.voltageScale = 0.0;
            const std::string zero = sim.applyCorner(nl, c);
            c.voltageScale = 1.1;
            const std::string tt = sim.applyCorner(nl, c);
            check(ff.find(".lib section") != std::string::npos && !zero.empty() && tt.empty() &&
                      nl.environment.global_voltage_scale == 1.1,
                  "applyCorner refuses a process corner and a non-positive supply, applies TT's scale (" + ff + " | " + zero + ")");
        }
        {   // S3.3: the metastable warning names a latch's q / qb (two identical inverters couple them in different roles),
            // not nets driven alike (a resistor-driven bus that one instance meets at interchangeable pins)
            auto warned = [&](const std::string& net) {
                std::ostringstream err;
                auto* old = std::cerr.rdbuf(err.rdbuf());
                auto engine = ISimulationEngine::create();
                SimulationRequestDTO q;
                q.source.payload = net;
                EngineCallbacks c;
                engine->createSession()->runDC(q, c);
                std::cerr.rdbuf(old);
                return err.str();
            };
            const std::string inv = ".subckt inv in out vdd\nM1 out in 0 0 nm W=1u L=0.18u\nM2 out in vdd vdd pm W=1u L=0.18u\n.ends\n"
                                    ".model nm nmos level=1 vto=0.5 kp=200u\n.model pm pmos level=1 vto=-0.5 kp=80u\nVDD vdd 0 1.8\n";
            const auto latch = warned("* latch\n" + inv + ".subckt latch vdd\nX1 q qb vdd inv\nX2 qb q vdd inv\n.ends\nXL1 vdd latch\n"
                                      "XL2 vdd latch\n.options hier\n.op\n.end\n");
            const auto bus = warned("* bus\n" + inv + ".subckt pair a b vdd\nX1 a oa vdd inv\nX2 b ob vdd inv\n.ends\n"
                                    ".subckt top vdd\nXP b0 b1 vdd pair\nVB p 0 1\nR0 p b0 1k\nR1 p b1 1k\n.ends\nXT vdd top\n.options hier\n.op\n.end\n");
            check(latch.find("nets Q QB of LATCH are equal by symmetry") != std::string::npos && bus.find("equal by symmetry") == std::string::npos,
                  "S3.3: the metastable warning names a latch's coupled pair, not a bus driven alike (" + latch.substr(0, latch.find('\n')) +
                      " | " + bus.substr(0, bus.find('\n')) + ")");
        }
        {   // .save narrows what a run stores to the vectors .save, .print and .meas name (ngspice saves only those);
            // .save all, or none, keeps every node and branch
            auto tr = [&](const char* save) {
                auto engine = ISimulationEngine::create();
                SimulationRequestDTO q;
                q.source.payload = std::string("save\nV1 a 0 PULSE(0 1 0 1u 1u 5u 20u)\nR1 a b 1k\nR2 b out 1k\nC1 out 0 1n\n"
                                               "R3 out 0 10k\n.meas tran vb max v(b)\n") + save + "\n.tran 0.5u 10u\n.end\n";
                q.transient.stepSizeS = 0.5e-6;
                q.transient.stopTimeS = 1e-5;
                EngineCallbacks c;
                return engine->createSession()->runTransient(q, c, nullptr);
            };
            const auto full = tr(""), sel = tr(".save v(out) i(v1)"), all = tr(".save all v(out)");
            const bool ok = full.converged && sel.converged && all.converged && !sel.waveformPoints.empty() &&
                            sel.waveformPoints.size() == full.waveformPoints.size();
            const auto& p = ok ? sel.waveformPoints.back() : WaveformDataPoint{};
            const auto& f = ok ? full.waveformPoints.back() : WaveformDataPoint{};
            // full columns: a b out / v1; .save v(out) i(v1) keeps out (and b, which .meas reads), v1
            check(ok && sel.nodeNames == std::vector<std::string>{"b", "out"} && sel.branchNames == std::vector<std::string>{"v1"} &&
                      p.nodeVoltages.size() == 2 && p.nodeVoltages[1] == f.nodeVoltages[2] && p.branchCurrents[0] == f.branchCurrents[0] &&
                      all.nodeNames.size() == 3 && !sel.measures.empty() && !full.measures.empty() &&
                      sel.measures[0].value == full.measures[0].value,
                  ".save keeps the vectors .save, .print and .meas name; .save all keeps every one");
        }
        {   // OSDI devices with the same model and parameters share one setup; .options bypass=0 (memory mode)
            // keeps no copy of each device's last evaluation and evaluates always: the same transient, to
            // within bypass's extrapolation (RELTOL 1e-3 of the step's voltage change)
            auto inv = [&](const char* opt) {
                const std::string n = std::string("osdi share\nVDD vdd 0 1.8\nVIN in 0 PULSE(0 1.8 0.2n 0.1n 0.1n 1n 2.4n)\n"
                                      "M1 o1 in 0 0 nm W=1u L=0.18u\nM2 o1 in vdd vdd pm W=2u L=0.18u\n"
                                      "M3 o2 o1 0 0 nm W=1u L=0.18u\nM4 o2 o1 vdd vdd pm W=2u L=0.18u\nC1 o2 0 5f\n"
                                      ".model nm nmos level=54 version=4.8\n.model pm pmos level=54 version=4.8\n") + opt + "\n.tran 0.02n 0.3n\n.end\n";
                const auto r = run(PrecisionMode::Native_F64, n, 0.02e-9, 0.3e-9);
                return r.converged && !r.waveformPoints.empty() ? V(r, 4) : NAN;  // o2, mid-rise
            };
            const double dflt = inv(""), mem = inv(".options bypass=0");
            check(dflt > 0.1 && dflt < 1.0 && near(dflt, mem, 2e-3),
                  "OSDI memory mode (.options bypass=0) runs the default's transient (" + std::to_string(dflt) + " vs " + std::to_string(mem) + " V)");
        }
        {   // HICUM/L2 (BJT LEVEL 8, OSDI): a PNP mirrors the NPN. ngspice 47's native HICUM PNP cannot be the
            // reference: its temperature limiter gets NaN and the operating point fails (the OSDI model in ngspice agrees).
            const char* card = "c10=2e-30 qp0=2e-14 ich=1e-2 hfe=100 hfc=10 cjei0=1e-14 vdei=0.9 zei=0.5 rbi0=10 rbx=5 rcx=5 re=2 "
                               "t0=1e-12 cjci0=5e-15 vdci=0.7 zci=0.4 ibeis=1e-18 mbei=1 ireis=1e-16 mrei=2 rth=200 cth=1e-12 "
                               "favl=1 qavl=5e-14 cjs0=3e-15 itss=1e-20 rsu=100 csu=1e-15 flsh=1";
            auto ic = [&](const char* type, double s) {
                std::string n = std::string("hicum\nVC c 0 ") + std::to_string(2 * s) + "\nVB b 0 " + std::to_string(0.85 * s) +
                                "\nQ1 c b 0 q\n.model q " + type + " level=8 " + card + "\n.op\n.end\n";
                const auto r = run(PrecisionMode::Native_F64, n);
                return r.converged && !r.waveformPoints.empty() && !r.waveformPoints.back().branchCurrents.empty()
                           ? r.waveformPoints.back().branchCurrents[0] : NAN;
            };
            const double n = ic("npn", 1.0), p = ic("pnp", -1.0);
            check(n < -1e-3 && near(p, -n, 1e-9 * std::abs(n)), "HICUM PNP mirrors the NPN (" + std::to_string(n) + " A)");
            const auto a = run(PrecisionMode::Native_F64, std::string("hicum\nVC c 0 2\nVB b 0 0.8\nQ1 c b 0 q area=2\n.model q npn level=8 ") + card + "\n.op\n.end\n");
            check(!a.converged && a.errorDetail.find("AREA/M on a HICUM device") != std::string::npos,
                  "HICUM AREA is an explicit error (" + a.errorDetail + ")");
        }
        {   // A MESFET's positional area is its AREA (ngspice 47's INP2Z drops it for MES, keeping area= only)
            const auto r = run(PrecisionMode::Native_F64, "z\nVDD 1 0 DC 3\nRA 1 2 1k\nRB 1 3 1k\nRC 1 4 1k\n"
                               "Z1 2 0 0 mn 2\nZ2 3 0 0 mn area=2\nZ3 4 0 0 mn\n.model mn NMF(VTO=-1.5 BETA=1e-3)\n.op\n.end\n");
            check(r.converged && near(V(r, 2), V(r, 3), 1e-9) && !near(V(r, 2), V(r, 4), 1e-3),
                  "MESFET positional area is AREA (" + std::to_string(V(r, 2)) + " V)");
        }
        {   // m= on a subcircuit multiplies its OSDI devices too: an m=3 cell keeps the m=1 cell's voltages
            // (ngspice 47 multiplies only the non-OSDI devices, so it cannot be the reference)
            const auto r = run(PrecisionMode::Native_F64, "m\n.osdi bsimcmg.osdi\n.subckt inv in out vdd\nN1 out in 0 0 nfin\n"
                               "RL vdd out 20k\n.ends\nVDD 1 0 DC 0.8\nVG 2 0 DC 0.5\nX1 2 3 1 inv m=3\nX2 2 4 1 inv\n"
                               ".model nfin bsimcmg_va type=1\n.op\n.end\n");
            check(r.converged && near(V(r, 3), V(r, 4), 1e-6) && V(r, 3) < 0.77,
                  "m=3 on a cell with an OSDI device keeps the cell's voltages (" + std::to_string(V(r, 3)) + " V)");
        }
        {   // OFF picks a symmetric latch's state: the OFF transistor ends off. ngspice 47's Newton fails with the device
            // held and its gmin stepping ignores OFF, so it lands on the metastable 1.186 V / 1.186 V point either way.
            auto latch = [&](const char* q1, const char* q2) {
                const auto r = run(PrecisionMode::Native_F64, std::string("latch\nVCC 1 0 5\nRC1 1 2 1k\nRC2 1 3 1k\nRB1 3 4 10k\n"
                                   "RB2 2 5 10k\nQ1 2 4 0 qn") + q1 + "\nQ2 3 5 0 qn" + q2 + "\n.model qn npn bf=100\n.op\n.end\n");
                return std::make_pair(V(r, 2), V(r, 3));
            };
            const auto a = latch(" OFF", ""), b = latch("", " OFF");
            check(a.first > 4.5 && a.second < 0.1 && b.first < 0.1 && b.second > 4.5, "OFF selects a symmetric latch's state");
        }
        {   // a diode's IC= under UIC starts its charge there (ngspice 47 drops it; its BJT and JFET honour theirs)
            auto v2 = [&](const char* ic) {
                SimulationRequestDTO req;
                req.source.payload = std::string("ic\nV1 1 0 DC 0\nR1 1 2 1k\nD1 2 0 dm ") + ic +
                                     "\n.model dm d is=1e-14 cjo=2p tt=5n\n.tran 0.1n 5n uic\n.end\n";
                auto engine = ISimulationEngine::create();
                const auto r = engine->createSession()->runTransient(req, {}, nullptr);
                return r.converged && r.waveformPoints.size() > 2 ? r.waveformPoints[2].nodeVoltages[1] : NAN;
            };
            const double with = v2("IC=0.6"), without = v2("");
            check(with > 0.4 && std::abs(without) < 1e-3, "diode IC=0.6 under UIC holds its node at first (" + std::to_string(with) + " V)");
        }
        {   // .tf i(vsrc): the true output impedance (-1/i), where ngspice prints 1/max(1e-20, i) = 1e20 (passive) or +1k (active)
            auto engine = ISimulationEngine::create();
            auto session = engine->createSession();
            SimulationRequestDTO req;
            auto zout = [&](const std::string& load) {
                req.source.payload = "tf\nV1 1 0 DC 1\nR1 1 2 1k\n" + load + "Vm 2 0 0\n.tf i(Vm) V1\n.end\n";
                const auto t = session->runTF(req).extended.dcTf;
                return t.names.size() == 3 && t.names[2] == "vm#output_impedance" ? t.values[2] : NAN;
            };
            check(near(zout(""), 1e3, 1e-6) && near(zout("G1 2 0 2 0 -2m\n"), -1e3, 1e-6),
                  ".tf i(Vm) output impedance: 1k (passive), -1k (negative-resistance load)");
        }
    }

    std::printf("\n%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
