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

#include <array>
#include <cmath>
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
        r = run(mode, "* unknown\nV1 1 0 5\nZ1 1 0 5\n.op\n");
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
            {"V1 1 0 DC 1\nR1 1 2 1k\nD1 2 0 DM OFF\n.model DM D\n", "OFF is not supported"},
            {"V1 1 0 DC 1\nR1 1 2 1k\nQ1 2 2 0 QM IC=0.6,5\n.model QM NPN\n", "IC is not supported"},
            {"V1 1 0 DC 1\nR1 1 2 1k\nQ1 2 2 0 NM\n.model NM NMOS\n", "is NMOS, not a NPN/PNP model"},
            {"V1 1 0 DC 1\nR1 1 2 1k\nM1 2 2 0 NM W=10u\n.model NM NMOS\n", "expected 4 nodes"},
            // QA round 2 and the parameter front end
            {"V1 1 0 DC 1\nR1 1 2 1k\nD1 2 0 DM\n.model DM D\n.model DM D IS=1e-15\n", "defined twice"},
            {"V1 1 0 DC {nosuch*2}\nR1 1 0 1k\n", "unknown parameter NOSUCH"},
            {".param a={b+1} b={a}\nV1 1 0 DC {a}\nR1 1 0 1k\n", "depends on itself"},
            {".if (1)\nV1 1 0 DC 1\nR1 1 0 1k\n", ".if without .endif"},
            {".subckt rr a b r=1k\nR1 a b {r}\n.ends\nV1 1 0 DC 1\nX1 1 0 rr q=2\n", "has no parameter Q"},
        };
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
    }

    std::printf("\n%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
