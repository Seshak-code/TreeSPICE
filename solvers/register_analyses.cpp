// ============================================================================
// compute/solvers/register_analyses.cpp — Explicit registration entrypoint
// ============================================================================
// Call registerAllAnalyses() to populate AnalysisRegistry with fully
// functional TF / FOURIER handlers that go through the solver.
//
// Each handler:
//   1. Applies corner parameters to the netlist.
//   2. Runs a DC operating point for linearisation.
//   3. Delegates to the dedicated solver function.
//
// Phase 4.5 / Reconcile — stubs replaced with real implementations.
// ============================================================================

#include "solvers/register_analyses.h"
#include "solvers/analysis_registry.h"
#include "solvers/tf_analysis.h"
#include "solvers/fourier_analysis.h"

// Full physics headers for handler implementations
#include "physics/circuitsim.h"
#include "physics/corner_config.h"
#include "netlist/circuit.h"
#include "netlist/spice_parser.h"

namespace acutesim {
namespace compute {
namespace solvers {

using namespace orchestration;

// The netlist a corner runs on: the caller's, or, when the corner asks for another temperature than
// it was parsed at, the request's source parsed at the corner's (the parser resolves every
// temperature-dependent value). A .temp list is for .op/.dc only.
// ponytail: 27 is the DTO's default, so a corner asking for 27 C cannot override a netlist .temp.
static bool cornerNetlist(const SimulationRequestDTO& req, const TensorNetlist& netlist, TensorNetlist& out,
                          SimulationResponseDTO& errOut) {
    out = netlist;
    errOut.success = false;
    if (netlist.environment.temp_count > 1) { errOut.errorDetail = TEMP_LIST_ERR; return false; }
    if (req.corner.temperature == 27.0 || req.corner.temperature + 273.15 == netlist.environment.ambient_temp_K) {
        errOut.success = true;
        return true;
    }
    SPICEParseResult p = SPICEParser().parse(req.source.payload, req.source.baseDir, req.corner.temperature);
    if (!p.success) { errOut.errorDetail = "SPICE parse error: " + p.errorMessage; return false; }
    out = std::move(p.netlist);
    errOut.success = true;
    return true;
}

// ── Helper: apply corner and run DC ─────────────────────────────────────────
static bool applyCornerAndDC(
    CircuitSim& sim,
    TensorNetlist& runNetlist,
    const SimulationRequestDTO& req,
    SolverStep& dcOut,
    SimulationResponseDTO& errOut)
{
    acutesim::physics::CornerConfig corner;
    corner.name           = req.corner.processCorner;
    corner.temperatureC   = req.corner.temperature;
    corner.voltageScale   = req.corner.supplyVoltage / 1.8;
    sim.applyCorner(runNetlist, corner);

    dcOut = sim.solveDC(runNetlist);
    if (!dcOut.stats.converged) {
        errOut.success      = false;
        errOut.errorDetail = "DC operating point failed: " + dcOut.stats.error_detail;
        return false;
    }
    return true;
}

// ── Transfer Function ────────────────────────────────────────────────────────
static SimulationResponseDTO tfHandler(
    const SimulationRequestDTO& req,
    const TensorNetlist& netlist,
    CircuitSim& sim)
{
    TensorNetlist runNetlist;  // handler owns this
    SimulationResponseDTO errOut;
    if (!cornerNetlist(req, netlist, runNetlist, errOut)) return errOut;
    SolverStep dc;
    if (!applyCornerAndDC(sim, runNetlist, req, dc, errOut)) return errOut;
    errOut.topologyHash = req.topologyHash;
    return runTFAnalysis(sim, runNetlist, req);
}

// ── Fourier ──────────────────────────────────────────────────────────────────
static SimulationResponseDTO fourierHandler(
    const SimulationRequestDTO& req,
    const TensorNetlist& netlist,
    CircuitSim& sim)
{
    // Fourier requires a transient waveform — run transient first.
    TensorNetlist runNetlist;
    if (SimulationResponseDTO bad; !cornerNetlist(req, netlist, runNetlist, bad)) return bad;
    acutesim::physics::CornerConfig corner;
    corner.name           = req.corner.processCorner;
    corner.temperatureC   = req.corner.temperature;
    corner.voltageScale   = req.corner.supplyVoltage / 1.8;
    sim.applyCorner(runNetlist, corner);

    // Extract stopTime from parameters (default 1ms)
    double stopTime = 1e-3;
    {
        auto it = req.parameters.find("stopTime");
        if (it != req.parameters.end()) {
            try { stopTime = std::stod(it->second); } catch (...) {}
        }
    }
    int probeNode = 1;
    {
        auto it = req.parameters.find("probeNode");
        if (it != req.parameters.end()) {
            try { probeNode = std::stoi(it->second); } catch (...) {}
        }
        probeNode = std::max(1, std::min(probeNode, runNetlist.numGlobalNodes));
    }

    std::vector<double> timeVec;
    std::vector<double> waveform;
    const std::string err = sim.runTransient(runNetlist, 1e-5, stopTime, req.transient.uic,
        [&](const SolverStep& step) {
            timeVec.push_back(step.time);
            waveform.push_back(probeNode - 1 < (int)step.nodeVoltages.size()
                               ? step.nodeVoltages[probeNode - 1] : 0.0);
            return true;
        });
    if (!err.empty()) {
        SimulationResponseDTO errOut;
        errOut.success     = false;
        errOut.errorDetail = err;
        return errOut;
    }

    return runFourierAnalysis(runNetlist, timeVec, waveform, req);
}

// ── Public entry point ───────────────────────────────────────────────────────
void registerAllAnalyses() {
    auto& reg = AnalysisRegistry::instance();
    // Idempotent — overwrites any previous stub; safe to call multiple times.
    reg.registerHandler(AnalysisTypeDTO::TRANSFER_FUNCTION, tfHandler);
    reg.registerHandler(AnalysisTypeDTO::FOURIER,           fourierHandler);
}

} // namespace solvers
} // namespace compute
} // namespace acutesim
