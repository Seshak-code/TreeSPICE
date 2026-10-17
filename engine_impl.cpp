// ============================================================================
// engine_impl.cpp — Concrete Engine + Session Implementation
// ============================================================================
// Wires the public API (ISimulationEngine + SimulationSession) to the
// internal solver stack:
//
//   SessionImpl::runDC()  →  CircuitSim::solveDC()  →  SolverStep  →  DTO
//   SessionImpl::runAC()  →  CircuitSim::solveAC()  →  SolverStep  →  DTO
//   SessionImpl::runTransient()  →  CircuitSim::stepTransient() x N  →  DTO
//
// The GPU path is selected transparently via CircuitSim::ExecutionMode::GPU
// when GPUContextManager::instance().isAvailable() is true.
// ============================================================================

#include "internal/engine_pch.h"
#include "engine_impl.h"
#include "gpu_context_manager.h"

// Engine internal headers (invisible to consumers via -fvisibility=hidden)
#include "physics/circuitsim.h"
#include "infrastructure/simtrace.h"
#include "netlist/circuit.h"

// engine_api public DTO types
#include "engine_api/simulation_dto.h"
#include "engine_api/isimulation_engine.h"
#include "engine_api/measure.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <random>
#include <atomic>
#include <mutex>
#include <stdexcept>
#include <sstream>
#include <cctype>
#include <map>

// Full SPICE3f5 parser (must be included OUTSIDE any namespace to avoid
// double-qualification: the parser itself already declares namespace acutesim)
#include "netlist/spice_parser.h"

namespace acutesim {

using namespace acutesim::compute::orchestration;

// ── Helper: SolverStep → SimulationResponseDTO ─────────────────────────────
static SimulationResponseDTO stepToResponse(const SolverStep& step,
                                              JobID jobId,
                                              PrecisionMode prec) {
    SimulationResponseDTO dto;
    dto.jobId      = jobId;
    dto.converged  = step.stats.converged;
    dto.iterations = step.stats.iterations;
    dto.residual   = step.stats.residual;
    dto.errorDetail = step.stats.error_detail;

    // Pack node voltages into waveform data point for single-step analyses
    WaveformDataPoint pt;
    pt.timeS = step.time;
    pt.nodeVoltages = step.nodeVoltages;
    pt.branchCurrents = step.branchCurrents;
    dto.waveformPoints.push_back(std::move(pt));

    // Trust provenance
    dto.solverProvenance = step.stats.provenance();
    dto.precisionMode    = prec;
    return dto;
}

static void nameColumns(SimulationResponseDTO& r, const SPICEParseResult& parsed, const CircuitSim& sim) {
    r.nodeNames.assign(parsed.netlist.numGlobalNodes, "");
    for (const auto& [name, idx] : parsed.nodeIndex)
        if (idx >= 1 && idx <= parsed.netlist.numGlobalNodes && r.nodeNames[idx - 1].empty()) {
            r.nodeNames[idx - 1] = name;
            for (auto& c : r.nodeNames[idx - 1]) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    r.branchNames = sim.branchNames(parsed.netlist);
    for (const auto& o : parsed.outputs) r.outputRequests.push_back({o.kind, o.analysis, o.specs});
}

// ── Helper: select CircuitSim ExecutionMode ────────────────────────────────
static CircuitSim::ExecutionMode selectExecMode(PrecisionMode prec) {
    auto& gpu = GPUContextManager::instance();
    if (gpu.isAvailable() && prec != PrecisionMode::Native_F64) {
        return CircuitSim::ExecutionMode::GPU;
    }
    return CircuitSim::ExecutionMode::TENSOR_SOA_SIMD;
}

// What ran, for the response: GPU fronts are the only GPU work (large fronts, FP32 + refinement),
// so a gpu request that factored none reports cpu and why.
static std::string backendOf(const CircuitSim& sim, PrecisionMode prec) {
    if (prec == PrecisionMode::Native_F64) return "cpu";
    if (sim.execMode != CircuitSim::ExecutionMode::GPU) return "cpu (gpu requested, no GPU device available)";
    if (sim.gpuFronts == 0) return "cpu (gpu requested, no front reached the GPU size threshold)";
    return "gpu (" + std::to_string(sim.gpuFronts) + " fronts factored in FP32)";
}

// ── Helper: generate JobID ────────────────────────────────────────────────
static JobID generateJobID() {
    static std::atomic<JobID> nextId{1};
    return nextId.fetch_add(1);
}

// ── Internal Netlist Deserializer ─────────────────────────────────────────
// Delegates to the full SPICE3f5 parser (netlist/spice_parser.h).
// Handles R, C, L, K, V, I, D, M, Q, J, E, G, X; rejects unsupported cards.
// Supports .model, .subckt/.ends, .param, waveform specs (PULSE, SIN), and SI suffixes.
// Returns false (filling `failure`) on a parse error. It used to be logged and the
// partial (or empty) netlist solved anyway, reporting "converged".
static std::function<void(double)> sweepSetter(TensorNetlist& netlist, const std::string& name);

// The request's netlist at circuit temperature tempC (NaN: the netlist's), with the values of this .step run applied.
// Returns an error text, or "".
static std::string parseRequest(const SimulationRequestDTO& req, double tempC, SPICEParseResult& out) {
    std::map<std::string, double> params;
    for (const auto& s : req.stepValues) {
        if (s.kind == 'P') params[s.name] = s.value;
        else if (s.kind == 'T') tempC = s.value;
    }
    out = SPICEParser().parse(req.source.payload, req.source.baseDir, tempC, params);
    if (!out.success) return "SPICE parse error: " + out.errorMessage;
    for (const auto& s : req.stepValues) {
        if (s.kind != 'E') continue;
        const auto set = sweepSetter(out.netlist, s.name);
        if (!set) return ".step element '" + s.name + "' not found at top level (a V, I or R card)";
        set(s.value);
    }
    return "";
}

// The corner temperature wins over the netlist's .temp (as an explicit .dc request wins over its .dc);
// every temperature-dependent value is resolved by the parser. A .temp list is for .op/.dc (listOk) only,
// a .step list for the analyses that run it (stepOk).
static bool deserialiseRequest(const SimulationRequestDTO& req, SPICEParseResult& out,
                               SimulationResponseDTO& failure, bool listOk = false, bool stepOk = false, bool digitalOk = false) {
    // ponytail: 27 is the DTO's default, so a corner asking for 27 C cannot override a netlist .temp
    if (const std::string e = parseRequest(req, req.corner.temperature != 27.0 ? req.corner.temperature : NAN, out); !e.empty()) {
        failure.errorDetail = e;
        return false;
    }
    if (!listOk && out.netlist.environment.temp_count > 1) {
        failure.errorDetail = TEMP_LIST_ERR;
        return false;
    }
    if (!stepOk && !out.steps.empty() && req.stepValues.empty()) {
        failure.errorDetail = ".step is not supported with this analysis yet";
        return false;
    }
    if (!digitalOk && out.netlist.digital) {
        failure.errorDetail = "XSPICE A devices are supported in .tran only so far";
        return false;
    }
    return true;
}

// A .step list: the analysis once per combination (the first card is the outermost loop). The runs join into one response
// whose points carry the step values after any outer sweep's (fastest card first), so splitByOuter gives one block per step.
static SimulationResponseDTO runSteps(const SimulationRequestDTO& req, const std::vector<SPICEParseResult::Step>& steps,
                                      const std::function<SimulationResponseDTO(const SimulationRequestDTO&)>& run) {
    SimulationResponseDTO all;
    std::vector<size_t> idx(steps.size(), 0);
    bool first = true;
    for (bool more = true; more;) {
        SimulationRequestDTO q = req;
        std::vector<double> tag;
        std::string at;
        for (size_t k = 0; k < steps.size(); ++k) {
            q.stepValues.push_back({steps[k].kind, steps[k].name, steps[k].values[idx[k]]});
            at += (k ? " " : "") + steps[k].name + "=" + std::to_string(steps[k].values[idx[k]]);
        }
        for (size_t k = steps.size(); k-- > 0;) tag.push_back(steps[k].values[idx[k]]);
        SimulationResponseDTO r = run(q);
        if (!r.converged) {
            r.errorDetail = "at .step " + at + ": " + r.errorDetail;
            return r;
        }
        for (auto& pt : r.waveformPoints) pt.outer.insert(pt.outer.end(), tag.begin(), tag.end());
        if (first) {
            first = false;
            all = std::move(r);
            for (size_t k = steps.size(); k-- > 0;) all.outerNames.push_back(steps[k].name);
            all.stepped = true;
        } else {
            all.waveformPoints.insert(all.waveformPoints.end(), std::make_move_iterator(r.waveformPoints.begin()),
                                      std::make_move_iterator(r.waveformPoints.end()));
        }
        more = false;
        for (size_t k = steps.size(); k-- > 0 && !more;) {
            if (++idx[k] < steps[k].values.size()) more = true;
            else idx[k] = 0;
        }
    }
    return all;
}

// ============================================================================
// SessionImpl
// ============================================================================

SessionImpl::SessionImpl(GPUContextManager& gpu, PrecisionMode prec)
    : gpu_(gpu)
    , precision_(prec)
    , jobId_(generateJobID())
{
    status_.store(JobStatus::Queued);
}

SessionImpl::~SessionImpl() = default;

void SessionImpl::cancel() {
    status_.store(JobStatus::Cancelled);
}

void SessionImpl::setIncrementalHints(uint64_t topologyHash, uint64_t parameterHash) {
    topologyHash_ = topologyHash;
    paramHash_    = parameterHash;
}

// ── DC Analysis ───────────────────────────────────────────────────────────
SimulationResponseDTO SessionImpl::runDC(const SimulationRequestDTO& req,
                                          const EngineCallbacks& cb) {
    if (status_.load() == JobStatus::Cancelled) {
        SimulationResponseDTO r;
        r.jobId     = jobId_;
        r.converged = false;
        r.errorDetail = "Session cancelled before start.";
        return r;
    }
    status_.store(JobStatus::Running);
    auto start = std::chrono::steady_clock::now();

    SimulationResponseDTO result = dispatch_dc_internal(req, cb);
    if (result.converged) attachMeasures(result);

    auto elapsed = std::chrono::steady_clock::now() - start;
    result.solveTimeMs = std::chrono::duration<double, std::milli>(elapsed).count();
    result.jobId = jobId_;

    result.success = result.converged;  // driver-path consumers read success
    status_.store(result.converged ? JobStatus::Completed : JobStatus::Failed);
    return result;
}

// The setter of a swept element's value at the top level (V/I source DC value, resistor nominal
// resistance), or empty. A source sweeps its DC value only.
static std::function<void(double)> sweepSetter(TensorNetlist& netlist, const std::string& name) {
    auto up = [](std::string x) { for (auto& ch : x) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch))); return x; };
    for (auto& v : netlist.globalBlock.voltageSources)
        if (v.name == name) { v.type.clear(); return [&v](double x) { v.voltage_V = x; }; }
    for (auto& i : netlist.globalBlock.currentSources)
        if (i.name == name) { i.type.clear(); return [&i](double x) { i.current_A = x; }; }
    for (auto& r : netlist.globalBlock.resistors)
        if (up(r.name) == name) return [&r](double x) { r.resistance_ohms = x * r.tcFactor; };
    return {};
}

// Monte Carlo sample: each variation scales the named top-level element's value. Gaussian is
// 1 + sigma*z (z ~ N(0,1)), Uniform is 1 + tolerance*u (u in [-1,1)), as ngspice's gauss()/unif().
// The draw depends only on (seed, element name), so a sample reproduces and elements are independent.
// Returns an error text for an unknown element or distribution.
// ponytail: top-level R/C/L/V/I only (not inside .subckt instances); propertyName is not consulted.
static std::string applyVariations(TensorNetlist& nl, const std::vector<MonteCarloVariationDTO>& vars, uint32_t seed) {
    auto& g = nl.globalBlock;
    for (const auto& v : vars) {
        std::string name = v.componentName;
        for (auto& ch : name) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
        std::mt19937_64 rng(CircuitSim::fnv1a(name, seed));
        auto u01 = [&] { return (double)(rng() >> 11) * (1.0 / 9007199254740992.0); };
        double f;
        if (v.distributionType == "Gaussian")
            f = 1.0 + v.sigma * std::sqrt(-2.0 * std::log(1.0 - u01())) * std::cos(2.0 * M_PI * u01());
        else if (v.distributionType == "Uniform") f = 1.0 + v.tolerance * (2.0 * u01() - 1.0);
        else return "Monte Carlo: unknown distribution '" + v.distributionType + "' (Gaussian or Uniform)";
        double* val = nullptr;
        for (auto& e : g.resistors)       if (e.name == name) val = &e.resistance_ohms;
        for (auto& e : g.capacitors)      if (e.name == name) val = &e.capacitance_farads;
        for (auto& e : g.inductors)       if (e.name == name) val = &e.inductance_henries;
        for (auto& e : g.voltageSources)  if (e.name == name) val = &e.voltage_V;
        for (auto& e : g.currentSources)  if (e.name == name) val = &e.current_A;
        if (!val) return "Monte Carlo: element '" + v.componentName + "' not found at top level";
        *val *= f;
    }
    return "";
}

// A DC run: every combination of the dimensions' values, the first dimension fastest (ngspice: the first
// .dc source is the innermost). One solve per point on a CircuitSim that keeps its last solution, so a point
// is seeded by the previous one and a failed step is approached in halves (continuation) down to 1/256
// step. TEMP re-parses the netlist at that temperature: every temperature-dependent value is resolved by
// the parser, and a sweep or .temp list pays one parse per temperature change.
// Each point is a waveform point whose timeS holds the first dimension's value and outer the others'.
struct DcDim { std::string name; std::vector<double> values; };

static SimulationResponseDTO dcRun(const SimulationRequestDTO& req, SPICEParseResult&& first, const std::vector<DcDim>& dims,
                                   PrecisionMode prec, const EngineCallbacks& cb, const std::function<bool()>& cancelled) {
    struct Run {  // the sim keeps references into the netlist: neither moves once built
        SPICEParseResult parsed;
        CircuitSim sim;
        std::vector<std::function<void(double)>> set;  // per dimension; empty for TEMP
        Run(SPICEParseResult&& p, PrecisionMode m) : parsed(std::move(p)) { sim.execMode = selectExecMode(m); }
    };
    SimulationResponseDTO r;
    r.analysisType = AnalysisTypeDTO::DC;
    r.scaleName = dims[0].name == "TEMP" ? "temp-sweep" : dims[0].name[0] == 'I' ? "i-sweep" : dims[0].name[0] == 'R' ? "res-sweep" : "v-sweep";
    long total = 1;
    for (size_t d = 0; d < dims.size(); ++d) {
        total *= static_cast<long>(dims[d].values.size());
        if (d) r.outerNames.push_back(dims[d].name);
    }
    double runTempC = first.netlist.environment.ambient_temp_K - 273.15;
    std::unique_ptr<Run> run;
    auto build = [&](SPICEParseResult&& p) {
        const int gpuSoFar = run ? run->sim.gpuFronts : 0;  // a TEMP re-parse keeps the count
        run = std::make_unique<Run>(std::move(p), prec);
        run->sim.gpuFronts = gpuSoFar;
        if (const std::string e = applyVariations(run->parsed.netlist, req.mc.variations, req.mcSeed); !e.empty()) {
            r.errorDetail = e;
            return false;
        }
        run->sim.onConvergenceStep = [&cb](int it, double res) { cb.onConvergenceStep(it, res); };
        for (const auto& d : dims) {
            run->set.push_back(d.name == "TEMP" ? std::function<void(double)>() : sweepSetter(run->parsed.netlist, d.name));
            if (d.name != "TEMP" && !run->set.back()) {
                r.errorDetail = ".dc source '" + d.name + "' not found at top level";
                return false;
            }
        }
        return true;
    };
    r.converged = build(std::move(first));
    std::vector<size_t> idx(dims.size(), 0), prev;
    for (long k = 0; r.converged && k < total && !cancelled(); ++k) {
        std::vector<double> val(dims.size());
        double tempC = runTempC;
        for (size_t d = 0; d < dims.size(); ++d) {
            val[d] = dims[d].values[idx[d]];
            if (dims[d].name == "TEMP") tempC = val[d];
        }
        if (tempC != runTempC) {
            SPICEParseResult np;
            if (const std::string e = parseRequest(req, tempC, np); !e.empty()) { r.converged = false; r.errorDetail = e; break; }
            if (!build(std::move(np))) { r.converged = false; break; }
            runTempC = tempC;
            prev.clear();  // a new circuit: no continuation
        }
        auto setAll = [&](size_t d0) { for (size_t d = d0; d < dims.size(); ++d) if (run->set[d]) run->set[d](val[d]); };
        setAll(0);
        const bool stepped = !prev.empty() && std::equal(idx.begin() + 1, idx.end(), prev.begin() + 1) && run->set[0];
        std::function<SolverStep(double, double, int)> reach = [&](double from, double to, int depth) {
            run->set[0](to);
            SolverStep s = run->sim.solveDC(run->parsed.netlist);
            if (s.stats.converged || depth == 0) return s;
            const double mid = 0.5 * (from + to);
            SolverStep m = reach(from, mid, depth - 1);
            return m.stats.converged ? reach(mid, to, depth - 1) : m;
        };
        SolverStep s = stepped ? reach(dims[0].values[prev[0]], val[0], 8) : run->sim.solveDC(run->parsed.netlist);
        setAll(0);
        if (!s.stats.converged) {
            r.converged = false;
            r.errorDetail = "DC sweep failed at " + dims[0].name + "=" + std::to_string(val[0]) + ": " + s.stats.error_detail;
            break;
        }
        WaveformDataPoint pt{val[0], s.nodeVoltages, {}, s.branchCurrents, {}, {}};
        pt.outer.assign(val.begin() + 1, val.end());
        r.waveformPoints.push_back(std::move(pt));
        cb.onProgress(static_cast<int>(k + 1), static_cast<int>(total));
        prev = idx;
        for (size_t d = 0; d < dims.size() && ++idx[d] == dims[d].values.size(); ++d) idx[d] = 0;
    }
    if (run) {
        r.backend = backendOf(run->sim, prec);
        nameColumns(r, run->parsed, run->sim);
    }
    return r;
}

SimulationResponseDTO SessionImpl::dispatch_dc_internal(const SimulationRequestDTO& req,
                                                          const EngineCallbacks& cb) {
    SPICEParseResult parsed;
    SimulationResponseDTO failure;
    if (!deserialiseRequest(req, parsed, failure, true, true)) return failure;
    if (!parsed.steps.empty() && req.stepValues.empty())
        return runSteps(req, parsed.steps, [&](const SimulationRequestDTO& q) { return dispatch_dc_internal(q, cb); });
    std::vector<DCConfigDTO> sweeps;  // innermost first
    if (!req.dc.sweepVariable.empty()) {  // an explicit request wins over the netlist's .dc
        sweeps.push_back(req.dc);
        sweeps.insert(sweeps.end(), req.dc.outer.begin(), req.dc.outer.end());
    } else {
        for (const auto& d : parsed.dcSweeps) sweeps.push_back({d.name, d.start, d.stop, d.step, {}});
    }
    std::vector<DcDim> dims;
    for (auto& sw : sweeps) {
        for (auto& ch : sw.sweepVariable) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
        const double span = sw.step != 0.0 ? (sw.stop - sw.start) / sw.step : -1.0;
        if (!(span >= -1e-9)) {
            failure.errorDetail = ".dc step must be non-zero and point from start to stop";
            return failure;
        }
        DcDim d{sw.sweepVariable, {}};
        // accumulated as ngspice steps it (value += step): a comparison at a sweep point (V(a) > 0.5 at 0.1 * 5) then
        // falls on the same side of its threshold on every platform as in ngspice
        double v = sw.start;
        for (long k = 0, n = static_cast<long>(std::floor(span + 1e-9)); k <= n; ++k, v += sw.step) d.values.push_back(v);
        dims.push_back(std::move(d));
    }
    if (parsed.tempsC.size() > 1) {  // .temp t1 t2 ...: the whole run once per temperature, outermost
        for (const auto& d : dims)
            if (d.name == "TEMP") { failure.errorDetail = "a .temp list and a .dc TEMP sweep together are not supported yet"; return failure; }
        for (const auto& s : req.stepValues)
            if (s.kind == 'T') { failure.errorDetail = "a .temp list and .step temp together are not supported yet"; return failure; }
        dims.push_back({"TEMP", parsed.tempsC});
    }
    for (const auto& s : req.stepValues)
        for (const auto& d : dims)
            if (s.kind == 'T' && d.name == "TEMP") { failure.errorDetail = "a .dc TEMP sweep and .step temp together are not supported yet"; return failure; }
    if (dims.empty()) {
        CircuitSim sim;
        sim.execMode = selectExecMode(precision_);
        sim.onConvergenceStep = [&](int it, double res) { cb.onConvergenceStep(it, res); };
        if (const std::string e = applyVariations(parsed.netlist, req.mc.variations, req.mcSeed); !e.empty()) {
            failure.errorDetail = e;
            return failure;
        }
        SimulationResponseDTO r = stepToResponse(sim.solveDC(parsed.netlist), jobId_, precision_);
        r.backend = backendOf(sim, precision_);
        nameColumns(r, parsed, sim);
        return r;
    }
    return dcRun(req, std::move(parsed), dims, precision_, cb, [this] { return status_.load() == JobStatus::Cancelled; });
}


// ── AC Analysis ───────────────────────────────────────────────────────────
SimulationResponseDTO SessionImpl::runAC(const SimulationRequestDTO& req,
                                          const EngineCallbacks& cb) {
    status_.store(JobStatus::Running);
    SimulationResponseDTO result = dispatch_ac_internal(req, cb);
    if (result.converged) attachMeasures(result);
    result.jobId = jobId_;
    result.success = result.converged;  // driver-path consumers read success
    status_.store(result.converged ? JobStatus::Completed : JobStatus::Failed);
    return result;
}

SimulationResponseDTO SessionImpl::dispatch_ac_internal(const SimulationRequestDTO& req,
                                                          const EngineCallbacks& cb) {
    SPICEParseResult parsed;
    SimulationResponseDTO r;
    if (!deserialiseRequest(req, parsed, r, false, true)) return r;
    if (!parsed.steps.empty() && req.stepValues.empty())
        return runSteps(req, parsed.steps, [&](const SimulationRequestDTO& q) { return dispatch_ac_internal(q, cb); });
    TensorNetlist& netlist = parsed.netlist;
    ACConfigDTO ac = req.ac;  // an explicit request wins over the netlist's .ac
    if (ac.points == 0) ac = {parsed.acSweep, parsed.acPoints, parsed.acStart, parsed.acStop};
    const auto freqs = CircuitSim::acFrequencies(ac.sweep, ac.points, ac.fStart, ac.fStop);
    if (freqs.empty()) {
        r.errorDetail = "AC needs a sweep: .ac dec|oct|lin n fstart fstop (0 < fstart <= fstop).";
        return r;
    }
    CircuitSim sim;
    sim.execMode = selectExecMode(precision_);
    SolverStep dc = sim.solveDC(netlist);
    if (!dc.stats.converged) {
        r.errorDetail = "DC operating point failed before AC analysis: " + dc.stats.error_detail;
        return r;
    }
    const auto X = sim.solveAC(netlist, freqs);
    r.converged    = X.size() == freqs.size();
    r.analysisType = AnalysisTypeDTO::AC;
    r.scaleName    = "frequency";
    r.backend      = backendOf(sim, precision_);
    for (size_t k = 0; k < X.size(); ++k) {
        WaveformDataPoint pt{freqs[k], {}, {}, {}};
        for (int i = 0; i < netlist.numGlobalNodes; ++i) {
            pt.nodeVoltages.push_back(X[k][i].real());
            pt.nodeVoltagesImag.push_back(X[k][i].imag());
        }
        const int nb = sim.branchCount(netlist);
        for (int i = netlist.numGlobalNodes; i < netlist.numGlobalNodes + nb && i < (int)X[k].size(); ++i) {
            pt.branchCurrents.push_back(X[k][i].real());
            pt.branchCurrentsImag.push_back(X[k][i].imag());
        }
        r.waveformPoints.push_back(std::move(pt));
        cb.onProgress(static_cast<int>(k + 1), static_cast<int>(freqs.size()));
    }
    nameColumns(r, parsed, sim);
    return r;
}

// ── Transient Analysis ────────────────────────────────────────────────────
SimulationResponseDTO SessionImpl::runTransient(const SimulationRequestDTO& req,
                                                  const EngineCallbacks& cb,
                                                  ResultChunkCallback streamCallback) {
    status_.store(JobStatus::Running);
    SimulationResponseDTO result = dispatch_tran_internal(req, cb, std::move(streamCallback));
    if (result.converged) attachMeasures(result);
    result.jobId = jobId_;
    result.success = result.converged;  // driver-path consumers read success
    status_.store(result.converged ? JobStatus::Completed : JobStatus::Failed);
    return result;
}

SimulationResponseDTO SessionImpl::dispatch_tran_internal(const SimulationRequestDTO& req,
                                                            const EngineCallbacks& cb,
                                                            ResultChunkCallback streamCb) {
    SPICEParseResult parsed;
    SimulationResponseDTO failure;
    if (!deserialiseRequest(req, parsed, failure, false, true, true)) return failure;
    if (!parsed.steps.empty() && req.stepValues.empty())
        return runSteps(req, parsed.steps, [&](const SimulationRequestDTO& q) { return dispatch_tran_internal(q, cb, streamCb); });
    TensorNetlist& netlist = parsed.netlist;
    const bool uic = parsed.tranUIC || req.transient.uic;
    const double  stopTime = req.transient.stopTimeS != 0 ? req.transient.stopTimeS : parsed.tranStop;  // else the netlist's .tran
    const double  dt       = req.transient.stepSizeS != 0 ? req.transient.stepSizeS : parsed.tranStep;

    CircuitSim sim;
    sim.execMode = selectExecMode(precision_);

    SimulationResponseDTO result;
    uint64_t stepIdx = 0;
    const std::string err = sim.runTransient(
        netlist, dt, stopTime, uic, [&](const SolverStep& step) {
            if (status_.load() == JobStatus::Cancelled) return false;
            WaveformDataPoint pt;
            pt.timeS        = step.time;
            pt.nodeVoltages = step.nodeVoltages;
            pt.branchCurrents = step.branchCurrents;
            result.waveformPoints.push_back(pt);
            if (streamCb) streamCb(pt, stepIdx);
            cb.onTimeStep(step.time, stopTime);
            ++stepIdx;
            return true;
        });
    result.converged   = err.empty();
    result.errorDetail = err;

    result.iterations = static_cast<int>(stepIdx);
    result.analysisType = AnalysisTypeDTO::TRANSIENT;
    result.scaleName = "time";
    nameColumns(result, parsed, sim);
    result.backend = backendOf(sim, precision_);
    if (const auto* d = sim.digital()) {
        result.digitalNodes = d->net().nodes;
        for (const auto& c : d->log()) result.digitalEvents.push_back({c.t, c.node, xspice::text(c.v)});
    }
    return result;
}

// ── Lockstep transient ────────────────────────────────────────────────────
namespace {
// Owns the parsed netlist and the CircuitSim whose state it advances (the sim keeps
// references into the netlist, so neither moves after construction).
class TransientImpl final : public TransientHandle {
public:
    TransientImpl(const SimulationRequestDTO& req, PrecisionMode prec) {
        SimulationResponseDTO failure;
        if (!deserialiseRequest(req, parsed_, failure, false, false, true)) { err_ = failure.errorDetail; return; }
        sim_.execMode = selectExecMode(prec);
        const double step = req.transient.stepSizeS > 0 ? req.transient.stepSizeS : parsed_.tranStep;
        err_ = sim_.beginTransient(parsed_.netlist, step, parsed_.tranUIC || req.transient.uic);
    }
    bool setSource(const std::string& name, double volts, double rampS) override {
        if (!err_.empty()) return false;
        if (sim_.setSource(parsed_.netlist, name, volts, rampS)) return true;
        err_ = "unknown source '" + name + "'";
        return false;
    }
    bool advanceTo(double t) override {
        if (!err_.empty()) return false;
        err_ = sim_.advanceTo(parsed_.netlist, t);
        return err_.empty();
    }
    double time() const override { return sim_.tranTime(); }
    bool voltage(const std::string& node, double& v) const override {
        std::string key = node;
        for (auto& c : key) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        if (key == "0" || key == "GND") { v = 0.0; return err_.empty(); }
        const auto it = parsed_.nodeIndex.find(key);
        if (it == parsed_.nodeIndex.end() || !err_.empty()) return false;
        v = sim_.nodeVoltage(it->second);
        return true;
    }
    std::string error() const override { return err_; }
private:
    SPICEParseResult parsed_;
    CircuitSim sim_;
    std::string err_;
};
}  // namespace

std::unique_ptr<TransientHandle> SessionImpl::beginTransient(const SimulationRequestDTO& req) {
    return std::make_unique<TransientImpl>(req, precision_);
}

// ── Monte Carlo ───────────────────────────────────────────────────────────
SimulationResponseDTO SessionImpl::runMonteCarlo(const SimulationRequestDTO& req,
                                                   const MonteCarloConfigDTO& mc,
                                                   const EngineCallbacks& cb) {
    status_.store(JobStatus::Running);
    SimulationResponseDTO agg;
    agg.analysisType = AnalysisTypeDTO::MONTE_CARLO;
    agg.converged    = true;
    agg.jobId        = jobId_;

    for (int i = 0; i < mc.sampleCount && status_.load() != JobStatus::Cancelled; ++i) {
        SimulationRequestDTO run_req = req;
        run_req.mc     = mc;
        run_req.mcSeed = mc.seed + i;
        auto r = dispatch_dc_internal(run_req, cb);
        if (!r.converged) { agg.converged = false; agg.errorDetail = r.errorDetail; }
        if (!r.waveformPoints.empty()) {
            agg.waveformPoints.push_back(r.waveformPoints.front());
        }
        cb.onProgress(i + 1, mc.sampleCount);
    }

    agg.success = agg.converged;  // driver-path consumers read success
    status_.store(agg.converged ? JobStatus::Completed : JobStatus::Failed);
    return agg;
}

// ── Noise ─────────────────────────────────────────────────────────────────
SimulationResponseDTO SessionImpl::runNoise(const SimulationRequestDTO& req,
                                             const EngineCallbacks& /*cb*/) {
    status_.store(JobStatus::Running);
    SimulationResponseDTO r;
    r.jobId        = jobId_;
    r.analysisType = AnalysisTypeDTO::NOISE;
    SPICEParseResult parsed;
    if (deserialiseRequest(req, parsed, r)) {
        TensorNetlist& netlist = parsed.netlist;
        const auto freqs = CircuitSim::acFrequencies(parsed.noiseSweep, parsed.noisePoints,
                                                     parsed.noiseStart, parsed.noiseStop, true);
        CircuitSim sim;
        sim.execMode = selectExecMode(precision_);
        if (parsed.noiseSource.empty() || freqs.empty())
            r.errorDetail = "Noise needs .noise v(out[,ref]) src dec|oct|lin n fstart fstop.";
        else if (SolverStep dc = sim.solveDC(netlist); !dc.stats.converged)
            r.errorDetail = "DC operating point failed before noise analysis: " + dc.stats.error_detail;
        else {
            auto nz = sim.solveNoise(netlist, freqs, parsed.noiseOut, parsed.noiseRef, parsed.noiseSource);
            r.errorDetail = nz.error;
            r.converged   = nz.error.empty();
            r.extended.hasNoise = r.converged;
            r.extended.noise = {freqs, std::move(nz.onoise), std::move(nz.inoise), nz.onoiseTotal, nz.inoiseTotal};
        }
        r.backend = backendOf(sim, precision_);
    }
    r.success = r.converged;  // driver-path consumers read success
    status_.store(r.converged ? JobStatus::Completed : JobStatus::Failed);
    return r;
}

// ── Poles and zeros ───────────────────────────────────────────────────────
SimulationResponseDTO SessionImpl::runPoleZero(const SimulationRequestDTO& req,
                                                const EngineCallbacks& /*cb*/) {
    status_.store(JobStatus::Running);
    SimulationResponseDTO r;
    r.jobId        = jobId_;
    r.analysisType = AnalysisTypeDTO::POLE_ZERO;
    SPICEParseResult parsed;
    if (deserialiseRequest(req, parsed, r)) {
        CircuitSim sim;
        sim.execMode = selectExecMode(precision_);
        if (!parsed.hasPZ)
            r.errorDetail = "Pole-zero needs .pz in+ in- out+ out- cur|vol pol|zer|pz.";
        else if (SolverStep dc = sim.solveDC(parsed.netlist); !dc.stats.converged)
            r.errorDetail = "DC operating point failed before pole-zero analysis: " + dc.stats.error_detail;
        else {
            const int* n = parsed.pzNode;
            auto pz = sim.solvePoleZero(parsed.netlist, n[0], n[1], n[2], n[3], parsed.pzVolt, parsed.pzPoles, parsed.pzZeros);
            r.errorDetail = pz.error;
            r.converged   = pz.error.empty();
            r.extended.hasPZ = r.converged;
            for (auto p : pz.poles) { r.extended.pz.poleRe.push_back(p.real()); r.extended.pz.poleIm.push_back(p.imag()); }
            for (auto z : pz.zeros) { r.extended.pz.zeroRe.push_back(z.real()); r.extended.pz.zeroIm.push_back(z.imag()); }
        }
        r.backend = backendOf(sim, precision_);
    }
    r.success = r.converged;
    status_.store(r.converged ? JobStatus::Completed : JobStatus::Failed);
    return r;
}

// ── DC sensitivity ────────────────────────────────────────────────────────
SimulationResponseDTO SessionImpl::runSens(const SimulationRequestDTO& req,
                                            const EngineCallbacks& /*cb*/) {
    status_.store(JobStatus::Running);
    SimulationResponseDTO r;
    r.jobId        = jobId_;
    r.analysisType = AnalysisTypeDTO::SENSITIVITY;
    SPICEParseResult parsed;
    if (deserialiseRequest(req, parsed, r)) {
        CircuitSim sim;
        sim.execMode = selectExecMode(precision_);
        if (!parsed.hasSens)
            r.errorDetail = "Sensitivity needs .sens v(out[,ref]) or .sens i(vsrc).";
        else if (SolverStep dc = sim.solveDC(parsed.netlist); !dc.stats.converged)
            r.errorDetail = "DC operating point failed before sensitivity analysis: " + dc.stats.error_detail;
        else {
            auto sn = sim.solveSens(parsed.netlist, parsed.sensOut, parsed.sensRef, parsed.sensSource);
            r.errorDetail = sn.error;
            r.converged   = sn.error.empty();
            r.extended.hasSens = r.converged;
            r.extended.sens = {std::move(sn.names), std::move(sn.values), std::move(sn.unsupported)};
        }
        r.backend = backendOf(sim, precision_);
    }
    r.success = r.converged;
    status_.store(r.converged ? JobStatus::Completed : JobStatus::Failed);
    return r;
}

// ============================================================================
// EngineImpl
// ============================================================================

EngineImpl::EngineImpl()
    : gpu_(GPUContextManager::instance())
{
    detectCapabilities();
}

EngineImpl::~EngineImpl() = default;

void EngineImpl::detectCapabilities() {
    capabilities_.gpuAvailable         = gpu_.isAvailable();
    capabilities_.emulatedF64Supported = true;
    capabilities_.batchedSolveSupported = true;
    capabilities_.maxParallelSessions  = 8;
    capabilities_.maxBatchSize         = 64;

    if (capabilities_.gpuAvailable) {
        capabilities_.gpuAdapterName = gpu_.adapterInfo().name;
    }

#if defined(__AVX512F__)
    capabilities_.avx512Supported = true;
#endif
#if defined(__ARM_NEON)
    capabilities_.neonSupported = true;
#endif

    capabilities_.engineVersionString = "TreeSPICE " TREESPICE_VERSION;
}

std::unique_ptr<SimulationSession> EngineImpl::createSession() {
    return std::make_unique<SessionImpl>(gpu_, precision_);
}

EngineCapabilities EngineImpl::capabilities() const {
    return capabilities_;
}

void EngineImpl::setPrecisionMode(PrecisionMode mode) {
    precision_ = mode;
}

PrecisionMode EngineImpl::precisionMode() const {
    return precision_;
}

// ── Factory (defined here, declared ENGINE_API in isimulation_engine.h) ────
std::unique_ptr<ISimulationEngine> ISimulationEngine::create() {
    return std::make_unique<EngineImpl>();
}

} // namespace acutesim
