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
#include <set>
#include <regex>

// Full SPICE3f5 parser (must be included OUTSIDE any namespace to avoid
// double-qualification: the parser itself already declares namespace acutesim)
#include "netlist/spice_parser.h"
#include "netlist/hier_partition.h"

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

// .save (ngspice): given, and not "all", the response keeps only the vectors .save, .print and .meas name (their
// v(...), i(...) and .save's bare node names), so a long transient of a large circuit stores those columns, not
// every node at every step. Indices into the solution's nodes and branches, ascending; all: no .save.
// hierarchical (S3): one column per name, top-level and named by an output (its node may be another name's); the
// names, lower case, then name the columns.
struct KeptColumns { bool all = true; std::vector<int> nodes, branches; std::vector<std::string> nodeNames, branchNames; };
static KeptColumns keptColumns(const SPICEParseResult& parsed, const CircuitSim& sim) {
    KeptColumns k;
    auto lower = [](std::string x) { for (auto& c : x) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); return x; };
    auto upper = [](std::string x) { for (auto& c : x) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c))); return x; };
    bool save = false, all = false;
    for (const auto& o : parsed.outputs)
        if (o.kind == 's') {
            save = true;
            for (const auto& sp : o.specs) all = all || lower(sp) == "all";
        }
    // the names the outputs read (upper-case nodes, lower-case sources)
    std::set<std::string> nodes, brs;
    static const std::regex ref(R"(\b([vi])[a-z]*\(([^()]*)\))", std::regex::icase), bare(R"([A-Za-z0-9_.#:$\[\]]+)");
    for (const auto& o : parsed.outputs)
        for (const auto& spec : o.specs) {
            if (o.kind == 's' && std::regex_match(spec, bare)) { if (lower(spec) != "all") nodes.insert(upper(spec)); continue; }
            for (auto it = std::sregex_iterator(spec.begin(), spec.end(), ref); it != std::sregex_iterator(); ++it) {
                std::string args = (*it)[2];
                args.erase(std::remove_if(args.begin(), args.end(), [](unsigned char c) { return std::isspace(c); }), args.end());
                if (std::tolower(static_cast<unsigned char>((*it)[1].str()[0])) == 'i') { brs.insert(lower(args)); continue; }
                const size_t comma = args.find(',');
                nodes.insert(upper(args.substr(0, comma)));
                if (comma != std::string::npos) nodes.insert(upper(args.substr(comma + 1)));
            }
        }
    if (parsed.hierarchical) {
        k.all = false;
        for (const auto& [name, idx] : parsed.nodeIndex)
            if (idx >= 1 && (!save || all || nodes.count(name))) { k.nodes.push_back(idx - 1); k.nodeNames.push_back(lower(name)); }
        for (const auto& [name, idx] : parsed.sourceIndex)
            if (!save || all || brs.count(lower(name))) { k.branches.push_back(idx); k.branchNames.push_back(lower(name)); }
        return k;
    }
    if (!save || all) return k;
    k.all = false;
    const auto branches = sim.branchNames(parsed.netlist);
    std::set<int> ni, bi;
    for (const auto& n : nodes)
        if (const auto it = parsed.nodeIndex.find(n); it != parsed.nodeIndex.end() && it->second >= 1) ni.insert(it->second - 1);
    for (const auto& b : brs)
        if (const auto it = std::find(branches.begin(), branches.end(), b); it != branches.end()) bi.insert((int)(it - branches.begin()));
    k.nodes.assign(ni.begin(), ni.end());
    k.branches.assign(bi.begin(), bi.end());
    return k;
}

// The kept columns of a point that still has every node (fullNodes) and branch (fullBranches)
static void keepColumns(const KeptColumns& k, WaveformDataPoint& p, size_t fullNodes, size_t fullBranches) {
    if (k.all) return;
    auto pick = [](std::vector<double>& v, const std::vector<int>& idx, size_t full) {
        if (v.size() != full) return;
        std::vector<double> out;
        out.reserve(idx.size());
        for (int i : idx) out.push_back(i < (int)v.size() ? v[i] : 0.0);
        v = std::move(out);
    };
    pick(p.nodeVoltages, k.nodes, fullNodes);
    pick(p.nodeVoltagesImag, k.nodes, fullNodes);
    pick(p.branchCurrents, k.branches, fullBranches);
    pick(p.branchCurrentsImag, k.branches, fullBranches);
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
    const KeptColumns k = keptColumns(parsed, sim);  // .save: the points (those stored whole) and names narrow
    if (k.all) return;
    for (auto& p : r.waveformPoints) keepColumns(k, p, r.nodeNames.size(), r.branchNames.size());
    std::vector<std::string> nn, bn;
    for (size_t c = 0; c < k.nodes.size(); ++c) nn.push_back(k.nodeNames.empty() ? r.nodeNames[k.nodes[c]] : k.nodeNames[c]);
    for (size_t c = 0; c < k.branches.size(); ++c) bn.push_back(k.branchNames.empty() ? r.branchNames[k.branches[c]] : k.branchNames[c]);
    r.nodeNames = std::move(nn);
    r.branchNames = std::move(bn);
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

// S3 (.options hier): the parse's quotient circuit in its place (hier_partition.h), the names mapped to its nodes:
// the top level's and the hierarchical ones (x1.x2.n) the outputs name. Nets two or more of one context's own share
// (symmetry) are reported: a bistable cell solved that way sits at its metastable point.
static std::string toQuotient(SPICEParseResult& out) {
    hierpart::Partition part;
    hierpart::Quotient q;
    try { q = hierpart::quotient(out.netlist, &part); }
    catch (const std::exception& e) { return std::string("hierarchical solve (.options hier): ") + e.what(); }
    // A bistable pair symmetry would hold at its metastable point: unseeded, undriven nets of one instance in one class
    // that one element or child instance couples through terminals of different roles (two identical inverters'
    // q / qb, in at one and out at the other); nets a row meets at interchangeable pins (bit lines) are fine
    for (const auto& c : part.ctx) {
        if (c.def.empty()) continue;
        const auto& blk = out.netlist.blockDefinitions.at(c.def);
        std::set<int> fixed;
        for (const auto& v : blk.voltageSources) { fixed.insert(v.nodePositive); fixed.insert(v.nodeNegative); }
        if (const auto sd = out.netlist.hier.seeds.find(c.def); sd != out.netlist.hier.seeds.end())
            for (const auto& sv : sd->second) fixed.insert(sv.node);
        for (const auto& sv : c.own) fixed.insert(sv.node);
        auto free = [&](int n) { return n > (int)blk.pinNames.size() && n < (int)c.net.size() && !fixed.count(n); };
        std::set<int> coupled;  // classes two of whose nets one element or child touches in different roles
        auto touch = [&](const std::vector<int>& t, const std::vector<int>& role) {
            std::map<int, std::pair<int, int>> seen;  // color -> its first (net, role) here
            for (size_t k = 0; k < t.size(); ++k)
                if (free(t[k]))
                    if (const auto [it, isNew] = seen.emplace(c.net[t[k]], std::make_pair(t[k], role[k]));
                        !isNew && it->second.first != t[k] && it->second.second != role[k])
                        coupled.insert(c.net[t[k]]);
        };
        hierpart::forEachElement(blk, [&](char, size_t, const std::vector<const int*>& t) {
            std::vector<int> v, role;  // an element's role: its terminal
            for (const int* n : t) { role.push_back((int)v.size()); v.push_back(*n); }
            touch(v, role);
        });
        if (const auto kids = out.netlist.hier.children.find(c.def); kids != out.netlist.hier.children.end())
            for (size_t x = 0; x < kids->second.size(); ++x) {  // a child's role: its pin's refined color there
                const auto& k = part.ctx[c.child[x]];
                std::vector<int> role;
                for (size_t p = 0; p < kids->second[x].ports.size(); ++p) role.push_back(k.net[p + 1]);
                touch(kids->second[x].ports, role);
            }
        std::map<int, std::vector<std::string>> same;
        for (const auto& [name, local] : out.netlist.hier.nodeNames[c.def])
            if (free(local) && coupled.count(c.net[local])) same[c.net[local]].push_back(name);
        for (const auto& [col, names] : same)
            if (names.size() > 1) {
                std::cerr << "Warning: .options hier: nets";
                for (const auto& n : names) std::cerr << " " << n;
                std::cerr << " of " << c.def << " are equal by symmetry (if the cell is bistable, it sits at its metastable point: "
                             "give it an .ic or .nodeset)\n";
            }
    }
    std::map<std::string, int> idx;
    for (const auto& [name, n] : out.nodeIndex)
        if (const int qn = hierpart::nodeFor(out.netlist, out.nodeIndex, part, q, name); qn > 0) idx[name] = qn;
    static const std::regex ref(R"(\b[vi][a-z]*\(([^()]*)\))", std::regex::icase), bare(R"([A-Za-z0-9_.#:$\[\]]+)");
    auto named = [&](std::string n) {  // a hierarchical name an output reads
        for (auto& ch : n) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
        if (n.find('.') == std::string::npos || idx.count(n)) return;
        if (const int qn = hierpart::nodeFor(out.netlist, out.nodeIndex, part, q, n); qn > 0) idx[n] = qn;
    };
    for (const auto& o : out.outputs)
        for (const auto& spec : o.specs) {
            if (std::regex_match(spec, bare)) named(spec);
            for (auto it = std::sregex_iterator(spec.begin(), spec.end(), ref); it != std::sregex_iterator(); ++it) {
                std::string args = (*it)[1];
                args.erase(std::remove_if(args.begin(), args.end(), [](unsigned char ch) { return std::isspace(ch); }), args.end());
                const size_t comma = args.find(',');
                named(args.substr(0, comma));
                if (comma != std::string::npos) named(args.substr(comma + 1));
            }
        }
    auto top = [&](int& n) {  // a top-level node an analysis names -> its class's node (ground: 0)
        const auto it = n > 0 && n < (int)part.ctx[0].net.size() ? q.node.find(part.ctx[0].net[n]) : q.node.end();
        n = it == q.node.end() ? 0 : it->second;
    };
    top(out.noiseOut); top(out.noiseRef); top(out.tfOut); top(out.tfRef);
    out.netlist = std::move(q.nl);
    out.nodeIndex = std::move(idx);
    out.sourceIndex = std::move(q.vsource);
    out.elementCount = std::move(q.topCount);
    out.hierarchical = true;
    return "";
}

// S3: an element the quotient circuit's class shares with identical ones cannot be swept or varied alone there
static std::string hierShared(const SPICEParseResult& p, const std::string& name) {
    if (!p.hierarchical) return "";
    const auto it = p.elementCount.find(name);
    if (it != p.elementCount.end() && it->second == 1.0) return "";
    return name + " stands for identical elements on the hierarchical path (.options hier): it cannot be swept or varied alone";
}

// S3: a node an analysis measures alone (noise, .tf output) must be its class's only net on the hierarchical path
static std::string hierNet(const SPICEParseResult& p, int node, const std::string& what) {
    if (!p.hierarchical || node <= 0 || node > (int)p.netlist.classSize.size() || p.netlist.classSize[node - 1] == 1.0) return "";
    return what + " stands for " + std::to_string((long long)p.netlist.classSize[node - 1]) +
           " identical nets on the hierarchical path (.options hier): one of them alone needs the flat netlist";
}

// The request's netlist at circuit temperature tempC (NaN: the netlist's), with the values of this .step run applied.
// Returns an error text, or "".
static std::string parseRequest(const SimulationRequestDTO& req, double tempC, SPICEParseResult& out,
                                const SensParam* sens = nullptr) {
    std::map<std::string, double> params;
    for (const auto& s : req.stepValues) {
        if (s.kind == 'P') params[s.name] = s.value;
        else if (s.kind == 'T') tempC = s.value;
    }
    out = SPICEParser().parse(req.source.payload, req.source.baseDir, tempC, params, sens);
    if (!out.success) return "SPICE parse error: " + out.errorMessage;
    if (!out.netlist.hier.top.empty())
        if (std::string e = toQuotient(out); !e.empty()) return e;
    for (const auto& s : req.stepValues) {
        if (s.kind != 'E') continue;
        if (const std::string e = hierShared(out, s.name); !e.empty()) return e;
        const auto set = sweepSetter(out.netlist, s.name);
        if (!set) return ".step element '" + s.name + "' not found at top level (a V, I or R card)";
        set(s.value);
    }
    return "";
}

// The corner temperature wins over the netlist's .temp (as an explicit .dc request wins over its .dc);
// every temperature-dependent value is resolved by the parser. A .temp list is for .op/.dc (listOk) only,
// a .step list for the analyses that run it (stepOk).
// ponytail: 27 is the DTO's default, so a corner asking for 27 C cannot override a netlist .temp
static double cornerTempC(const SimulationRequestDTO& req) { return req.corner.temperature != 27.0 ? req.corner.temperature : NAN; }

static bool deserialiseRequest(const SimulationRequestDTO& req, SPICEParseResult& out,
                               SimulationResponseDTO& failure, bool listOk = false, bool stepOk = false, bool digitalOk = false,
                               bool hierOk = false) {
    if (const std::string e = parseRequest(req, cornerTempC(req), out); !e.empty()) {
        failure.errorDetail = e;
        return false;
    }
    if (out.hierarchical && !hierOk) {  // S3: .op, .dc, .tran, .ac, .noise and .tf so far
        failure.errorDetail = "this analysis on the hierarchical path (.options hier) is not supported yet";
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

    finish(result);
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
        for (const auto& v : req.mc.variations) {
            std::string n = v.componentName;
            for (auto& ch : n) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
            if (const std::string e = hierShared(run->parsed, n); !e.empty()) { r.errorDetail = e; return false; }
        }
        if (const std::string e = applyVariations(run->parsed.netlist, req.mc.variations, req.mcSeed); !e.empty()) {
            r.errorDetail = e;
            return false;
        }
        run->sim.onConvergenceStep = [&cb](int it, double res) { cb.onConvergenceStep(it, res); };
        for (const auto& d : dims) {
            if (d.name != "TEMP")
                if (const std::string e = hierShared(run->parsed, d.name); !e.empty()) { r.errorDetail = e; return false; }
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

SimulationResponseDTO& SessionImpl::finish(SimulationResponseDTO& r) {
    const bool cancelled = status_.load() == JobStatus::Cancelled;
    if (cancelled) {
        r.converged = false;
        if (r.errorDetail.empty()) r.errorDetail = "Cancelled.";
    }
    r.success = r.converged;  // driver-path consumers read success
    status_.store(cancelled ? JobStatus::Cancelled : r.converged ? JobStatus::Completed : JobStatus::Failed);
    return r;
}

SimulationResponseDTO SessionImpl::dispatch_dc_internal(const SimulationRequestDTO& req,
                                                          const EngineCallbacks& cb) {
    SPICEParseResult parsed;
    SimulationResponseDTO failure;
    if (!deserialiseRequest(req, parsed, failure, true, true, false, true)) return failure;
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
        for (const auto& v : req.mc.variations) {
            std::string n = v.componentName;
            for (auto& ch : n) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
            if (const std::string e = hierShared(parsed, n); !e.empty()) { failure.errorDetail = e; return failure; }
        }
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
    finish(result);
    return result;
}

SimulationResponseDTO SessionImpl::dispatch_ac_internal(const SimulationRequestDTO& req,
                                                          const EngineCallbacks& cb) {
    SPICEParseResult parsed;
    SimulationResponseDTO r;
    if (!deserialiseRequest(req, parsed, r, false, true, false, true)) return r;
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
    if (!r.converged) r.errorDetail = "AC: the small-signal system is singular";
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
    finish(result);
    return result;
}

SimulationResponseDTO SessionImpl::dispatch_tran_internal(const SimulationRequestDTO& req,
                                                            const EngineCallbacks& cb,
                                                            ResultChunkCallback streamCb) {
    SPICEParseResult parsed;
    SimulationResponseDTO failure;
    if (!deserialiseRequest(req, parsed, failure, false, true, true, true)) return failure;
    if (!parsed.steps.empty() && req.stepValues.empty())
        return runSteps(req, parsed.steps, [&](const SimulationRequestDTO& q) { return dispatch_tran_internal(q, cb, streamCb); });
    TensorNetlist& netlist = parsed.netlist;
    const bool uic = parsed.tranUIC || req.transient.uic;
    const double  stopTime = req.transient.stopTimeS != 0 ? req.transient.stopTimeS : parsed.tranStop;  // 0: the netlist's .tran (else validated)
    const double  dt       = req.transient.stepSizeS != 0 ? req.transient.stepSizeS : parsed.tranStep;

    CircuitSim sim;
    sim.execMode = selectExecMode(precision_);
    sim.tranMaxStep = parsed.tranMax;

    SimulationResponseDTO result;
    uint64_t stepIdx = 0;
    const KeptColumns kept = keptColumns(parsed, sim);  // .save: each step stores only these columns
    const size_t fullNodes = (size_t)netlist.numGlobalNodes, fullBranches = sim.branchNames(netlist).size();
    const std::string err = sim.runTransient(
        netlist, dt, stopTime, uic, [&](const SolverStep& step) {
            if (status_.load() == JobStatus::Cancelled) return false;
            WaveformDataPoint pt;
            pt.timeS        = step.time;
            pt.nodeVoltages = step.nodeVoltages;
            pt.branchCurrents = step.branchCurrents;
            keepColumns(kept, pt, fullNodes, fullBranches);
            if (req.transient.keepPoints) result.waveformPoints.push_back(pt);
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
        sim_.tranMaxStep = parsed_.tranMax;
        const double step = req.transient.stepSizeS != 0 ? req.transient.stepSizeS : parsed_.tranStep;
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
    agg.converged    = mc.sampleCount > 0;
    agg.jobId        = jobId_;
    if (!agg.converged) agg.errorDetail = "Monte Carlo needs at least one sample";

    int done = 0;
    for (int i = 0; i < mc.sampleCount && status_.load() != JobStatus::Cancelled; ++i, ++done) {
        SimulationRequestDTO run_req = req;
        run_req.mc     = mc;
        run_req.mcSeed = mc.seed + i;
        auto r = dispatch_dc_internal(run_req, cb);
        if (!r.converged && agg.converged) {  // the first failure, by sample
            agg.converged = false;
            agg.errorDetail = "sample " + std::to_string(i) + ": " + r.errorDetail;
        }
        if (!r.waveformPoints.empty()) {
            agg.waveformPoints.push_back(r.waveformPoints.front());
        }
        cb.onProgress(i + 1, mc.sampleCount);
    }
    if (status_.load() == JobStatus::Cancelled)  // a partial set is not the requested distribution
        agg.errorDetail = "Monte Carlo cancelled after " + std::to_string(done) + " of " + std::to_string(mc.sampleCount) + " samples";
    return finish(agg);
}

// ── Noise ─────────────────────────────────────────────────────────────────
SimulationResponseDTO SessionImpl::runNoise(const SimulationRequestDTO& req,
                                             const EngineCallbacks& /*cb*/) {
    status_.store(JobStatus::Running);
    SimulationResponseDTO r;
    r.jobId        = jobId_;
    r.analysisType = AnalysisTypeDTO::NOISE;
    SPICEParseResult parsed;
    if (deserialiseRequest(req, parsed, r, false, false, false, true)) {
        TensorNetlist& netlist = parsed.netlist;
        const auto freqs = CircuitSim::acFrequencies(parsed.noiseSweep, parsed.noisePoints,
                                                     parsed.noiseStart, parsed.noiseStop, true);
        CircuitSim sim;
        sim.execMode = selectExecMode(precision_);
        std::string shared;
        for (const auto& e : {hierNet(parsed, parsed.noiseOut, "the noise output"), hierNet(parsed, parsed.noiseRef, "the noise reference"),
                              hierShared(parsed, parsed.noiseSource)})
            if (shared.empty()) shared = e;
        if (parsed.noiseSource.empty() || freqs.empty())
            r.errorDetail = "Noise needs .noise v(out[,ref]) src dec|oct|lin n fstart fstop.";
        else if (!shared.empty())
            r.errorDetail = shared;
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
    finish(r);
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
    finish(r);
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
            auto reparse = [&](const SensParam& p, TensorNetlist& nl) {
                SPICEParseResult q;
                if (!parseRequest(req, cornerTempC(req), q, &p).empty()) return false;
                nl = std::move(q.netlist);
                return true;
            };
            if (parsed.sensSweep.empty()) {
                auto sn = sim.solveSens(parsed.netlist, parsed.sensOut, parsed.sensRef, parsed.sensSource,
                                        parsed.sensParams, parsed.sensZeroParams, reparse);
                r.errorDetail = sn.error;
                r.extended.sens = {std::move(sn.names), std::move(sn.values), {}, {}, std::move(sn.unsupported)};
            } else {
                auto freqs = CircuitSim::acFrequencies(parsed.sensSweep, parsed.sensPoints, parsed.sensStart, parsed.sensStop);
                auto sn = sim.solveSensAC(parsed.netlist, parsed.sensOut, parsed.sensRef, parsed.sensSource, freqs,
                                          parsed.sensParams, parsed.sensZeroParams, reparse);
                r.errorDetail = sn.error;
                r.extended.sens = {std::move(sn.names), std::move(sn.values), std::move(freqs), std::move(sn.imag),
                                   std::move(sn.unsupported)};
            }
            r.converged   = r.errorDetail.empty();
            r.extended.hasSens = r.converged;
        }
        r.backend = backendOf(sim, precision_);
    }
    finish(r);
    return r;
}

// ── DC transfer function ──────────────────────────────────────────────────
SimulationResponseDTO SessionImpl::runTF(const SimulationRequestDTO& req,
                                          const EngineCallbacks& /*cb*/) {
    status_.store(JobStatus::Running);
    SimulationResponseDTO r;
    r.jobId        = jobId_;
    r.analysisType = AnalysisTypeDTO::TRANSFER_FUNCTION;
    SPICEParseResult parsed;
    if (deserialiseRequest(req, parsed, r, false, false, false, true)) {
        CircuitSim sim;
        sim.execMode = selectExecMode(precision_);
        std::string shared;
        for (const auto& e : {hierNet(parsed, parsed.tfOut, "the .tf output"), hierNet(parsed, parsed.tfRef, "the .tf reference"),
                              hierShared(parsed, parsed.tfSource),
                              parsed.tfOutSource.empty() ? std::string() : hierShared(parsed, parsed.tfOutSource)})
            if (shared.empty()) shared = e;
        if (!parsed.hasTF)
            r.errorDetail = "Transfer function needs .tf v(out[,ref]) src or .tf i(vsrc) src.";
        else if (!shared.empty())
            r.errorDetail = shared;
        else if (SolverStep dc = sim.solveDC(parsed.netlist); !dc.stats.converged)
            r.errorDetail = "DC operating point failed before transfer function analysis: " + dc.stats.error_detail;
        else {
            auto tf = sim.solveTF(parsed.netlist, parsed.tfOut, parsed.tfRef, parsed.tfOutSource, parsed.tfSource);
            r.errorDetail = tf.error;
            r.converged   = tf.error.empty();
            r.extended.hasDcTF = r.converged;
            auto lower = [](std::string s) { for (auto& c : s) c = (char)std::tolower((unsigned char)c); return s; };
            // ngspice's vectors: transfer_function, then the impedances by name
            std::string in = lower(parsed.tfSource) + "#input_impedance",
                        out = parsed.tfOutSource.empty() ? "output_impedance_at_" + parsed.tfOutLabel
                                                         : lower(parsed.tfOutSource) + "#output_impedance";
            double zin = tf.zin, zout = tf.zout;
            if (out < in) { std::swap(in, out); std::swap(zin, zout); }
            r.extended.dcTf = {{"transfer_function", in, out}, {tf.gain, zin, zout}};
        }
        r.backend = backendOf(sim, precision_);
    }
    finish(r);
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
