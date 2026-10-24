#include <filesystem>
#include "osdi_device.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <tuple>

#if defined(_WIN32)
#  include <windows.h>
static void *openLib(const std::string &p) { return (void *)LoadLibraryA(p.c_str()); }
static void *sym(void *h, const char *s) { return (void *)GetProcAddress((HMODULE)h, s); }
static std::string libError() { return "LoadLibrary failed"; }
#  define PATH_SEP ';'
#else
#  include <dlfcn.h>
static void *openLib(const std::string &p) { return dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL); }
static void *sym(void *h, const char *s) { return dlsym(h, s); }
static std::string libError() { const char *e = dlerror(); return e ? e : "dlopen failed"; }
#  define PATH_SEP ':'
#endif

namespace osdi {
namespace {

std::string lower(std::string s) {
    for (auto &c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

// Modules by lowercase name; libraries stay loaded for the process lifetime.
std::map<std::string, const Descriptor *> &modules() {
    static std::map<std::string, const Descriptor *> m;
    return m;
}
std::map<std::string, void *> &libraries() {
    static std::map<std::string, void *> m;
    return m;
}

// The last error the models logged (their $error / $fatal text), for Instance::error().
thread_local std::string lastError;

extern "C" void treespiceOsdiLog(void *handle, char *msg, uint32_t lvl) {
    static const char *names[] = {"debug", "display", "info", "warning", "error", "fatal"};
    const uint32_t l = lvl & LOG_LVL_MASK;
    if (l >= LOG_LVL_ERR) lastError = msg ? msg : "";
    std::fprintf(stderr, "[OSDI %s] %s: %s\n", handle ? (const char *)handle : "?", l <= 5 ? names[l] : "log",
                 msg ? msg : "");
}

// $simparam values the models may ask for (ngspice passes the same gmin).
char kGmin[] = "gmin";
char *kSimNames[] = {kGmin, nullptr};
char *kSimStrNames[] = {nullptr};
SimParas simParas(double *gmin) { return SimParas{kSimNames, gmin, kSimStrNames, nullptr}; }

// The parameter of d named `name` (any alias, case-insensitive), or NONE.
uint32_t findParam(const Descriptor *d, const std::string &name) {
    const std::string n = lower(name);
    for (uint32_t i = 0; i < d->num_params; ++i) {
        const ParamOpvar &p = d->param_opvar[i];
        for (uint32_t a = 0; a <= p.num_alias; ++a)
            if (p.name[a] && lower(p.name[a]) == n) return i;
    }
    return NONE;
}

void writeParam(const Descriptor *d, void *inst, void *model, uint32_t id, double value, uint32_t flags,
                const std::string &owner) {
    const ParamOpvar &p = d->param_opvar[id];
    void *dst = d->access(inst, model, id, ACCESS_FLAG_SET | flags);
    if (!dst || p.len != 0 || (p.flags & PARA_TY_MASK) == PARA_TY_STR)
        throw std::runtime_error(owner + ": parameter " + p.name[0] + " (array/string) is not supported yet");
    if ((p.flags & PARA_TY_MASK) == PARA_TY_INT) *(int32_t *)dst = (int32_t)std::lround(value);
    else *(double *)dst = value;
}

void checkInit(const Descriptor *d, const InitInfo &r, const std::string &owner) {
    if (r.num_errors == 0) return;
    std::string what = owner + ": parameter out of bounds";
    if (r.errors && r.errors[0].parameter_id < d->num_params)
        what += " (" + std::string(d->param_opvar[r.errors[0].parameter_id].name[0]) + ")";
    std::free(r.errors);
    throw std::runtime_error(what);
}

}  // namespace

std::string resolvePath(const std::string &path) {
    auto exists = [](const std::string &p) { FILE *f = std::fopen(p.c_str(), "rb"); if (f) std::fclose(f); return f != nullptr; };
    // absolute: dlopen and LoadLibrary search their own lists, not the working directory, for a bare file name
    auto abs = [](const std::string &p) { return std::filesystem::absolute(p).string(); };
    if (exists(path)) return abs(path);
    if (const char *env = std::getenv("TREESPICE_OSDI_PATH")) {
        std::string dirs = env;
        size_t start = 0;
        while (start <= dirs.size()) {
            size_t end = dirs.find(PATH_SEP, start);
            if (end == std::string::npos) end = dirs.size();
            const std::string cand = dirs.substr(start, end - start) + "/" + path;
            if (end > start && exists(cand)) return abs(cand);
            start = end + 1;
        }
    }
    throw std::runtime_error("OSDI library '" + path + "' not found (also searched TREESPICE_OSDI_PATH)");
}

void loadLibrary(const std::string &given) {
    const std::string path = resolvePath(given);
    if (libraries().count(path)) return;
    void *h = openLib(path);
    if (!h) throw std::runtime_error("cannot load OSDI library '" + path + "': " + libError());
    auto *major = (const uint32_t *)sym(h, "OSDI_VERSION_MAJOR");
    auto *minor = (const uint32_t *)sym(h, "OSDI_VERSION_MINOR");
    auto *count = (const uint32_t *)sym(h, "OSDI_NUM_DESCRIPTORS");
    auto *base = (const char *)sym(h, "OSDI_DESCRIPTORS");
    if (!major || !minor || !count || !base) throw std::runtime_error("'" + path + "' is not an OSDI library");
    if (*major != 0 || (*minor != 3 && *minor != 4))
        throw std::runtime_error("'" + path + "': OSDI version " + std::to_string(*major) + "." +
                                 std::to_string(*minor) + " is not supported (0.3, 0.4)");
    auto *size = (const uint32_t *)sym(h, "OSDI_DESCRIPTOR_SIZE");  // 0.4: step through the table by this
    const size_t stride = (*minor >= 4 && size) ? *size : sizeof(Descriptor);
    if (auto *logVar = (void **)sym(h, "osdi_log")) *logVar = (void *)&treespiceOsdiLog;
    // $limit functions: none supported yet (BSIM4 and BSIM-CMG from OpenVAF use none)
    if (auto *limLen = (const uint32_t *)sym(h, "OSDI_LIM_TABLE_LEN"); limLen && *limLen > 0)
        throw std::runtime_error("'" + path + "': $limit functions are not supported yet");
    for (uint32_t i = 0; i < *count; ++i) {
        const auto *d = (const Descriptor *)(base + i * stride);
        if (d->num_states)  // model state across steps (limiting); BSIM4 and BSIM-CMG keep none
            throw std::runtime_error("'" + path + "': module " + d->name + " keeps state, not supported yet");
        modules()[lower(d->name)] = d;
    }
    libraries()[path] = h;
}

const Descriptor *findModule(const std::string &name) {
    auto it = modules().find(lower(name));
    return it == modules().end() ? nullptr : it->second;
}

Model::Model(const Descriptor *desc, const std::string &cardName, const std::map<std::string, double> &params)
    : d(desc), data(std::calloc(1, std::max<uint32_t>(desc->model_size, 1))), name(cardName) {
    for (const auto &[k, v] : params) {
        const std::string key = lower(k);
        const uint32_t id = findParam(d, k);
        if (key == "level" || (key == "version" && id == NONE)) continue;  // SPICE card bookkeeping
        if (id == NONE) throw std::runtime_error("model " + name + ": parameter " + k + " is not a " + d->name + " parameter");
        writeParam(d, nullptr, data, id, v, 0, "model " + name);
    }
    double gminVal = 1e-12;  // setup sees the default; evaluate() gets the run's gmin
    SimParas sp = simParas(&gminVal);
    InitInfo r{0, 0, nullptr};
    d->setup_model((void *)name.c_str(), data, &sp, &r);
    checkInit(d, r, "model " + name);
}

Model::~Model() { std::free(data); }

Setup::Setup(std::shared_ptr<Model> m, const std::string &name,
             const std::vector<std::pair<std::string, double>> &params, uint32_t conn, double tempK)
    : model(std::move(m)), connected(conn) {
    const Descriptor *d = model->d;
    if (conn > d->num_terminals || conn == 0)
        throw std::runtime_error(name + ": " + d->name + " has " + std::to_string(d->num_terminals) + " terminals");
    data_ = std::calloc(1, std::max<uint32_t>(d->instance_size, 1));
    try {
        init(name, params, tempK);
    } catch (...) {
        std::free(data_);
        throw;
    }
}

void Setup::init(const std::string &name, const std::vector<std::pair<std::string, double>> &params, double tempK) {
    const Descriptor *d = model->d;
    const uint32_t conn = connected;
    for (const auto &[k, v] : params) {
        const uint32_t id = findParam(d, k);
        if (id == NONE && lower(k) == "m") {  // the simulator's device multiplier, as in ngspice
            if (!(v > 0.0)) throw std::runtime_error(name + ": m must be > 0");
            mult = v;
            continue;
        }
        if (id == NONE || (d->param_opvar[id].flags & PARA_KIND_MASK) != PARA_KIND_INST)
            throw std::runtime_error(name + ": " + k + " is not an instance parameter of " + d->name);
        writeParam(d, data_, model->data, id, v, ACCESS_FLAG_INSTANCE, name);
    }
    double gminVal = 1e-12;  // setup sees the default; evaluate() gets the run's gmin
    SimParas sp = simParas(&gminVal);
    InitInfo r{0, 0, nullptr};
    d->setup_instance((void *)name.c_str(), data_, model->data, tempK, conn, &sp, &r);
    checkInit(d, r, name);

    // Node collapsing, as OpenVAF's reference simulator does it: each collapsed
    // pair merges its higher-numbered node into the lower one (or into ground);
    // terminals never disappear. Surviving nodes are renumbered densely.
    const uint32_t N = d->num_nodes;
    std::vector<uint32_t> map(N);
    for (uint32_t i = 0; i < N; ++i) map[i] = i;
    const bool *collapsed = (const bool *)((char *)data_ + d->collapsed_offset);
    for (uint32_t c = 0; c < d->num_collapsible; ++c) {
        if (!collapsed[c]) continue;
        uint32_t from = map[d->collapsible[c].node_1];
        const uint32_t to0 = d->collapsible[c].node_2;
        bool toGnd = to0 == NONE;
        uint32_t to = toGnd ? NONE : map[to0];
        toGnd = toGnd || to == NONE;
        if (from == NONE && !toGnd) { from = to; toGnd = true; }  // its partner is already ground
        if (from == NONE || from == to) continue;
        if (from < conn && (toGnd || to < conn)) continue;  // terminals cannot be collapsed
        if (!toGnd && from < to) std::swap(from, to);
        for (auto &mm : map) {
            if (mm == from) mm = toGnd ? NONE : to;
            else if (mm != NONE && mm > from) --mm;
        }
    }
    local.assign(N, -1);
    for (uint32_t i = 0; i < N; ++i)
        if (map[i] != NONE) { local[i] = (int)map[i]; numLocal = std::max(numLocal, map[i] + 1); }
    flow.assign(numLocal, false);  // unknowns that are branch currents (V(a,b) <+ ... branches)
    for (uint32_t i = 0; i < N; ++i)
        if (local[i] >= 0 && d->nodes[i].is_flow) flow[local[i]] = true;

    // OSDI-facing numbering: slot 0 = ground, local node k = slot k + 1. The Jacobian pointers address dense
    // (numLocal+1)^2 buffers in that numbering (set per evaluation, on the thread's buffers); slot / numSlots
    // index the distinct local positions for a compact copy.
    uint32_t *nodeMap = (uint32_t *)((char *)data_ + d->node_mapping_offset);
    for (uint32_t i = 0; i < N; ++i) nodeMap[i] = local[i] < 0 ? 0 : (uint32_t)local[i] + 1;
    const size_t L1 = numLocal + 1;
    slot.assign(numLocal * numLocal, -1);
    for (uint32_t e = 0; e < d->num_jacobian_entries; ++e) {
        const JacobianEntry &je = d->jacobian_entries[e];
        const uint32_t a = nodeMap[je.nodes.node_1], b = nodeMap[je.nodes.node_2];
        entryAt.push_back((uint32_t)(a * L1 + b));
        if (a > 0 && b > 0 && slot[(a - 1) * numLocal + b - 1] < 0) slot[(a - 1) * numLocal + b - 1] = (int)numSlots++;
    }
}

Setup::~Setup() { std::free(data_); }

namespace {
// One thread's evaluation buffers: the instance block copied from a Setup, and the OSDI-facing (ground at 0)
// solution, residuals and dense Jacobians it writes
struct Scratch {
    std::vector<std::max_align_t> block;
    std::vector<double> solve, f, q, G, C, noise;
};
thread_local Scratch scratch;

// Run eval and the loads on this thread's copy of the setup at v; s holds the results (multiplier applied)
uint32_t evalOn(const Setup &st, const std::string &name, const double *v, uint32_t flags, double abstime, double gmin) {
    const Descriptor *d = st.model->d;
    Scratch &s = scratch;
    const size_t L1 = st.numLocal + 1, words = (d->instance_size + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t);
    if (s.block.size() < std::max<size_t>(words, 1)) s.block.resize(std::max<size_t>(words, 1));
    std::memcpy(s.block.data(), st.block(), d->instance_size);
    char *blk = (char *)s.block.data();
    s.solve.assign(L1, 0.0);
    s.f.assign(L1, 0.0);
    s.q.assign(L1, 0.0);
    s.G.assign(L1 * L1, 0.0);
    s.C.assign(L1 * L1, 0.0);
    double **resistPtr = (double **)(blk + d->jacobian_ptr_resist_offset);
    for (uint32_t e = 0; e < d->num_jacobian_entries; ++e) {
        resistPtr[e] = &s.G[st.entryAt[e]];
        const uint32_t off = d->jacobian_entries[e].react_ptr_off;
        if (off != NONE) *(double **)(blk + off) = &s.C[st.entryAt[e]];
    }
    for (uint32_t k = 0; k < st.numLocal; ++k) s.solve[k + 1] = v[k];
    double gminVal = gmin;
    SimInfo info{simParas(&gminVal), abstime, s.solve.data(), nullptr, nullptr,
                 flags | CALC_RESIST_RESIDUAL | CALC_REACT_RESIDUAL | CALC_RESIST_JACOBIAN | CALC_REACT_JACOBIAN};
    const uint32_t ret = d->eval((void *)name.c_str(), blk, st.model->data, &info);
    d->load_residual_resist(blk, st.model->data, s.f.data());
    d->load_residual_react(blk, st.model->data, s.q.data());
    d->load_jacobian_resist(blk, st.model->data);
    d->load_jacobian_react(blk, st.model->data, 1.0);
    if (st.mult != 1.0) {  // m parallel devices: every current, charge and derivative scales
        for (auto *b : {&s.f, &s.q, &s.G, &s.C})
            for (auto &x : *b) x *= st.mult;
    }
    return ret;
}
}  // namespace

Instance::Instance(std::shared_ptr<const Setup> setup, std::string name, bool keepEval)
    : setup_(std::move(setup)), name_(std::move(name)), keep_(keepEval) {
    s_.assign((keep_ ? 8 * numLocal() + 2 * setup_->numSlots : 5 * numLocal()), 0.0);
}

const std::string &Instance::error() { return lastError; }

bool Instance::evaluate(const double *v, uint32_t analysisFlags, double abstime, Eval &out) {
    const Setup &st = *setup_;
    const uint32_t L = st.numLocal, L1 = L + 1;
    lastError.clear();
    const uint32_t ret = evalOn(st, name_, v, analysisFlags, abstime, gmin);
    const Scratch &s = scratch;
    out.f.resize(L); out.q.resize(L); out.G.resize(L * L); out.C.resize(L * L);
    for (uint32_t a = 0; a < L; ++a) {
        out.f[a] = s.f[a + 1];
        out.q[a] = s.q[a + 1];
        for (uint32_t b = 0; b < L; ++b) { out.G[a * L + b] = s.G[(a + 1) * L1 + b + 1]; out.C[a * L + b] = s.C[(a + 1) * L1 + b + 1]; }
    }
    if (ret & EVAL_RET_FLAG_FATAL) {
        if (lastError.empty()) lastError = "fatal model error";
        evalFlags_ = 0;
        return false;
    }
    if (keep_) {
        std::copy(v, v + L, &s_[5 * L]);
        std::copy(out.f.begin(), out.f.end(), &s_[6 * L]);
        std::copy(out.q.begin(), out.q.end(), &s_[7 * L]);
        double *g = &s_[8 * L], *c = g + st.numSlots;
        for (uint32_t k = 0; k < L * L; ++k)
            if (st.slot[k] >= 0) { g[st.slot[k]] = out.G[k]; c[st.slot[k]] = out.C[k]; }
    }
    evalFlags_ = analysisFlags;
    return true;
}

std::vector<Instance::NoiseGen> Instance::noise(const double *v, double freq) {
    const Setup &st = *setup_;
    const Descriptor *d = st.model->d;
    std::vector<NoiseGen> out;
    if (!d->num_noise_src) return out;
    evalOn(st, name_, v, ANALYSIS_NOISE | CALC_NOISE, 0.0, gmin);  // the noise load reads this evaluation
    Scratch &s = scratch;
    s.noise.assign(d->num_noise_src, 0.0);
    d->load_noise(s.block.data(), st.model->data, freq, s.noise.data());
    for (uint32_t k = 0; k < d->num_noise_src; ++k) {
        const NodePair &np = d->noise_sources[k].nodes;
        const int a = np.node_1 == NONE ? -1 : st.local[np.node_1];
        const int b = np.node_2 == NONE ? -1 : st.local[np.node_2];
        if (a != b) out.push_back({a, b, st.mult * s.noise[k]});
    }
    return out;
}

}  // namespace osdi

// ---------------------------------------------------------------------------
// TensorNetlist::instantiateOsdi (declared in netlist/circuit.h)
// ---------------------------------------------------------------------------
#include "../netlist/circuit.h"

void TensorNetlist::instantiateOsdi(double tempK, bool keepEval) {
    // one Setup per model, instance parameters and terminal count (tempK is one per call)
    std::map<std::tuple<const osdi::Model *, std::vector<std::pair<std::string, double>>, uint32_t>,
             std::shared_ptr<const osdi::Setup>> setups;
    auto make = [&](const TensorBlock &blk, BlockState &st, const std::string &prefix) {
        st.osdiState.resize(blk.osdiDevices.size());
        for (size_t i = 0; i < blk.osdiDevices.size(); ++i) {
            const OsdiDevice &dev = blk.osdiDevices[i];
            if (st.osdiState[i] || !dev.model) continue;
            auto &su = setups[{dev.model.get(), dev.params, (uint32_t)dev.nodes.size()}];
            if (!su) su = std::make_shared<const osdi::Setup>(dev.model, prefix + dev.instanceName, dev.params,
                                                              (uint32_t)dev.nodes.size(), tempK);
            st.osdiState[i] = std::make_shared<osdi::Instance>(su, prefix + dev.instanceName, keepEval);
        }
    };
    make(globalBlock, globalState, "");
    for (size_t k = 0; k < instances.size() && k < instanceStates.size(); ++k) {
        auto it = blockDefinitions.find(instances[k].blockName);
        if (it != blockDefinitions.end()) make(it->second, instanceStates[k], instances[k].blockName + "#" + std::to_string(k) + ".");
    }
}
