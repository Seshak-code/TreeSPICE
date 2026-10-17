#include "osdi_device.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

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
    if (exists(path)) return path;
    if (const char *env = std::getenv("TREESPICE_OSDI_PATH")) {
        std::string dirs = env;
        size_t start = 0;
        while (start <= dirs.size()) {
            size_t end = dirs.find(PATH_SEP, start);
            if (end == std::string::npos) end = dirs.size();
            const std::string cand = dirs.substr(start, end - start) + "/" + path;
            if (end > start && exists(cand)) return cand;
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

Instance::Instance(std::shared_ptr<Model> model, const std::string &name,
                   const std::vector<std::pair<std::string, double>> &params, uint32_t connected, double tempK)
    : model_(std::move(model)), name_(name), connected_(connected) {
    const Descriptor *d = model_->d;
    if (connected > d->num_terminals || connected == 0)
        throw std::runtime_error(name + ": " + d->name + " has " + std::to_string(d->num_terminals) + " terminals");
    data_ = std::calloc(1, std::max<uint32_t>(d->instance_size, 1));
    for (const auto &[k, v] : params) {
        const uint32_t id = findParam(d, k);
        if (id == NONE && lower(k) == "m") {  // the simulator's device multiplier, as in ngspice
            if (!(v > 0.0)) throw std::runtime_error(name + ": m must be > 0");
            mult_ = v;
            continue;
        }
        if (id == NONE || (d->param_opvar[id].flags & PARA_KIND_MASK) != PARA_KIND_INST)
            throw std::runtime_error(name + ": " + k + " is not an instance parameter of " + d->name);
        writeParam(d, data_, model_->data, id, v, ACCESS_FLAG_INSTANCE, name);
    }
    double gminVal = 1e-12;  // setup sees the default; evaluate() gets the run's gmin
    SimParas sp = simParas(&gminVal);
    InitInfo r{0, 0, nullptr};
    d->setup_instance((void *)name_.c_str(), data_, model_->data, tempK, connected, &sp, &r);
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
        if (from < connected && (toGnd || to < connected)) continue;  // terminals cannot be collapsed
        if (!toGnd && from < to) std::swap(from, to);
        for (auto &m : map) {
            if (m == from) m = toGnd ? NONE : to;
            else if (m != NONE && m > from) --m;
        }
    }
    numLocal_ = 0;
    local_.assign(N, -1);
    for (uint32_t i = 0; i < N; ++i)
        if (map[i] != NONE) { local_[i] = (int)map[i]; numLocal_ = std::max(numLocal_, map[i] + 1); }
    flow_.assign(numLocal_, false);  // unknowns that are branch currents (V(a,b) <+ ... branches)
    for (uint32_t i = 0; i < N; ++i)
        if (local_[i] >= 0 && d->nodes[i].is_flow) flow_[local_[i]] = true;

    // OSDI-facing numbering: slot 0 = ground, local node k = slot k + 1. The
    // Jacobian pointers address dense (numLocal+1)^2 buffers in that numbering.
    uint32_t *nodeMap = (uint32_t *)((char *)data_ + d->node_mapping_offset);
    for (uint32_t i = 0; i < N; ++i) nodeMap[i] = local_[i] < 0 ? 0 : (uint32_t)local_[i] + 1;
    const size_t L1 = numLocal_ + 1;
    G_.assign(L1 * L1, 0.0);
    C_.assign(L1 * L1, 0.0);
    double **resistPtr = (double **)((char *)data_ + d->jacobian_ptr_resist_offset);
    for (uint32_t e = 0; e < d->num_jacobian_entries; ++e) {
        const JacobianEntry &je = d->jacobian_entries[e];
        const size_t at = nodeMap[je.nodes.node_1] * L1 + nodeMap[je.nodes.node_2];
        resistPtr[e] = &G_[at];
        if (je.react_ptr_off != NONE) *(double **)((char *)data_ + je.react_ptr_off) = &C_[at];
    }
    solve_.assign(L1, 0.0);
    fBuf_.assign(L1, 0.0);
    qBuf_.assign(L1, 0.0);
    f_.assign(numLocal_, 0.0);
    q_.assign(numLocal_, 0.0);
    qPrev.assign(numLocal_, 0.0);
    iPrev.assign(numLocal_, 0.0);
    for (auto &h : vHist) h.assign(numLocal_, 0.0);
    noiseBuf_.assign(d->num_noise_src, 0.0);
}

Instance::~Instance() { std::free(data_); }

bool Instance::evaluate(const double *v, uint32_t analysisFlags, double abstime) {
    const Descriptor *d = model_->d;
    solve_[0] = 0.0;
    for (uint32_t k = 0; k < numLocal_; ++k) solve_[k + 1] = v[k];
    std::fill(fBuf_.begin(), fBuf_.end(), 0.0);
    std::fill(qBuf_.begin(), qBuf_.end(), 0.0);
    std::fill(G_.begin(), G_.end(), 0.0);
    std::fill(C_.begin(), C_.end(), 0.0);
    double gminVal = gmin;
    SimInfo info{simParas(&gminVal), abstime, solve_.data(), nullptr, nullptr,
                 analysisFlags | CALC_RESIST_RESIDUAL | CALC_REACT_RESIDUAL | CALC_RESIST_JACOBIAN |
                     CALC_REACT_JACOBIAN};
    const uint32_t ret = d->eval((void *)name_.c_str(), data_, model_->data, &info);
    d->load_residual_resist(data_, model_->data, fBuf_.data());
    d->load_residual_react(data_, model_->data, qBuf_.data());
    d->load_jacobian_resist(data_, model_->data);
    d->load_jacobian_react(data_, model_->data, 1.0);
    if (mult_ != 1.0) {  // m parallel devices: every current, charge and derivative scales
        for (auto &x : fBuf_) x *= mult_;
        for (auto &x : qBuf_) x *= mult_;
        for (auto &x : G_) x *= mult_;
        for (auto &x : C_) x *= mult_;
    }
    for (uint32_t k = 0; k < numLocal_; ++k) { f_[k] = fBuf_[k + 1]; q_[k] = qBuf_[k + 1]; }
    if (ret & EVAL_RET_FLAG_FATAL) { error_ = lastError.empty() ? "fatal model error" : lastError; evalFlags = 0; return false; }
    evalAt.assign(v, v + numLocal_);
    evalFlags = analysisFlags;
    return true;
}

std::vector<Instance::NoiseGen> Instance::noise(double freq) {
    const Descriptor *d = model_->d;
    std::vector<NoiseGen> out;
    if (!d->num_noise_src) return out;
    std::fill(noiseBuf_.begin(), noiseBuf_.end(), 0.0);
    d->load_noise(data_, model_->data, freq, noiseBuf_.data());
    for (uint32_t s = 0; s < d->num_noise_src; ++s) {
        const NodePair &np = d->noise_sources[s].nodes;
        const int a = np.node_1 == NONE ? -1 : local_[np.node_1];
        const int b = np.node_2 == NONE ? -1 : local_[np.node_2];
        if (a != b) out.push_back({a, b, mult_ * noiseBuf_[s]});
    }
    return out;
}

}  // namespace osdi

// ---------------------------------------------------------------------------
// TensorNetlist::instantiateOsdi (declared in netlist/circuit.h)
// ---------------------------------------------------------------------------
#include "../netlist/circuit.h"

void TensorNetlist::instantiateOsdi(double tempK) {
    auto make = [&](const TensorBlock &blk, BlockState &st, const std::string &prefix) {
        st.osdiState.resize(blk.osdiDevices.size());
        for (size_t i = 0; i < blk.osdiDevices.size(); ++i) {
            const OsdiDevice &dev = blk.osdiDevices[i];
            if (st.osdiState[i] || !dev.model) continue;
            st.osdiState[i] = std::make_shared<osdi::Instance>(dev.model, prefix + dev.instanceName, dev.params,
                                                               (uint32_t)dev.nodes.size(), tempK);
        }
    };
    make(globalBlock, globalState, "");
    for (size_t k = 0; k < instances.size() && k < instanceStates.size(); ++k) {
        auto it = blockDefinitions.find(instances[k].blockName);
        if (it != blockDefinitions.end()) make(it->second, instanceStates[k], instances[k].blockName + "#" + std::to_string(k) + ".");
    }
}
