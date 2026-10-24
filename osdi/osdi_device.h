#pragma once
// Compiled Verilog-A devices through OSDI (OpenVAF output): BSIM4, BSIM-CMG, ...
//
// A Library is loaded once per path and registers its modules by name; a
// Model is one .model card bound to a module (parameters + setup_model); an
// Instance is one device (instance parameters, setup_instance, node
// collapsing; identical devices share one Setup). Each instance numbers its own nodes locally: connected
// terminals 0..T-1, then the internal nodes that survive collapsing. Every
// OSDI call works on per-thread buffers in that numbering (index 0 of
// the OSDI-facing arrays is ground); the engine maps local nodes to unknowns.
#include "osdi_abi.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace osdi {

// Load an OSDI library (once per path) and register its modules. Throws on failure.
void loadLibrary(const std::string &path);
// The module with this name (case-insensitive), or null.
const Descriptor *findModule(const std::string &name);
// Resolve a .osdi path: as given, else relative to each directory of TREESPICE_OSDI_PATH.
std::string resolvePath(const std::string &path);

class Model {
public:
    // params: .model card values (any case). Instance parameters given here
    // become the defaults of every instance. Throws on unknown parameters.
    Model(const Descriptor *d, const std::string &cardName, const std::map<std::string, double> &params);
    ~Model();
    Model(const Model &) = delete;
    Model &operator=(const Model &) = delete;
    const Descriptor *d;
    void *data;
    std::string name;
};

// What an instance's evaluation starts from: the instance parameters written, setup_instance run, the
// nodes collapsed. Devices with the same model, instance parameters, terminal count and temperature share
// one (TensorNetlist::instantiateOsdi): a module's instance block is kilobytes (BSIM4: 4.3 KB). Each
// evaluation runs on a per-thread copy of the block. Immutable after construction.
class Setup {
public:
    Setup(std::shared_ptr<Model> model, const std::string &name,
          const std::vector<std::pair<std::string, double>> &params, uint32_t connected, double tempK);
    ~Setup();
    Setup(const Setup &) = delete;
    Setup &operator=(const Setup &) = delete;
    const std::shared_ptr<Model> model;
    uint32_t connected = 0, numLocal = 0;
    double mult = 1.0;                // instance m (simulator multiplier) when the module has no m of its own
    std::vector<int> local;           // descriptor node -> local index, -1 = ground
    std::vector<bool> flow;           // local unknown is a branch current
    std::vector<uint32_t> entryAt;    // per Jacobian entry: its slot in the dense (numLocal+1)^2 buffers
    std::vector<int> slot;            // local (row * numLocal + col) -> compact Jacobian index, -1: none
    uint32_t numSlots = 0;            // distinct local Jacobian positions
    const void *block() const { return data_; }

private:
    void init(const std::string &name, const std::vector<std::pair<std::string, double>> &params, double tempK);
    void *data_ = nullptr;
};

// One evaluation in local numbering: residuals f (resistive), q (reactive) and their Jacobians G, C
// (row-major numLocal x numLocal)
struct Eval { std::vector<double> f, q, G, C; };

class Instance {
public:
    // keepEval: remember each evaluation (point, analysis, f, q and the compact G, C) for device bypass
    Instance(std::shared_ptr<const Setup> setup, std::string name, bool keepEval);
    Instance(const Instance &) = delete;
    Instance &operator=(const Instance &) = delete;

    uint32_t terminals() const { return setup_->connected; }
    uint32_t numLocal() const { return setup_->numLocal; }   // terminals + internal nodes
    uint32_t numInternal() const { return setup_->numLocal - setup_->connected; }
    bool isFlow(uint32_t k) const { return setup_->flow[k]; }  // local unknown k is a branch current

    // Evaluate at local node voltages v[0..numLocal) into out; f/q are the currents / charges out of each
    // local node into the device. False on a fatal model error (error() says why, on this thread).
    bool evaluate(const double *v, uint32_t analysisFlags, double abstime, Eval &out);
    static const std::string &error();

    // Noise at local voltages v (evaluated there with CALC_NOISE): per source, density [A^2/Hz] between
    // local nodes a, b (-1 = ground).
    struct NoiseGen { int a, b; double psd; };
    std::vector<NoiseGen> noise(const double *v, double freq);

    // The last kept evaluation (keepEval), for bypass: the point, and f, q, G, C there
    bool kept(uint32_t analysisFlags) const { return keep_ && evalFlags_ == analysisFlags; }
    const double *evalAt() const { return &s_[5 * numLocal()]; }
    double f(uint32_t k) const { return s_[6 * numLocal() + k]; }
    double q(uint32_t k) const { return s_[7 * numLocal() + k]; }
    double G(uint32_t r, uint32_t c) const { return compact(r, c, 0); }
    double C(uint32_t r, uint32_t c) const { return compact(r, c, 1); }

    // Transient charge history per local node (engine-integrated), and the local node voltages of the last
    // three accepted steps (LTE)
    double *qPrev() { return &s_[0]; }
    double *iPrev() { return &s_[numLocal()]; }
    double *vHist(int i) { return &s_[(2 + i) * numLocal()]; }
    const double *vHist(int i) const { return &s_[(2 + i) * numLocal()]; }
    double gmin = 1e-12;  // $simparam gmin handed to the model (.options gmin, set per run by the solver)

private:
    double compact(uint32_t r, uint32_t c, int which) const {
        const int k = setup_->slot[r * numLocal() + c];
        return k < 0 ? 0.0 : s_[8 * numLocal() + which * setup_->numSlots + k];
    }
    std::shared_ptr<const Setup> setup_;
    std::string name_;
    bool keep_ = false;
    uint32_t evalFlags_ = 0;
    // qPrev, iPrev, vHist[0..2] (numLocal each); with keepEval then evalAt, f, q (numLocal each), G, C (numSlots each)
    std::vector<double> s_;
};

}  // namespace osdi
