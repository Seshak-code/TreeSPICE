#pragma once
// Compiled Verilog-A devices through OSDI (OpenVAF output): BSIM4, BSIM-CMG, ...
//
// A Library is loaded once per path and registers its modules by name; a
// Model is one .model card bound to a module (parameters + setup_model); an
// Instance is one device (instance parameters, setup_instance, node
// collapsing). Each instance numbers its own nodes locally: connected
// terminals 0..T-1, then the internal nodes that survive collapsing. Every
// OSDI call works on small per-instance buffers in that numbering (index 0 of
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

class Instance {
public:
    Instance(std::shared_ptr<Model> model, const std::string &name,
             const std::vector<std::pair<std::string, double>> &params, uint32_t connected, double tempK);
    ~Instance();
    Instance(const Instance &) = delete;
    Instance &operator=(const Instance &) = delete;

    uint32_t terminals() const { return connected_; }
    uint32_t numLocal() const { return numLocal_; }          // terminals + internal nodes
    uint32_t numInternal() const { return numLocal_ - connected_; }
    bool isFlow(uint32_t k) const { return flow_[k]; }  // local unknown k is a branch current

    // Evaluate at local node voltages v[0..numLocal). Afterwards f/q are the
    // resistive/reactive residuals (current / charge out of each local node into
    // the device) and G/C their Jacobians (row-major numLocal x numLocal).
    // Returns false if the model reports a fatal error.
    bool evaluate(const double *v, uint32_t analysisFlags, double abstime);
    const std::string &error() const { return error_; }  // the model's message after a fatal evaluate()
    const std::vector<double> &f() const { return f_; }
    const std::vector<double> &q() const { return q_; }
    double G(uint32_t r, uint32_t c) const { return G_[(r + 1) * (numLocal_ + 1) + c + 1]; }
    double C(uint32_t r, uint32_t c) const { return C_[(r + 1) * (numLocal_ + 1) + c + 1]; }

    // Noise at the last evaluate() (needs CALC_NOISE): per source, density [A^2/Hz]
    // between local nodes a, b (-1 = ground).
    struct NoiseGen { int a, b; double psd; };
    std::vector<NoiseGen> noise(double freq);

    // Transient charge history per local node (engine-integrated), and the
    // local node voltages of the last three accepted steps (LTE).
    std::vector<double> qPrev, iPrev;
    std::vector<double> vHist[3];
    // Where and how the last successful evaluate() ran (device bypass).
    std::vector<double> evalAt;
    uint32_t evalFlags = 0;
    double gmin = 1e-12;  // $simparam gmin handed to the model (.options gmin, set per run by the solver)

private:
    std::shared_ptr<Model> model_;
    std::string name_;
    void *data_ = nullptr;
    uint32_t connected_ = 0, numLocal_ = 0;
    double mult_ = 1.0;  // instance m (simulator multiplier) when the module has no m of its own
    std::string error_;
    std::vector<int> local_;  // descriptor node -> local index, -1 = ground
    std::vector<bool> flow_;
    std::vector<double> solve_, f_, q_, fBuf_, qBuf_, G_, C_, noiseBuf_;
};

}  // namespace osdi
