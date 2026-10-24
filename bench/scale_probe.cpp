// scale_probe <netlist>: live heap bytes (macOS malloc zones) after parsing and after the DC operating point,
// per device; the scale track's (S1) measure. Prints the sizes of the per-device and per-instance records too.
#include "netlist/spice_parser.h"
#include "physics/circuitsim.h"
#include "osdi/osdi_abi.h"
#include "osdi/osdi_device.h"
#include <malloc/malloc.h>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>

using namespace acutesim;

static double liveMB() {
    malloc_statistics_t s{};
    malloc_zone_statistics(nullptr, &s);
    return s.size_in_use / 1048576.0;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: scale_probe <netlist>\n"); return 2; }
    std::ifstream f(argv[1]);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();
    std::printf("sizeof Mosfet %zu, DeviceCharge %zu, BlockState %zu, BlockInstance %zu\n", sizeof(Mosfet),
                sizeof(BlockState::DeviceCharge), sizeof(BlockState), sizeof(BlockInstance));
    const double m0 = liveMB();
    auto t0 = std::chrono::steady_clock::now();
    auto parsed = SPICEParser().parse(text, "");
    if (!parsed.success) { std::fprintf(stderr, "parse: %s\n", parsed.errorMessage.c_str()); return 1; }
    const double m1 = liveMB();
    auto t1 = std::chrono::steady_clock::now();
    auto count = [](const TensorBlock& b) { return b.mosfets.size() + b.osdiDevices.size() + b.diodes.size() + b.bjts.size(); };
    size_t devices = count(parsed.netlist.globalBlock);
    const OsdiDevice* od = nullptr;
    for (const auto& in : parsed.netlist.instances) {
        const auto& b = parsed.netlist.blockDefinitions.at(in.blockName);
        devices += count(b);
        if (!od && !b.osdiDevices.empty()) od = &b.osdiDevices[0];
    }
    if (od && od->model) {
        const auto* d = od->model->d;
        std::printf("OSDI %s: instance_size %u, model_size %u, nodes %u, jacobian entries %u\n", d->name,
                    d->instance_size, d->model_size, d->num_nodes, d->num_jacobian_entries);
    }
    for (const auto& st : parsed.netlist.instanceStates)
        if (!st.osdiState.empty() && st.osdiState[0]) {
            const size_t L = st.osdiState[0]->numLocal();
            std::printf("OSDI local nodes %zu: dense G+C %zu B per instance\n", L, 2 * (L + 1) * (L + 1) * sizeof(double));
            break;
        }
    std::printf("instances %zu, MOSFETs %zu, nodes %d\n", parsed.netlist.instances.size(), devices, parsed.netlist.numGlobalNodes);
    std::printf("parse: %.1f MB (%.0f B/device) %.2f s\n", m1 - m0, (m1 - m0) * 1048576.0 / devices,
                std::chrono::duration<double>(t1 - t0).count());
    CircuitSim sim;
    double peak = 0.0;
    const auto r = sim.solveDC(parsed.netlist);
    const double m2 = liveMB();
    peak = m2 - m0;
    auto t2 = std::chrono::steady_clock::now();
    std::printf("op: converged %d, live after %.1f MB (%.0f B/device; solver-held %.1f MB) %.2f s\n", (int)r.stats.converged,
                peak, peak * 1048576.0 / devices, m2 - m1, std::chrono::duration<double>(t2 - t1).count());
    return 0;
}
