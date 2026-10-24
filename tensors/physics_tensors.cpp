#include "../tensors/physics_tensors.h"
#include "../physics/physics_constants.h"
#include "../netlist/circuit.h"
#include <iostream>
#include <cmath>
#include <algorithm>

/*
 * physics_tensors.cpp - AoS -> SoA conversion of the linear elements.
 */


// ============================================================================
// AOS -> SOA CONVERSION
// ============================================================================

// Flattened linear elements (R, C, L, V, I) of the global block and every
// instance, in that order (the transient source refresh walks the same order).
TensorizedBlock tensorizeNetlist(const TensorNetlist& netlist) {
    TensorizedBlock result;
    auto collect = [&](const TensorBlock& block, const std::function<int(int)>& remap) {
        for (const auto& r : block.resistors)
            result.resistors.push(remap(r.nodeTerminal1), remap(r.nodeTerminal2), r.resistance_ohms);
        for (const auto& c : block.capacitors)
            result.capacitors.push(remap(c.nodePlate1), remap(c.nodePlate2), c.capacitance_farads);
        for (const auto& ind : block.inductors)
            result.inductors.push(remap(ind.nodeCoil1), remap(ind.nodeCoil2), ind.inductance_henries);
        for (const auto& v : block.voltageSources)
            result.voltageSources.push(remap(v.nodePositive), remap(v.nodeNegative), v.voltage_V);
        for (const auto& i : block.currentSources)
            result.currentSources.push(remap(i.nodePositive), remap(i.nodeNegative), i.m * i.current_A);
    };
    collect(netlist.globalBlock, [](int n) { return n; });
    for (const auto& inst : netlist.instances) {
        auto it = netlist.blockDefinitions.find(inst.blockName);
        if (it == netlist.blockDefinitions.end()) continue;
        collect(it->second, [&](int n) { return (n <= 0 || n > (int)inst.nodeMapping.size()) ? 0 : inst.nodeMapping[n - 1]; });
    }
    return result;
}
