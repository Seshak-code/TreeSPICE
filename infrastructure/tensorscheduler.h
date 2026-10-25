#pragma once
#include "../netlist/circuit.h"
#include <map>
#include <string>
#include <vector>

// SolverJob: the instances of one block definition (they share its element values)
struct SolverJob {
    const TensorBlock* blockTmpl;
    std::vector<int> instanceIds;
};

// TensorScheduler: groups the netlist's instances by block definition for the per-block audits and stamps
class TensorScheduler {
public:
    std::vector<SolverJob> jobs;

    void schedule(const TensorNetlist& netlist) {
        jobs.clear();
        // By block definition, not topology hash: the hash ignores values, so same-shaped blocks would be
        // audited and stamped with the first one's
        std::map<std::string, std::vector<int>> byBlock;
        for (int i = 0; i < (int)netlist.instances.size(); ++i)
            if (netlist.blockDefinitions.count(netlist.instances[i].blockName))
                byBlock[netlist.instances[i].blockName].push_back(i);
        for (auto const& [name, instances] : byBlock)
            jobs.push_back({ &netlist.blockDefinitions.at(name), instances });
    }
};
