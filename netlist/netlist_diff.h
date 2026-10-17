#pragma once
#include "circuit.h"
#include "graph_canonical.h"
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

/*
 * netlist_diff.h - Semantic Netlist Differ
 * Implement semantic comparison between two TensorBlocks.
 */

enum class DiffChangeType {
    ADDED,
    REMOVED,
    MODIFIED_PARAMETER,
    CONNECTIVITY_CHANGED
};

// Simulation Impact Bitmask
enum class SimulationImpact : uint32_t {
    NONE = 0,
    MATRIX_REBUILD = 1 << 0,     // Connectivity change -> sparse pattern change
    JACOBIAN_UPDATE = 1 << 1,    // Parameter change -> numeric update only
    STATE_RESET = 1 << 2         // History invalid -> clear transient history
};

inline SimulationImpact operator|(SimulationImpact a, SimulationImpact b) {
    return static_cast<SimulationImpact>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

struct ComponentChange {
    std::string name;
    std::string type;
    DiffChangeType changeType;
    std::string details;
    uint32_t impactMask;

    bool operator<(const ComponentChange& other) const {
        if (type != other.type) return type < other.type;
        return name < other.name;
    }
};

struct DiffReport {
    bool isSemanticallyIdentical = true;
    bool isTopologyChanged = false;
    std::vector<ComponentChange> changes;

    void addChange(const std::string& name, const std::string& type, 
                   DiffChangeType changeType, const std::string& details, SimulationImpact impact) {
        changes.push_back({name, type, changeType, details, static_cast<uint32_t>(impact)});
        isSemanticallyIdentical = false;
    }
    
    void sort() {
        std::sort(changes.begin(), changes.end());
    }
};

class SemanticNetlistDiffer {
public:
    static DiffReport diff(TensorBlock& oldBlock, TensorBlock& newBlock) {
        DiffReport report;
        
        // 1. Topology Hash Check (Fast Path)
        oldBlock.computeTopologyHash();
        newBlock.computeTopologyHash();
        
        if (oldBlock.topologyHash != newBlock.topologyHash) {
            report.isTopologyChanged = true;
        }

        // ---- Passives (keyed by .name) ----
        diffByName<Resistor>(oldBlock.resistors, newBlock.resistors, "Resistor", report,
            [](const Resistor& a, const Resistor& b) {
                if (std::abs(a.resistance_ohms - b.resistance_ohms) > 1e-9)
                    return "R: " + std::to_string(a.resistance_ohms) + " -> " + std::to_string(b.resistance_ohms);
                return std::string("");
            },
            [](const Resistor& r) -> std::pair<int, int> { return {r.nodeTerminal1, r.nodeTerminal2}; }
        );

        diffByName<Capacitor>(oldBlock.capacitors, newBlock.capacitors, "Capacitor", report,
            [](const Capacitor& a, const Capacitor& b) {
                if (std::abs(a.capacitance_farads - b.capacitance_farads) > 1e-15)
                    return "C: " + std::to_string(a.capacitance_farads) + " -> " + std::to_string(b.capacitance_farads);
                if (a.icGiven != b.icGiven || a.ic_V != b.ic_V)
                    return "IC: " + std::to_string(a.ic_V) + " -> " + std::to_string(b.ic_V);
                return std::string("");
            },
            [](const Capacitor& c) -> std::pair<int, int> { return {c.nodePlate1, c.nodePlate2}; }
        );

        diffByName<Inductor>(oldBlock.inductors, newBlock.inductors, "Inductor", report,
            [](const Inductor& a, const Inductor& b) {
                if (std::abs(a.inductance_henries - b.inductance_henries) > 1e-15)
                    return "L: " + std::to_string(a.inductance_henries) + " -> " + std::to_string(b.inductance_henries);
                if (a.icGiven != b.icGiven || a.ic_A != b.ic_A)
                    return "IC: " + std::to_string(a.ic_A) + " -> " + std::to_string(b.ic_A);
                return std::string("");
            },
            [](const Inductor& l) -> std::pair<int, int> { return {l.nodeCoil1, l.nodeCoil2}; }
        );

        // ---- Semiconductors (keyed by .instanceName) ----
        diffByInstanceName<Mosfet>(oldBlock.mosfets, newBlock.mosfets, "MOSFET", report,
            [](const Mosfet& a, const Mosfet& b) {
                std::string details;
                if (std::abs(a.w - b.w) > 1e-12) details += "W: " + std::to_string(a.w) + " -> " + std::to_string(b.w) + "; ";
                if (std::abs(a.l - b.l) > 1e-12) details += "L: " + std::to_string(a.l) + " -> " + std::to_string(b.l) + "; ";
                if (a.modelName != b.modelName) details += "model: " + a.modelName + " -> " + b.modelName + "; ";
                return details;
            },
            [](const Mosfet& m) { return std::vector<int>{m.drain, m.gate, m.source, m.body}; }
        );

        diffByInstanceName<BJT>(oldBlock.bjts, newBlock.bjts, "BJT", report,
            [](const BJT& a, const BJT& b) {
                std::string details;
                if (std::abs(a.betaF - b.betaF) > 1e-6) details += "betaF: " + std::to_string(a.betaF) + " -> " + std::to_string(b.betaF) + "; ";
                if (std::abs(a.saturationCurrent_I_S_A - b.saturationCurrent_I_S_A) > 1e-20)
                    details += "Is: " + std::to_string(a.saturationCurrent_I_S_A) + " -> " + std::to_string(b.saturationCurrent_I_S_A) + "; ";
                if (a.isNPN != b.isNPN) details += "polarity changed; ";
                if (a.modelName != b.modelName) details += "model: " + a.modelName + " -> " + b.modelName + "; ";
                if (a.area != b.area) details += "area: " + std::to_string(a.area) + " -> " + std::to_string(b.area) + "; ";
                return details;
            },
            [](const BJT& b) { return std::vector<int>{b.nodeCollector, b.base, b.emitter, b.substrate}; }
        );

        diffByInstanceName<JFET>(oldBlock.jfets, newBlock.jfets, "JFET", report,
            [](const JFET& a, const JFET& b) {
                std::string details;
                if (std::abs(a.beta - b.beta) > 1e-9) details += "beta: " + std::to_string(a.beta) + " -> " + std::to_string(b.beta) + "; ";
                if (std::abs(a.Vto - b.Vto) > 1e-6) details += "Vto: " + std::to_string(a.Vto) + " -> " + std::to_string(b.Vto) + "; ";
                if (a.isNChannel != b.isNChannel) details += "polarity changed; ";
                return details;
            },
            [](const JFET& j) { return std::vector<int>{j.drain, j.gate, j.source}; }
        );

        diffByInstanceName<OsdiDevice>(oldBlock.osdiDevices, newBlock.osdiDevices, "OSDI", report,
            [](const OsdiDevice& a, const OsdiDevice& b) {
                std::string details;
                if (a.modelName != b.modelName) details += "model: " + a.modelName + " -> " + b.modelName + "; ";
                if (a.params != b.params) details += "instance parameters changed; ";
                return details;
            },
            [](const OsdiDevice& o) { return o.nodes; }
        );

        // ---- Anonymous components (keyed by index) ----
        diffByIndex<Diode>(oldBlock.diodes, newBlock.diodes, "Diode", report,
            [](const Diode& a, const Diode& b) {
                std::string details;
                if (std::abs(a.saturationCurrent_I_S_A - b.saturationCurrent_I_S_A) > 1e-20)
                    details += "Is: " + std::to_string(a.saturationCurrent_I_S_A) + " -> " + std::to_string(b.saturationCurrent_I_S_A) + "; ";
                if (std::abs(a.emissionCoefficient_N - b.emissionCoefficient_N) > 1e-6)
                    details += "N: " + std::to_string(a.emissionCoefficient_N) + " -> " + std::to_string(b.emissionCoefficient_N) + "; ";
                if (a.modelName != b.modelName) details += "model: " + a.modelName + " -> " + b.modelName + "; ";
                if (a.area != b.area) details += "area: " + std::to_string(a.area) + " -> " + std::to_string(b.area) + "; ";
                return details;
            },
            [](const Diode& d) -> std::pair<int, int> { return {d.anode, d.cathode}; }
        );

        diffByIndex<VoltageSource>(oldBlock.voltageSources, newBlock.voltageSources, "VoltageSource", report,
            [](const VoltageSource& a, const VoltageSource& b) {
                if (std::abs(a.voltage_V - b.voltage_V) > 1e-9)
                    return "V: " + std::to_string(a.voltage_V) + " -> " + std::to_string(b.voltage_V);
                return std::string("");
            },
            [](const VoltageSource& v) -> std::pair<int, int> { return {v.nodePositive, v.nodeNegative}; }
        );

        diffByIndex<CurrentSource>(oldBlock.currentSources, newBlock.currentSources, "CurrentSource", report,
            [](const CurrentSource& a, const CurrentSource& b) {
                if (std::abs(a.current_A - b.current_A) > 1e-15)
                    return "I: " + std::to_string(a.current_A) + " -> " + std::to_string(b.current_A);
                return std::string("");
            },
            [](const CurrentSource& c) -> std::pair<int, int> { return {c.nodePositive, c.nodeNegative}; }
        );

        diffByIndex<TransmissionLine>(oldBlock.transmissionLines, newBlock.transmissionLines, "TransmissionLine", report,
            [](const TransmissionLine& a, const TransmissionLine& b) {
                std::string details;
                if (std::abs(a.characteristicImpedance_Z0_ohms - b.characteristicImpedance_Z0_ohms) > 1e-6)
                    details += "Z0: " + std::to_string(a.characteristicImpedance_Z0_ohms) + " -> " + std::to_string(b.characteristicImpedance_Z0_ohms) + "; ";
                if (std::abs(a.propagationDelay_Td_s - b.propagationDelay_Td_s) > 1e-15)
                    details += "Td: " + std::to_string(a.propagationDelay_Td_s) + " -> " + std::to_string(b.propagationDelay_Td_s) + "; ";
                return details;
            },
            [](const TransmissionLine& t) -> std::pair<int, int> { return {t.nodePort1Pos, t.nodePort2Pos}; }
        );

        // ---- Dependent Sources (keyed by index) ----
        diffByIndex<VCVS>(oldBlock.voltageControlledVoltageSources, newBlock.voltageControlledVoltageSources, "VCVS", report,
            [](const VCVS& a, const VCVS& b) {
                if (std::abs(a.gain - b.gain) > 1e-9)
                    return "gain: " + std::to_string(a.gain) + " -> " + std::to_string(b.gain);
                return std::string("");
            },
            [](const VCVS& v) -> std::pair<int, int> { return {v.outPos, v.ctrlPos}; }
        );

        diffByIndex<VCCS>(oldBlock.voltageControlledCurrentSources, newBlock.voltageControlledCurrentSources, "VCCS", report,
            [](const VCCS& a, const VCCS& b) {
                if (std::abs(a.gm - b.gm) > 1e-9)
                    return "gm: " + std::to_string(a.gm) + " -> " + std::to_string(b.gm);
                return std::string("");
            },
            [](const VCCS& v) -> std::pair<int, int> { return {v.outPos, v.ctrlPos}; }
        );

        diffByIndex<CCVS>(oldBlock.currentControlledVoltageSources, newBlock.currentControlledVoltageSources, "CCVS", report,
            [](const CCVS& a, const CCVS& b) {
                if (std::abs(a.rm - b.rm) > 1e-9)
                    return "rm: " + std::to_string(a.rm) + " -> " + std::to_string(b.rm);
                return std::string("");
            },
            [](const CCVS& v) -> std::pair<int, int> { return {v.outPos, v.ctrlPos}; }
        );

        diffByIndex<CCCS>(oldBlock.currentControlledCurrentSources, newBlock.currentControlledCurrentSources, "CCCS", report,
            [](const CCCS& a, const CCCS& b) {
                if (std::abs(a.alpha - b.alpha) > 1e-9)
                    return "alpha: " + std::to_string(a.alpha) + " -> " + std::to_string(b.alpha);
                return std::string("");
            },
            [](const CCCS& v) -> std::pair<int, int> { return {v.outPos, v.ctrlPos}; }
        );

        report.sort();
        return report;
    }

private:
    // ========================================================================
    // Diff by .name field (Resistor, Capacitor, Inductor)
    // Uses 2-terminal node extraction (std::pair<int,int>).
    // ========================================================================
    template<typename T, typename ParamCheckFunc, typename NodeExtractFunc>
    static void diffByName(const std::vector<T>& oldList, 
                           const std::vector<T>& newList,
                           const std::string& typeName,
                           DiffReport& report,
                           ParamCheckFunc checkParams,
                           NodeExtractFunc getNodes) 
    {
        std::map<std::string, const T*> oldMap;
        for (const auto& item : oldList) oldMap[item.name] = &item;

        std::map<std::string, const T*> newMap;
        for (const auto& item : newList) newMap[item.name] = &item;

        for (const auto& kv : oldMap) {
            if (newMap.find(kv.first) == newMap.end()) {
                report.addChange(kv.first, typeName, DiffChangeType::REMOVED, "Component deleted", 
                                 SimulationImpact::MATRIX_REBUILD | SimulationImpact::STATE_RESET);
            }
        }

        for (const auto& kv : newMap) {
            const std::string& name = kv.first;
            const T* newItem = kv.second;

            if (oldMap.find(name) == oldMap.end()) {
                report.addChange(name, typeName, DiffChangeType::ADDED, "Component added", 
                                 SimulationImpact::MATRIX_REBUILD | SimulationImpact::STATE_RESET);
            } else {
                const T* oldItem = oldMap.at(name);
                auto nodesOld = getNodes(*oldItem);
                auto nodesNew = getNodes(*newItem);
                
                if (nodesOld.first != nodesNew.first || nodesOld.second != nodesNew.second) {
                   report.addChange(name, typeName, DiffChangeType::CONNECTIVITY_CHANGED, 
                                    "Nodes changed", SimulationImpact::MATRIX_REBUILD);
                } else {
                    std::string details = checkParams(*oldItem, *newItem);
                    if (!details.empty()) {
                        report.addChange(name, typeName, DiffChangeType::MODIFIED_PARAMETER, 
                                         details, SimulationImpact::JACOBIAN_UPDATE);
                    }
                }
            }
        }
    }

    // ========================================================================
    // Diff by .instanceName field (MOSFET, BJT, JFET)
    // Uses N-terminal node extraction (std::vector<int>).
    // ========================================================================
    template<typename T, typename ParamCheckFunc, typename NodeExtractFunc>
    static void diffByInstanceName(const std::vector<T>& oldList, 
                                   const std::vector<T>& newList,
                                   const std::string& typeName,
                                   DiffReport& report,
                                   ParamCheckFunc checkParams,
                                   NodeExtractFunc getNodes) 
    {
        std::map<std::string, const T*> oldMap;
        for (const auto& item : oldList) oldMap[item.instanceName] = &item;

        std::map<std::string, const T*> newMap;
        for (const auto& item : newList) newMap[item.instanceName] = &item;

        for (const auto& kv : oldMap) {
            if (newMap.find(kv.first) == newMap.end()) {
                report.addChange(kv.first, typeName, DiffChangeType::REMOVED, "Component deleted", 
                                 SimulationImpact::MATRIX_REBUILD | SimulationImpact::STATE_RESET);
            }
        }

        for (const auto& kv : newMap) {
            const std::string& name = kv.first;
            const T* newItem = kv.second;

            if (oldMap.find(name) == oldMap.end()) {
                report.addChange(name, typeName, DiffChangeType::ADDED, "Component added", 
                                 SimulationImpact::MATRIX_REBUILD | SimulationImpact::STATE_RESET);
            } else {
                const T* oldItem = oldMap.at(name);
                auto nodesOld = getNodes(*oldItem);
                auto nodesNew = getNodes(*newItem);
                
                if (nodesOld != nodesNew) {
                   report.addChange(name, typeName, DiffChangeType::CONNECTIVITY_CHANGED, 
                                    "Nodes changed", SimulationImpact::MATRIX_REBUILD);
                } else {
                    std::string details = checkParams(*oldItem, *newItem);
                    if (!details.empty()) {
                        report.addChange(name, typeName, DiffChangeType::MODIFIED_PARAMETER, 
                                         details, SimulationImpact::JACOBIAN_UPDATE);
                    }
                }
            }
        }
    }

    // ========================================================================
    // Diff by index (Diode, VoltageSource, CurrentSource, dependent sources,
    // TransmissionLine) — types without a unique name field.
    // Compares element-by-element; size changes reported as bulk ADDED/REMOVED.
    // ========================================================================
    template<typename T, typename ParamCheckFunc, typename NodeExtractFunc>
    static void diffByIndex(const std::vector<T>& oldList, 
                            const std::vector<T>& newList,
                            const std::string& typeName,
                            DiffReport& report,
                            ParamCheckFunc checkParams,
                            NodeExtractFunc getNodes) 
    {
        size_t common = std::min(oldList.size(), newList.size());

        for (size_t i = 0; i < common; ++i) {
            std::string label = typeName + "[" + std::to_string(i) + "]";
            auto nodesOld = getNodes(oldList[i]);
            auto nodesNew = getNodes(newList[i]);

            if (nodesOld.first != nodesNew.first || nodesOld.second != nodesNew.second) {
                report.addChange(label, typeName, DiffChangeType::CONNECTIVITY_CHANGED, 
                                 "Nodes changed", SimulationImpact::MATRIX_REBUILD);
            } else {
                std::string details = checkParams(oldList[i], newList[i]);
                if (!details.empty()) {
                    report.addChange(label, typeName, DiffChangeType::MODIFIED_PARAMETER, 
                                     details, SimulationImpact::JACOBIAN_UPDATE);
                }
            }
        }

        // Report added elements
        for (size_t i = common; i < newList.size(); ++i) {
            std::string label = typeName + "[" + std::to_string(i) + "]";
            report.addChange(label, typeName, DiffChangeType::ADDED, "Component added", 
                             SimulationImpact::MATRIX_REBUILD | SimulationImpact::STATE_RESET);
        }

        // Report removed elements
        for (size_t i = common; i < oldList.size(); ++i) {
            std::string label = typeName + "[" + std::to_string(i) + "]";
            report.addChange(label, typeName, DiffChangeType::REMOVED, "Component deleted", 
                             SimulationImpact::MATRIX_REBUILD | SimulationImpact::STATE_RESET);
        }
    }
};
