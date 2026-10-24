#pragma once

#include "circuit.h"
#include <vector>
#include <string>
#include <set>

/**
 * subcircuit_validator.h
 * 
 * Validation utilities for SubCircuitDefinition to ensure extracted
 * subcircuits are "simulation ready" before saving to the component library.
 * 
 * This supports the hierarchical design flow where users create custom "chips"
 * from schematic regions.
 */

// ============================================================================
// VALIDATION RESULT
// ============================================================================

struct ValidationIssue {
    enum class Severity { ERROR, WARNING, INFO };
    
    Severity severity;
    std::string code;          // e.g., "FLOATING_NODE", "MISSING_GROUND"
    std::string message;       // Human-readable description
    int nodeOrPinIndex = -1;   // Related node/pin if applicable
};

struct ValidationResult {
    bool isValid = true;       // True if no errors (warnings allowed)
    std::vector<ValidationIssue> issues;
    
    bool hasErrors() const {
        for (const auto& issue : issues) {
            if (issue.severity == ValidationIssue::Severity::ERROR) return true;
        }
        return false;
    }
    
    bool hasWarnings() const {
        for (const auto& issue : issues) {
            if (issue.severity == ValidationIssue::Severity::WARNING) return true;
        }
        return false;
    }
    
    std::string summary() const {
        int errors = 0, warnings = 0;
        for (const auto& issue : issues) {
            if (issue.severity == ValidationIssue::Severity::ERROR) errors++;
            else if (issue.severity == ValidationIssue::Severity::WARNING) warnings++;
        }
        return std::to_string(errors) + " error(s), " + std::to_string(warnings) + " warning(s)";
    }
};

// ============================================================================
// VALIDATION FUNCTION
// ============================================================================

/**
 * validateSubCircuit
 * 
 * Validates a SubCircuitDefinition for simulation readiness.
 * 
 * Checks performed:
 * 1. Floating Node Detection - Internal nodes not connected to any component
 * 2. Ground Reference - At least one pin/node maps to ground (node 0)
 * 3. Parameter Completeness - Device parameters are valid (Is > 0, R > 0, etc.)
 * 4. Pin Connectivity - All declared pins connect to at least one component
 * 5. PDK Readiness (Warnings) - Flag missing optional PDK parameters
 */
inline ValidationResult validateSubCircuit(const SubCircuitDefinition& def) {
    ValidationResult result;
    result.isValid = true;
    
    const auto& block = def.internalBlock;
    
    // Collect all nodes referenced by components
    std::set<int> connectedNodes;
    
    // Resistors
    for (const auto& r : block.resistors) {
        connectedNodes.insert(r.nodeTerminal1);
        connectedNodes.insert(r.nodeTerminal2);
        
        // Parameter check
        if (r.resistance_ohms <= 0.0) {
            result.issues.push_back({
                ValidationIssue::Severity::ERROR,
                "INVALID_RESISTANCE",
                "Resistor '" + r.name + "' has non-positive resistance: " + std::to_string(r.resistance_ohms),
                -1
            });
            result.isValid = false;
        }
    }
    
    // Capacitors
    for (const auto& c : block.capacitors) {
        connectedNodes.insert(c.nodePlate1);
        connectedNodes.insert(c.nodePlate2);
        
        if (c.capacitance_farads <= 0.0) {
            result.issues.push_back({
                ValidationIssue::Severity::ERROR,
                "INVALID_CAPACITANCE",
                "Capacitor has non-positive capacitance: " + std::to_string(c.capacitance_farads),
                -1
            });
            result.isValid = false;
        }
    }
    
    // Inductors
    for (const auto& l : block.inductors) {
        connectedNodes.insert(l.nodeCoil1);
        connectedNodes.insert(l.nodeCoil2);
        
        if (l.inductance_henries <= 0.0) {
            result.issues.push_back({
                ValidationIssue::Severity::ERROR,
                "INVALID_INDUCTANCE",
                "Inductor '" + l.name + "' has non-positive inductance",
                -1
            });
            result.isValid = false;
        }
    }
    
    // Voltage Sources
    for (const auto& vs : block.voltageSources) {
        connectedNodes.insert(vs.nodePositive);
        connectedNodes.insert(vs.nodeNegative);
    }
    
    // Current Sources
    for (const auto& cs : block.currentSources) {
        connectedNodes.insert(cs.nodePositive);
        connectedNodes.insert(cs.nodeNegative);
    }
    
    // Diodes
    for (const auto& d : block.diodes) {
        connectedNodes.insert(d.anode);
        connectedNodes.insert(d.cathode);
        if (d.thermal > 0) connectedNodes.insert(d.thermal);
        
        if (d.saturationCurrent_I_S_A <= 0.0) {
            result.issues.push_back({
                ValidationIssue::Severity::ERROR,
                "INVALID_DIODE_IS",
                "Diode (anode=" + std::to_string(d.anode) + ", cathode=" + std::to_string(d.cathode) + ") has non-positive saturation current",
                -1
            });
            result.isValid = false;
        }
        
        // PDK Readiness warning for missing parasitics
        // Future: Check for junction capacitance, series resistance, etc.
        result.issues.push_back({
            ValidationIssue::Severity::INFO,
            "PDK_PARASITIC_MISSING",
            "Diode (anode=" + std::to_string(d.anode) + ") lacks parasitic parameters (Cj, Rs) for advanced PDK compatibility",
            -1
        });
    }
    
    // MOSFETs
    for (const auto& m : block.mosfets) {
        connectedNodes.insert(m.drain);
        connectedNodes.insert(m.gate);
        connectedNodes.insert(m.source);
        connectedNodes.insert(m.body);
        
        if (m.w <= 0.0 || m.l <= 0.0) {
            result.issues.push_back({
                ValidationIssue::Severity::ERROR,
                "INVALID_MOSFET_GEOMETRY",
                "MOSFET '" + m.instanceName + "' has invalid W/L dimensions",
                -1
            });
            result.isValid = false;
        }
        
        // PDK Readiness: Check for tunneling, parasitic capacitances
        result.issues.push_back({
            ValidationIssue::Severity::INFO,
            "PDK_ADVANCED_MISSING",
            "MOSFET '" + m.instanceName + "' lacks advanced PDK parameters (tunneling, overlap caps) for sub-28nm nodes",
            -1
        });
    }
    
    // BJTs
    for (const auto& q : block.bjts) {
        connectedNodes.insert(q.nodeCollector);
        connectedNodes.insert(q.base);
        connectedNodes.insert(q.emitter);
        
        if (q.saturationCurrent_I_S_A <= 0.0) {
            result.issues.push_back({
                ValidationIssue::Severity::ERROR,
                "INVALID_BJT_IS",
                "BJT '" + q.instanceName + "' has non-positive saturation current",
                -1
            });
            result.isValid = false;
        }
        
        if (q.betaF <= 0.0) {
            result.issues.push_back({
                ValidationIssue::Severity::ERROR,
                "INVALID_BJT_BETA",
                "BJT '" + q.instanceName + "' has non-positive forward beta",
                -1
            });
            result.isValid = false;
        }
    }
    
    // Check 1: Ground Reference
    bool hasGround = connectedNodes.count(0) > 0;
    for (const auto& [pinIdx, internalNode] : def.pinToInternalNode) {
        if (internalNode == 0) hasGround = true;
    }
    
    if (!hasGround) {
        result.issues.push_back({
            ValidationIssue::Severity::ERROR,
            "MISSING_GROUND",
            "Subcircuit has no connection to ground (node 0). At least one pin or internal node must reference ground.",
            -1
        });
        result.isValid = false;
    }
    
    // Check 2: Pin Connectivity
    for (int pinIdx = 0; pinIdx < def.pinCount; ++pinIdx) {
        auto it = def.pinToInternalNode.find(pinIdx);
        if (it == def.pinToInternalNode.end()) {
            result.issues.push_back({
                ValidationIssue::Severity::ERROR,
                "UNMAPPED_PIN",
                "Pin " + std::to_string(pinIdx) + " (" + 
                    (pinIdx < (int)def.pinNames.size() ? def.pinNames[pinIdx] : "unnamed") + 
                    ") has no internal node mapping",
                pinIdx
            });
            result.isValid = false;
        } else {
            int internalNode = it->second;
            if (connectedNodes.count(internalNode) == 0 && internalNode != 0) {
                result.issues.push_back({
                    ValidationIssue::Severity::WARNING,
                    "FLOATING_PIN",
                    "Pin " + std::to_string(pinIdx) + " (" + 
                        (pinIdx < (int)def.pinNames.size() ? def.pinNames[pinIdx] : "unnamed") + 
                        ") maps to internal node " + std::to_string(internalNode) + 
                        " which has no component connections",
                    pinIdx
                });
            }
        }
    }
    
    // Check 3: Empty subcircuit
    if (block.resistors.empty() && block.capacitors.empty() && block.inductors.empty() &&
        block.diodes.empty() && block.mosfets.empty() && block.bjts.empty() &&
        block.voltageSources.empty() && block.currentSources.empty()) {
        result.issues.push_back({
            ValidationIssue::Severity::ERROR,
            "EMPTY_SUBCIRCUIT",
            "Subcircuit contains no components",
            -1
        });
        result.isValid = false;
    }
    
    return result;
}

// ============================================================================
// HELPER: Format validation result for display
// ============================================================================

inline std::string formatValidationResult(const ValidationResult& result) {
    std::string output;
    
    for (const auto& issue : result.issues) {
        std::string prefix;
        switch (issue.severity) {
            case ValidationIssue::Severity::ERROR:   prefix = "[ERROR] "; break;
            case ValidationIssue::Severity::WARNING: prefix = "[WARN]  "; break;
            case ValidationIssue::Severity::INFO:    prefix = "[INFO]  "; break;
        }
        output += prefix + issue.code + ": " + issue.message + "\n";
    }
    
    if (result.isValid) {
        output += "\n✓ Subcircuit is valid for simulation.\n";
    } else {
        output += "\n✗ Subcircuit has errors that must be fixed before saving.\n";
    }
    
    return output;
}
