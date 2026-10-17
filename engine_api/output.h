// Output selection and formatting for a finished analysis: the netlist's .print/.save/.probe
// expressions (v(a), v(a,b)*2, i(v1), vdb(out), db(), mag(), ph(), real(), imag()), the ngspice
// .print text table, and ngspice raw files (ascii and binary) that ngspice, gwave, Xschem and
// KLayout read.
#pragma once
#include "simulation_dto.h"

#include <complex>
#include <iosfwd>
#include <string>
#include <vector>

#ifndef ENGINE_API
  #define ENGINE_API
#endif

namespace acutesim {

struct OutputTable {
    std::string plotName;   // "Transient Analysis", "AC Analysis", "DC transfer characteristic", "Operating Point"
    std::string scaleName;  // "time", "frequency", "v-sweep", "i-sweep"; "" for .op
    bool complexData = false;
    std::vector<double> scale;
    struct Column { std::string name, type; std::vector<std::complex<double>> values; bool isComplex = false; };  // isComplex: the .print shows re,im
    std::vector<Column> columns;
};

// "tran", "ac", "dc" or "op".
ENGINE_API std::string analysisName(const compute::orchestration::SimulationResponseDTO& r);

// Evaluate `specs` over the response; an empty list (or "all") selects every node voltage and branch current.
// Throws std::runtime_error on an unknown node or source, or on a device current (not supported yet).
ENGINE_API OutputTable selectOutputs(const compute::orchestration::SimulationResponseDTO& r,
                                     const std::vector<std::string>& specs);

// A nested .dc / .temp run as one response per value of the outer sweeps (for the .print tables); any other
// response comes back whole.
ENGINE_API std::vector<compute::orchestration::SimulationResponseDTO> splitByOuter(
    const compute::orchestration::SimulationResponseDTO& r);

// The specs of the response's netlist cards of `kind` ('p' print, 's' save/probe) that apply to its analysis.
ENGINE_API std::vector<std::string> requestedSpecs(const compute::orchestration::SimulationResponseDTO& r, char kind);

// ngspice's .print layout (Index, scale, columns; a complex value prints as "re,<tab>im").
ENGINE_API void writeTable(std::ostream& os, const OutputTable& t, const std::string& title);

// One plot of an ngspice raw file; call again to append another plot to the same stream.
ENGINE_API void writeRaw(std::ostream& os, const OutputTable& t, const std::string& title, bool ascii);

}  // namespace acutesim
