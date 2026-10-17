// .meas / .measure (ngspice chapter 11.4): TRIG/TARG, FIND/DERIV AT|WHEN, WHEN, AVG|RMS|MIN|MAX|PP|INTEG|MAX_AT|MIN_AT
// with FROM/TO, and PARAM over earlier measures, over a finished tran, dc or ac response.
#pragma once
#include "simulation_dto.h"

#include <iosfwd>

#ifndef ENGINE_API
  #define ENGINE_API
#endif

namespace acutesim {

// Evaluate the response's netlist .meas cards (kind 'm') that apply to its analysis into r.measures: one set per
// part of a nested .dc / .temp / .step run (MeasureResultDTO::step is the part's index, as splitByOuter orders them).
ENGINE_API void attachMeasures(compute::orchestration::SimulationResponseDTO& r);

// ngspice's text for the measures of one part: "name = value" lines, or the failure message.
ENGINE_API void writeMeasures(std::ostream& os, const compute::orchestration::SimulationResponseDTO& r, int step);

}  // namespace acutesim
