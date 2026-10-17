#pragma once
// ============================================================================
// simulation_session.h — Isolated Simulation Execution Object
// ============================================================================
// Each session represents one independent simulation context.
//
// Design rationale:
//   - Avoids global engine state (which blocks concurrency)
//   - AI speculative runs get their own session → cancel without side effects
//   - GPU contexts are acquired from GPUContextManager pool per session
//   - Multiple sessions can run concurrently (limited by maxParallelSessions)
//
// RULE: Do NOT add virtual methods to runDC_step, stepNewton, etc.
// Virtual dispatch in the NR inner loop destroys SIMD vectorization.
// Only the top-level run* methods are virtual (called once per simulation).
// ============================================================================

#include "simulation_dto.h"
#include "engine_callbacks.h"
#include "job_status.h"

#include <memory>
#include <string>
#include <functional>
#include <cstdint>

namespace acutesim {

// ── Streaming result callback (for live waveform updates) ─────────────────
using ResultChunkCallback = std::function<void(const compute::orchestration::WaveformDataPoint&, uint64_t timeStepIdx)>;

// ── TransientHandle ────────────────────────────────────────────────────────
// A transient kept alive between calls, for a digital simulator that drives
// analog nodes in lockstep: setSource, advanceTo, voltage, repeat. Methods
// return false (and error() says why) after a failed begin or step.
class ENGINE_API TransientHandle {
public:
    virtual ~TransientHandle() = default;
    // Re-drive V (or I) source `name` from its value now to `volts` over `rampS` (0 = step)
    virtual bool setSource(const std::string& name, double volts, double rampS) = 0;
    virtual bool advanceTo(double t) = 0;
    virtual double time() const = 0;
    // Node voltage by netlist name (case-insensitive; "0"/"gnd" = ground)
    virtual bool voltage(const std::string& node, double& v) const = 0;
    virtual std::string error() const = 0;
};

// ── SimulationSession ──────────────────────────────────────────────────────
class ENGINE_API SimulationSession {
public:
    virtual ~SimulationSession() = default;

    // === Analysis methods ===
    virtual compute::orchestration::SimulationResponseDTO runDC(
        const compute::orchestration::SimulationRequestDTO&,
        const EngineCallbacks& = {}) = 0;

    virtual compute::orchestration::SimulationResponseDTO runAC(
        const compute::orchestration::SimulationRequestDTO&,
        const EngineCallbacks& = {}) = 0;

    virtual compute::orchestration::SimulationResponseDTO runTransient(
        const compute::orchestration::SimulationRequestDTO&,
        const EngineCallbacks& = {},
        ResultChunkCallback   streamCallback = nullptr) = 0;

    virtual compute::orchestration::SimulationResponseDTO runMonteCarlo(
        const compute::orchestration::SimulationRequestDTO&,
        const compute::orchestration::MonteCarloConfigDTO&,
        const EngineCallbacks& = {}) = 0;

    virtual compute::orchestration::SimulationResponseDTO runNoise(
        const compute::orchestration::SimulationRequestDTO&,
        const EngineCallbacks& = {}) = 0;

    // .pz: poles and/or zeros of the netlist's .pz card at the DC operating point (extended.pz).
    virtual compute::orchestration::SimulationResponseDTO runPoleZero(
        const compute::orchestration::SimulationRequestDTO&,
        const EngineCallbacks& = {}) = 0;

    // .sens: DC sensitivity of the netlist's .sens output to every supported element value (extended.sens).
    virtual compute::orchestration::SimulationResponseDTO runSens(
        const compute::orchestration::SimulationRequestDTO&,
        const EngineCallbacks& = {}) = 0;

    // Starts a transient at t = 0 (request.transient.stepSizeS = max step, uic) and
    // returns it for lockstep stepping; check error() for a failed start.
    virtual std::unique_ptr<TransientHandle> beginTransient(
        const compute::orchestration::SimulationRequestDTO&) = 0;

    // === Lifecycle ===
    virtual JobID       jobId()   const = 0;
    virtual JobStatus   status()  const = 0;
    virtual void        cancel()        = 0;

    // === Incremental reuse ===
    // Provide topology/param hashes from SchematicSimRequest for cache reuse.
    virtual void setIncrementalHints(uint64_t topologyHash,
                                      uint64_t parameterHash) = 0;
};

} // namespace acutesim
