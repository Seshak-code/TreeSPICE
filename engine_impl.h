#pragma once
// ============================================================================
// engine_impl.h — Concrete Engine Implementation (INTERNAL)
// ============================================================================
// This header is NOT part of engine_api/ — it is an internal implementation
// detail hidden behind -fvisibility=hidden.
//
// Three-layer stack:
//   ISimulationEngine (virtual)   ← engine_api boundary
//            ↓
//   EngineImpl (non-virtual)       ← this file
//            ↓
//   inline / templated kernels    ← hot path, ZERO vtable call overhead
//
// SessionImpl (also non-virtual internally) implements SimulationSession
// and calls CircuitSim directly — no virtual dispatch in NR loop.
// ============================================================================

#include "engine_export.h"
#include "gpu_context_manager.h"
#include "engine_api/isimulation_engine.h"
#include "engine_api/simulation_session.h"

#include <memory>
#include <atomic>

namespace acutesim {

using namespace compute::orchestration;

// ── SessionImpl ────────────────────────────────────────────────────────────
// Internal, non-virtual session. All solver calls are compile-time bound.
// (CircuitSim, WebGPUSolver, etc. are included in engine_impl.cpp — not here)
class ENGINE_INTERNAL SessionImpl final : public SimulationSession {
public:
    explicit SessionImpl(GPUContextManager& gpu, PrecisionMode prec);
    ~SessionImpl() override;

    // Outer virtual dispatch (called once per simulation — acceptable cost)
    SimulationResponseDTO runDC(const SimulationRequestDTO&,
                                  const EngineCallbacks&) override;
    SimulationResponseDTO runAC(const SimulationRequestDTO&,
                                  const EngineCallbacks&) override;
    SimulationResponseDTO runTransient(const SimulationRequestDTO&,
                                        const EngineCallbacks&,
                                        ResultChunkCallback) override;
    SimulationResponseDTO runMonteCarlo(const SimulationRequestDTO&,
                                         const MonteCarloConfigDTO&,
                                         const EngineCallbacks&) override;
    SimulationResponseDTO runNoise(const SimulationRequestDTO&,
                                    const EngineCallbacks&) override;
    SimulationResponseDTO runPoleZero(const SimulationRequestDTO&,
                                       const EngineCallbacks&) override;
    SimulationResponseDTO runSens(const SimulationRequestDTO&,
                                   const EngineCallbacks&) override;
    std::unique_ptr<TransientHandle> beginTransient(const SimulationRequestDTO&) override;

    JobID     jobId()  const override { return jobId_; }
    JobStatus status() const override { return status_.load(); }
    void      cancel()       override;

    void setIncrementalHints(uint64_t topologyHash,
                              uint64_t parameterHash) override;

private:
    GPUContextManager&       gpu_;
    PrecisionMode            precision_;
    JobID                    jobId_;
    std::atomic<JobStatus>   status_{ JobStatus::Queued };
    uint64_t                 topologyHash_ = 0;
    uint64_t                 paramHash_    = 0;

    // Non-virtual hot-path helpers (defined in engine_impl.cpp)
    // These call CircuitSim / WebGPUSolver directly — ZERO vtable overhead
    SimulationResponseDTO dispatch_dc_internal(const SimulationRequestDTO&,
                                                 const EngineCallbacks&);
    SimulationResponseDTO dispatch_ac_internal(const SimulationRequestDTO&,
                                                 const EngineCallbacks&);
    SimulationResponseDTO dispatch_tran_internal(const SimulationRequestDTO&,
                                                   const EngineCallbacks&,
                                                   ResultChunkCallback);
};

// ── EngineImpl ─────────────────────────────────────────────────────────────
class ENGINE_INTERNAL EngineImpl final : public ISimulationEngine {
public:
    EngineImpl();
    ~EngineImpl() override;

    std::unique_ptr<SimulationSession> createSession() override;

    EngineCapabilities capabilities() const override;
    void setPrecisionMode(PrecisionMode mode) override;
    PrecisionMode precisionMode() const override;

private:
    PrecisionMode      precision_    = PrecisionMode::Mixed;
    EngineCapabilities capabilities_;
    GPUContextManager& gpu_;

    void detectCapabilities();
};

} // namespace acutesim
