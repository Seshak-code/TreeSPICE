#pragma once
#include "gui_boundary_guard.h"
#include "../netlist/circuit.h"
#include "../math/linalg.h"
#include "../math/contraction_solver.h"
#include "../math/worker_pool.h"
#include <mutex>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <functional>
#include <type_traits>
#include <vector>

/*
 * circuitsim.h
 * Circuit Simulator - Physics and Solver Engine.
 */

// ============================================================================
// STAMP FUNCTIONS (Component -> Matrix Entry)
// ============================================================================

#include "../autodiff/dual.h"
#include "device_physics.h"
#include "../osdi/osdi_device.h"
#include "physics_constants.h"
#include "../tensors/physics_tensors.h"
#include "stamps.h"

// ============================================================================
// CIRCUITSIM CLASS
// ============================================================================

#include "../infrastructure/simtrace.h"
#include "../infrastructure/tensorscheduler.h"
#include "corner_config.h"
#include "../solvers/neural_seeder.h"
#include "../solvers/lte_controller.h"

// ============================================================================
// CIRCUITSIM CLASS
// ============================================================================

#include "../math/integration_method.h"

/**
 * CircuitSim is the primary engine of the simulator.
 */

// ... (SimTrace/TensorScheduler includes)

// ... (Previous includes) ...

/*
 * Heuristic Arbiter for Adaptive Integration (Hardened).
 * Monitors energy stability, weighted ringing, and LTE.
 */

class IntegratorArbiter {
private:
  std::vector<double> prev_voltages;
  std::vector<double> prev_prev_voltages;
  std::vector<double> node_weights; 
  double prev_energy = 0.0;
  double prev_dt = 1e-9; 
  int instability_counter = 0;
  static constexpr int INSTABILITY_THRESHOLD = 3;

  // ─── TAA (Temporal Anti-Aliasing) History Rejection ──────────────────────
  // Per-node "motion vectors": tracks voltage acceleration (ddV/dt²) to
  // detect violent switching events where polynomial predictors diverge.
  // Analogy: TAA rejects history frames when motion vectors indicate rapid
  // scene changes. Here we reject polynomial extrapolations when ddV/dt²
  // indicates rapid switching, falling back to first-order Euler.
  std::vector<double> prev_dv_dt_;   // First derivative at each node (t-1)
  bool taaInitialized_ = false;
  // Counts accepted steps; LTE predictor needs ≥2 real data points to be valid.
  int stepsAccepted_ = 0;

  // Threshold for acceleration-based rejection. If |ddV/dt²| * dt² exceeds
  // this value (in volts), the polynomial predictor is untrustworthy.
  // Calibrated to reject at ~10mV of quadratic prediction error per step.
  static constexpr double TAA_ACCEL_THRESHOLD = 0.01;  // 10mV

public:
  // ─── TAA: History Rejection API ──────────────────────────────────────────
  // Computes per-node "motion vectors" (voltage acceleration) and rejects
  // polynomial predictor guesses for nodes undergoing violent switching.
  //
  // @param predicted     In/out: polynomial-predicted voltage state
  // @param current_v     Most recent converged voltage state (= prev_voltages)
  // @param dt            Current timestep
  // @return Number of nodes where history was rejected
  int rejectHistory(std::vector<double>& predicted, double dt) const {
      if (!taaInitialized_ || prev_voltages.empty() || prev_prev_voltages.empty() ||
          dt <= 0 || prev_dt <= 0) {
          return 0;
      }

      int rejectionCount = 0;
      const size_t N = std::min(predicted.size(), prev_voltages.size());
      const double dt_sq = dt * dt;

      for (size_t i = 0; i < N; ++i) {
          // Current first derivative estimate: dV/dt_{n-0.5}
          double dv_dt_current = (prev_voltages[i] - prev_prev_voltages[i]) / prev_dt;

          // Acceleration (second derivative): ddV/dt²
          double ddv_dt2 = 0.0;
          if (i < prev_dv_dt_.size()) {
              ddv_dt2 = (dv_dt_current - prev_dv_dt_[i]) / (0.5 * (dt + prev_dt));
          }

          // Quadratic prediction error estimate: |ddV/dt²| * dt²
          double accel_error = std::abs(ddv_dt2) * dt_sq;

          if (accel_error > TAA_ACCEL_THRESHOLD) {
              // Reject polynomial predictor → fall back to first-order Euler
              // predicted[i] = V[n-1] + dV/dt * dt  (linear extrapolation only)
              predicted[i] = prev_voltages[i] + dv_dt_current * dt;
              ++rejectionCount;
          }
      }
      return rejectionCount;
  }


  void initialize(const TensorNetlist &netlist) {
    int n = netlist.numGlobalNodes;
    prev_voltages.assign(n, 0.0);
    prev_prev_voltages.assign(n, 0.0);
    node_weights.assign(n, 1e-12);
    prev_dv_dt_.assign(n, 0.0);
    taaInitialized_ = false;
    stepsAccepted_ = 0;
    prev_energy = 0.0;
    prev_dt = 1e-9;
    instability_counter = 0;

    auto addWeight = [&](int node, double w) {
      if (node > 0 && node <= n)
        node_weights[node - 1] += std::abs(w);
    };

    for (const auto &r : netlist.globalBlock.resistors) {
      double g = 1.0 / r.resistance_ohms;
      addWeight(r.nodeTerminal1, g);
      addWeight(r.nodeTerminal2, g);
    }
    for (const auto &c : netlist.globalBlock.capacitors) {
      addWeight(c.nodePlate1, c.capacitance_farads);
      addWeight(c.nodePlate2, c.capacitance_farads);
    }
    
    // Add hierarchical weights
    for (const auto &inst : netlist.instances) {
        auto it = netlist.blockDefinitions.find(inst.blockName);
        if (it != netlist.blockDefinitions.end()) {
            const auto &block = it->second;
            auto mapNode = [&](int local) { return local == 0 ? 0 : inst.nodeMapping[local-1]; };
            for (const auto &c : block.capacitors) {
                addWeight(mapNode(c.nodePlate1), c.capacitance_farads);
                addWeight(mapNode(c.nodePlate2), c.capacitance_farads);
            }
        }
    }
  }

  const std::vector<double> &getPrevVoltages() const { return prev_voltages; }

  // Research-Grade LTE: Combined Node & Device Error
  double calculateLTE(const TensorNetlist &netlist, const std::vector<double>& current_voltages, double dt) {
      if (prev_voltages.empty() || prev_prev_voltages.empty() || dt <= 0 || prev_dt <= 0) return 0.0;
      // The second-order predictor 2*v(t) - v(t-h) requires at least 2 accepted
      // steps so that both prev_voltages and prev_prev_voltages contain real
      // simulation data.  Before that, the predictor is 0 (initialization) which
      // makes the LTE appear enormous for any non-trivial circuit — causing
      // unnecessary step-halving and premature GEAR_2 switches on cold starts.
      if (stepsAccepted_ < 2) return 0.0;
      
      double c_factor = dt / (dt + prev_dt);
      double sum_sq_error = 0.0;
      double max_err = 0.0;
      int count = 0;

      auto getNorm = [](double v) { return 1e-6 + std::abs(v) * 1e-3; };

      // 1. Nodal Voltages (State Variables)
      for (size_t i = 0; i < current_voltages.size(); ++i) {
          double v_pred = prev_voltages[i] + (prev_voltages[i] - prev_prev_voltages[i]) * (dt / prev_dt);
          double err = std::abs(current_voltages[i] - v_pred) * c_factor;
          double w_err = err / getNorm(current_voltages[i]);
          
          sum_sq_error += w_err * w_err;
          max_err = std::max(max_err, w_err);
          count++;
      }

      // 2. Capacitor Branch Voltages (Internal State), global + instances
      auto getVc = [&](int n) { return (n > 0 && n <= (int)current_voltages.size()) ? current_voltages[n-1] : 0.0; };
      auto capErr = [&](const TensorBlock &blk, const BlockState &st, auto mapNode) {
        for (size_t i = 0; i < blk.capacitors.size() && i < st.capacitorState.size(); ++i) {
          const auto &c = blk.capacitors[i];
          const auto &hist = st.capacitorState[i];
          if (hist.v.size() < 2) continue;
          double v_branch = getVc(mapNode(c.nodePlate1)) - getVc(mapNode(c.nodePlate2));
          double v_pred = hist.v[0] + (hist.v[0] - hist.v[1]) * (dt / prev_dt);
          double w_err = std::abs(v_branch - v_pred) * c_factor / getNorm(v_branch);
          sum_sq_error += w_err * w_err;
          max_err = std::max(max_err, w_err);
          count++;
        }
      };
      capErr(netlist.globalBlock, netlist.globalState, [](int n) { return n; });
      for (size_t k = 0; k < netlist.instances.size() && k < netlist.instanceStates.size(); ++k) {
        const auto &inst = netlist.instances[k];
        auto it = netlist.blockDefinitions.find(inst.blockName);
        if (it == netlist.blockDefinitions.end()) continue;
        capErr(it->second, netlist.instanceStates[k], [&](int n) {
          return (n <= 0 || n > (int)inst.nodeMapping.size()) ? 0 : inst.nodeMapping[n - 1];
        });
      }


      // 3. MOSFET Gate Drive (Vgs) - Critical for Switching
      for (const auto &m : netlist.globalBlock.mosfets) {
           auto getV = [&](int n, const std::vector<double>& src) { return (n>0 && n<=(int)src.size()) ? src[n-1] : 0.0; };
           
           double vgs = getV(m.gate, current_voltages) - getV(m.source, current_voltages);
           double vgs_p = getV(m.gate, prev_voltages) - getV(m.source, prev_voltages);
           double vgs_pp = getV(m.gate, prev_prev_voltages) - getV(m.source, prev_prev_voltages);
           
           double vgs_pred = vgs_p + (vgs_p - vgs_pp) * (dt / prev_dt);
           double err = std::abs(vgs - vgs_pred) * c_factor;
           double w_err = err / getNorm(vgs);
           
           sum_sq_error += w_err * w_err;
           max_err = std::max(max_err, w_err);
           count++;
      }

      if (count == 0) return 0.0;

      // Return Max-Norm (Conservative) effectively, but could switch to RMS
      // Standard SPICE uses RMS, but Max-Norm guarantees no single node deviates.
      // We return Max-Norm as it safely bounds RMS.
      return max_err;
  }

  // M4: Formal Passivity Checks (Static)
  bool checkPassivity(const TensorNetlist &netlist) {
      // 1. Check Global Block
      if (!checkBlockPassivity(netlist.globalBlock)) return false;
      // 2. Check Instances
      for (const auto &inst : netlist.instances) {
          auto it = netlist.blockDefinitions.find(inst.blockName);
          if (it != netlist.blockDefinitions.end()) {
              if (!checkBlockPassivity(it->second)) return false;
          }
      }
      return true;
  }

  bool checkBlockPassivity(const TensorBlock &block) {
      for (const auto &r : block.resistors) {
          if (r.resistance_ohms <= 0.0) return false;
      }
      for (const auto &c : block.capacitors) {
          if (c.capacitance_farads < 0.0) return false;
      }
      for (const auto &m : block.mosfets) {
          if (m.w <= 0.0 || m.l <= 0.0) return false;
      }
      return true;
  }

  // Returns: 0=Stable, 1=Reduce dt, 2=Switch Method
  int analyzeStability(const TensorNetlist &netlist,
                       const std::vector<double> &current_voltages,
                       double dt,
                       double &next_dt_suggestion) {
    if (current_voltages.empty()) return 0;
    if (current_voltages.size() != prev_voltages.size()) {
       initialize(netlist);
       next_dt_suggestion = dt; 
       return 0;
    }

    // 1. Check LTE
    double lte = calculateLTE(netlist, current_voltages, dt);
    // The predictor-corrector formula used in calculateLTE is the FIRST-ORDER
    // linear-extrapolation mismatch (≈ h²·v''/2), which overestimates the
    // actual TRAPEZOIDAL truncation error (≈ h³·v'''/12) by a factor of
    // ~12·τ/h.  For an RC circuit with dt/τ=0.22 this ratio is ~27×.
    // Setting the threshold at 30.0 means: the step is accepted when the
    // predictor mismatch is < 30×getNorm(v), which corresponds to roughly
    // 0.6% relative voltage error — consistent with standard SPICE LTE targets.
    // Real instabilities (NR oscillations, diverging active circuits) produce
    // LTE values well above 100 and are still caught by this guard.
    bool high_lte = (lte > 35.0); // Threshold

    // 2. Check Energy
    double current_energy = 0.0;
    for (const auto &c : netlist.globalBlock.capacitors) {
        double v = 0.0;
        if (c.nodePlate1 > 0 && c.nodePlate1 <= (int)current_voltages.size()) v += current_voltages[c.nodePlate1 - 1];
        if (c.nodePlate2 > 0 && c.nodePlate2 <= (int)current_voltages.size()) v -= current_voltages[c.nodePlate2 - 1];
        current_energy += 0.5 * c.capacitance_farads * v * v;
    }
    
    // Add instance energy
    for (const auto &inst : netlist.instances) {
        auto it = netlist.blockDefinitions.find(inst.blockName);
        if (it != netlist.blockDefinitions.end()) {
            const auto &block = it->second;
            auto mapNode = [&](int local) { return local == 0 ? 0 : inst.nodeMapping[local-1]; };
            auto getV = [&](int n) { return (n > 0 && n <= (int)current_voltages.size()) ? current_voltages[n-1] : 0.0; };
            for (const auto &c : block.capacitors) {
                 double v = getV(mapNode(c.nodePlate1)) - getV(mapNode(c.nodePlate2));
                 current_energy += 0.5 * c.capacitance_farads * v * v;
            }
        }
    }



    // M4: Lyapunov Energy Monitor
    // Calculate Work Done by Sources (Approximate Pin)
    double power_in = 0.0;
    // Current Sources
    for (const auto &i : netlist.globalBlock.currentSources) {
         double v_diff = 0.0;
         if (i.nodePositive > 0 && i.nodePositive <= (int)current_voltages.size()) v_diff += current_voltages[i.nodePositive-1];
         if (i.nodeNegative > 0 && i.nodeNegative <= (int)current_voltages.size()) v_diff -= current_voltages[i.nodeNegative-1];
         power_in += std::abs(i.current_A * v_diff);
    }
    // Voltage Sources: estimate power as |V| * sum of cap-branch currents at the
    // VS node (using the previous step's stored hist.i).  This prevents the
    // Lyapunov energy check from flagging normal RC charging as instability —
    // the cap current from hist reflects the actual energy the source delivers.
    for (const auto &vs : netlist.globalBlock.voltageSources) {
        int nPos = vs.nodePositive;
        double V = std::abs(vs.voltage_V);
        for (size_t ci = 0; ci < netlist.globalBlock.capacitors.size(); ++ci) {
            const auto &c = netlist.globalBlock.capacitors[ci];
            if (c.nodePlate1 != nPos && c.nodePlate2 != nPos) continue;
            if (ci < netlist.globalState.capacitorState.size() &&
                !netlist.globalState.capacitorState[ci].i.empty()) {
                power_in += V * std::abs(netlist.globalState.capacitorState[ci].i[0]);
            }
        }
    }
    // Safety Margin for Voltage Sources/Norton Equivalents (Heuristic)
    // We allow energy to grow if there is significant voltage activity, scaling with system size
    double energy_tolerance = 1e-9 + power_in * dt * 1.5; 
    
    // Strict Lyapunov Bound: E(t) <= E(t-1) + Pin*dt + tolerance
    // We use a relaxed bound here to account for numeric noise and voltage source estimation gaps
    // Guard against cold-start false positive: when prev_energy == 0 (arbiter
    // just initialized), the "* 1.5" condition is trivially true and causes a
    // spurious instability on every first step.
    bool energy_exploding = (prev_energy > 1e-12) &&
                            (current_energy > prev_energy + energy_tolerance + 1e-6) &&
                            (current_energy > prev_energy * 1.5);
    
    if (high_lte || energy_exploding) {
        instability_counter++;
        next_dt_suggestion = dt * 0.5; // Cut step
        if (instability_counter >= INSTABILITY_THRESHOLD) return 2; // Switch Method
        return 1; // Retry
    } else {
        instability_counter = std::max(0, instability_counter - 1);

        // ─── TAA: Update motion vectors (dV/dt) for next step ──────────
        // These are used by rejectHistory() to compute acceleration and
        // detect violent switching events where polynomial predictors fail.
        if (!prev_voltages.empty() && prev_dt > 0) {
            const size_t N = current_voltages.size();
            prev_dv_dt_.resize(N, 0.0);
            for (size_t i = 0; i < N; ++i) {
                prev_dv_dt_[i] = (current_voltages[i] - prev_voltages[i]) / dt;
            }
            taaInitialized_ = true;
        }

        prev_prev_voltages = prev_voltages;
        prev_voltages = current_voltages;
        prev_energy = current_energy;
        prev_dt = dt;
        stepsAccepted_++;
        
        if (lte < 0.1) next_dt_suggestion = dt * 1.5; 
        else if (lte < 0.5) next_dt_suggestion = dt * 1.1; 
        else next_dt_suggestion = dt; 
        
        if (next_dt_suggestion > 1e-3) next_dt_suggestion = 1e-3; 
        return 0;
    }
  }
};

class CircuitSim {
public:
  enum class ExecutionMode {
      TENSOR_SOA_SCALAR = 1,
      TENSOR_SOA_SIMD = 2,
      GPU = 3,
      GPU_DEBUG = 4,
      HYBRID = 5    // CPU+GPU parallel: GPU handles TN contraction while CPU stamps next iteration
  };

  IntegratorArbiter arbiter;
  bool arbiterInitialized = false;
  std::unique_ptr<IIntegrationMethod> integrator;
  ExecutionMode execMode = ExecutionMode::TENSOR_SOA_SIMD;

  // CSR pattern cache (Sparse Merge — Phase D5): rebuilt when a stamp falls
  // outside it (the topology hash ignores sources and node numbering)
  std::unique_ptr<CachedCsrPattern> cachedPattern_;

  // Default linear solver: tensor-contraction plan, rebuilt only when the
  // Jacobian's sparsity pattern (circuit topology) changes.
  std::unique_ptr<ContractionPlan> contractionPlan_;
  ContractionCache contractionCache_;  // unchanged subtrees keep their factors between solves
  SolverResult solveLinear(const Csr_matrix& A, const std::vector<double>& b) {
    if (!contractionPlan_ || !contractionPlan_->matches(A)) {
      contractionPlan_ = std::make_unique<ContractionPlan>(planContraction(A));
      contractionCache_ = {};  // factors belong to the old plan
    }
    SolverResult r = solveContraction(A, b, *contractionPlan_, 2, &contractionCache_, execMode == ExecutionMode::GPU);
    gpuFronts += r.gpuFronts;
    return r;
  }
  int gpuFronts = 0;  // fronts factored on the GPU so far (what the engine reports as the backend)

  // Converged solution of the prior solve (seeder history, dV/dt)
  std::vector<double> prevVoltages_;

  // Gap 4: Neural convergence seeder — physics-informed dV/dt extrapolation.
  // Records each converged solution and uses history for next initial guess.
  NeuralSeeder neuralSeeder_;

  // Gap 5: LTE-based adaptive timestep controller (truncation error, step size).
  LTEController lteController_;
  // Accepted steps after a waveform corner whose LTE is not checked: divided differences across
  // the kink measure the kink (an inductor's voltage jumps with dI/dt), not truncation error, and
  // rejected step after step down to femtoseconds. Three refill a trapezoidal history past it.
  int lteHold_ = 0;
  // runTransient's loop state, kept between advanceTo calls
  struct TranLoop {
    double step = 0, dtMin = 0, endTol = 0, hMax = 0, h = 0, t = 0;
    bool landedOnCorner = false;
    int tinySteps = 0;
    int accepted = 0;  // steps taken so far
    std::vector<std::function<double(double)>> corners;  // next waveform corner after x, per source
  } tran_;
  void tranCorners(const TensorNetlist &netlist);
  // Append an accepted step to a line's history and note a kink in either wave (its break is TD later)
  static void pushLine(BlockState::LineHistory &h, double dt, double w1, double w2, double td);
  double lteTolerance_ = 1e-4;  // absolute LTE tolerance
  // .options (SimulationEnvironment), applied at each analysis entry
  double reltol_ = 1e-3, vntol_ = 1e-6, abstol_ = 1e-12, gmin_ = 1e-12, gshunt_ = 0.0;
  int itl1_ = 100, itl2_ = 50, itl4_ = 0;
  IntegrationType tranMethod_ = IntegrationType::TRAPEZOIDAL;
  void applyOptions(const SimulationEnvironment &e) {
    reltol_ = e.reltol; vntol_ = e.vntol; abstol_ = e.abstol; gmin_ = e.gmin;
    gshunt_ = e.rshunt > 0.0 ? 1.0 / e.rshunt : 0.0;
    itl1_ = e.itl1; itl2_ = e.itl2; itl4_ = e.itl4;
    lteController_.setTRTOL(e.trtol);
    lteTolerance_ = 1e-4 * e.reltol / 1e-3;  // the engine's LTE bound scales with RELTOL (default unchanged)
    lteController_.setTolerance(lteTolerance_);
    // ngspice: order 1 is backward Euler for either method; trapezoidal is order 2 at most
    tranMethod_ = e.maxord <= 1 ? IntegrationType::GEAR_1_EULER : IntegrationType::TRAPEZOIDAL;
  }
  int nrLimit() const { return itl4_ > 0 ? itl4_ : PhysicsConstants::MAX_NR_ITER; }  // transient Newton iterations

  // Callbacks for orchestration
  std::function<void(int, double)> onConvergenceStep;
  std::function<void(float)>       onProgress;

  CircuitSim() {
    integrator = std::make_unique<TrapezoidalIntegration>();
  }

  /* Implicitly deleted move ops due to atomic members - safe to omit or implement deep copy if needed */
  /* CircuitSim(CircuitSim &&) = delete; */
  /* CircuitSim &operator=(CircuitSim &&) = delete; */

  void setIntegrationMethod(IntegrationType type) {
    if(!integrator || integrator->getType() != type) {
        switch (type) {
        case IntegrationType::TRAPEZOIDAL:
          integrator = std::make_unique<TrapezoidalIntegration>();
          break;
        case IntegrationType::GEAR_1_EULER:
          integrator = std::make_unique<Gear1Integration>();
          break;
        case IntegrationType::GEAR_2:
          integrator = std::make_unique<Gear2Integration>();
          break;
        }
    }
  }
  

  /*
   * Stamping Helpers
   */
  // ===========================================================================
  // COMPONENT STAMPERS (Reusability Layer)
  // ===========================================================================
  // These helpers encapsulate the physics-to-matrix mapping.

  void stampResistor(int n1, int n2, double R, MatrixConstructor& mb) {
      if (std::abs(R) < 1e-15) R = 1e-15; // Safety
      double g = 1.0 / R;
      if (n1 != 0 && n2 != 0) {
          mb.add(n1-1, n1-1, g);
          mb.add(n2-1, n2-1, g);
          mb.add(n1-1, n2-1, -g);
          mb.add(n2-1, n1-1, -g);
      } else if (n1 != 0) {
          mb.add(n1-1, n1-1, g);
      } else if (n2 != 0) {
          mb.add(n2-1, n2-1, g);
      }
  }

  // MOSFET charge branches gs, gd, gb, bd, bs (mos1_charges) as terminal
  // pairs (d=0, g=1, s=2, b=3): q flows into `from` and out of `to`.
  static constexpr int mosFrom[5] = {1, 1, 1, 3, 3}, mosTo[5] = {2, 0, 3, 0, 2};

  // Terminal currents (d, g, s, b) into a MOSFET at x: DC (mos1_currents) plus,
  // in transient, the currents of its charges integrated from the last accepted
  // step `st` (TRAP, else BE: ngspice NIintegrate). One definition for the Newton
  // stamp (T = Dual), the KCL audit and the history update (next != null).
  // ponytail: GEAR_2 (never chosen by runTransient) integrates the charges with BE.
  template <typename T>
  std::array<T, 4> mosCurrents(const std::array<T, 4> &x, const Mos1Params<double> &p,
                               const BlockState::DeviceCharge *st, double h,
                               BlockState::DeviceCharge *next = nullptr) const {
      const auto dc = mos1_currents(x[0], x[1], x[2], x[3], p);
      std::array<T, 4> I{dc.d, T(0.0), dc.s, dc.b};
      if (!p.caps || !st || !(h > 1e-15) || !integrator) return I;
      const auto c = mos1_charges(x[0], x[1], x[2], x[3], p, st->v[0], st->q, st->half);
      if constexpr (std::is_same_v<T, double>)
          if (next) {  // shift the voltage history (next may alias st)
              std::copy(st->v[1], st->v[1] + 5, next->v[2]);
              std::copy(st->v[0], st->v[0] + 5, next->v[1]);
          }
      const bool trap = integrator->getType() == IntegrationType::TRAPEZOIDAL;
      for (int k = 0; k < 5; ++k) {
          const T i = trap ? (2.0 / h) * (c.q[k] - st->q[k]) - st->i[k] : (c.q[k] - st->q[k]) / h;
          I[mosFrom[k]] += p.polarity * i;
          I[mosTo[k]] -= p.polarity * i;
          if constexpr (std::is_same_v<T, double>)
              if (next) { next->v[0][k] = c.v[k]; next->q[k] = c.q[k]; next->i[k] = i; }
      }
      if constexpr (std::is_same_v<T, double>)
          if (next) std::copy(c.half, c.half + 3, next->half);
      return I;
  }

  // Every MOSFET of the flattened netlist with its global nodes (d, g, s, b) and
  // charge history (null when the block state is not sized, e.g. GUI-built).
  template <class F> void forEachMosfet(const TensorNetlist &nl, F &&f) const {
      int total = 0;
      for (const auto &sl : branchSlots(nl, total))
          for (size_t i = 0; i < sl.blk->mosfets.size(); ++i) {
              const Mosfet &m = sl.blk->mosfets[i];
              f(m, std::array<int, 4>{sl.map(m.drain), sl.map(m.gate), sl.map(m.source), sl.map(m.body)},
                i < sl.st->mosfetState.size() ? &sl.st->mosfetState[i] : nullptr);
          }
  }

  // Newton stamp of mosCurrents() at v. The Jacobian comes from Dual (one pass
  // per terminal), so it is the exact derivative of the currents the KCL audit
  // checks: gm, gds, gmbs, the junctions and charges, either orientation.
  void stampMos1(const std::array<int, 4>& n, const Mos1Params<double>& p,
                 const BlockState::DeviceCharge* st, double h, const std::vector<double>& v) {
      stampByDual(n, 4, v, [&](const std::array<Dual<double>, 4> &x) { return mosCurrents(x, p, st, h); });
  }

  // Diodes and BJTs ("junction devices"): one evaluation of currents and charges
  // (diode_iq / bjt_gp) for the Newton stamp, both KCL audits, the history
  // update, AC and noise. Series resistances (diode RS; BJT RB, RC, RE) add an
  // internal node each, an extra unknown after the branch currents (BranchSlot::dev);
  // without the resistance the internal terminal is the external one.
  static int junctionInternals(const TensorBlock &blk) {
    int k = 0;
    for (const auto &d : blk.diodes) k += d.card.rs > 0.0;
    for (const auto &q : blk.bjts) k += (q.card.rb > 0.0) + (q.card.rc > 0.0) + (q.card.re > 0.0);
    return k;
  }

  // A junction device of the flattened netlist. n: its terminal unknowns, 1-based
  // (0 = ground; internal nodes > numGlobalNodes). Diode: A, C, A'. BJT: C, B, E,
  // S, C', B', E'. st: charge history (null when the block state is not sized).
  struct Junction {
    const Diode *d = nullptr;
    const BJT *q = nullptr;
    const JFET *f = nullptr;  // terminals D, G, S (RD / RS are series resistors from the parser)
    std::array<int, 7> n{};
    int terms = 0;
    int sub = 4;           // BJT: substrate junction's inner terminal, C' (vertical) or B' (lateral)
    double subSign = 1.0;  // BJT SUBS: substrate voltage type*SUBS*(S - that terminal)
    BlockState::DeviceCharge *st = nullptr;
  };
  template <class F> void forEachJunction(const TensorNetlist &nl, F &&f) const {
    int total = 0;
    for (const auto &sl : branchSlots(nl, total)) {
      int next = sl.dev + 1;  // 1-based unknown of the next internal node
      auto *st = const_cast<BlockState *>(sl.st);
      for (size_t i = 0; i < sl.blk->diodes.size(); ++i) {
        const Diode &d = sl.blk->diodes[i];
        Junction j;
        j.d = &d; j.terms = 3;
        j.n[0] = sl.map(d.anode); j.n[1] = sl.map(d.cathode);
        j.n[2] = d.card.rs > 0.0 ? next++ : j.n[0];
        j.st = i < st->diodeState.size() ? &st->diodeState[i] : nullptr;
        f(j);
      }
      for (size_t i = 0; i < sl.blk->bjts.size(); ++i) {
        const BJT &q = sl.blk->bjts[i];
        Junction j;
        j.q = &q; j.terms = 7;
        j.n[0] = sl.map(q.nodeCollector); j.n[1] = sl.map(q.base);
        j.n[2] = sl.map(q.emitter); j.n[3] = sl.map(q.substrate);
        j.n[4] = q.card.rc > 0.0 ? next++ : j.n[0];
        j.n[5] = q.card.rb > 0.0 ? next++ : j.n[1];
        j.n[6] = q.card.re > 0.0 ? next++ : j.n[2];
        if (q.card.lateral) { j.sub = 5; j.subSign = -1.0; }
        j.st = i < st->bjtState.size() ? &st->bjtState[i] : nullptr;
        f(j);
      }
      for (size_t i = 0; i < sl.blk->jfets.size(); ++i) {
        const JFET &jf = sl.blk->jfets[i];
        Junction j;
        j.f = &jf; j.terms = 3;
        j.n[0] = sl.map(jf.drain); j.n[1] = sl.map(jf.gate); j.n[2] = sl.map(jf.source);
        j.st = i < st->jfetState.size() ? &st->jfetState[i] : nullptr;
        f(j);
      }
    }
  }

  // Switches (S, W) of the flattened netlist. n: node unknowns 1-based (0 = ground): n+, n-, then the
  // control nodes (S) or the controlling V source's branch current (W).
  struct SwitchRef { const Switch *s; std::array<int, 3> n; char *on, *prev; };
  template <class F> void forEachSwitch(const TensorNetlist &nl, F &&f) const {
    int total = 0;
    for (const auto &sl : branchSlots(nl, total)) {
      auto *st = const_cast<BlockState *>(sl.st);
      for (size_t i = 0; i < sl.blk->switches.size() && i < st->swOn.size(); ++i) {
        const Switch &s = sl.blk->switches[i];
        SwitchRef r{&s, {sl.map(s.nodePos), sl.map(s.nodeNeg), 0}, &st->swOn[i], &st->swPrev[i]};
        r.n[2] = s.ctrlName.empty() ? 0 : sl.vs + s.ctrlSource + 1;
        f(r, s.ctrlName.empty() ? std::array<int, 2>{sl.map(s.ctrlPos), sl.map(s.ctrlNeg)} : std::array<int, 2>{0, 0});
      }
    }
  }
  // ngspice swload: above VT+VH on, below VT-VH off, in between the state of the last iteration
  // (the last accepted one at the start of a step)
  // The first DC iteration (swForce_) holds the accepted state, as ngspice's INITJCT / INITFIX.
  bool swForce_ = false;
  bool swAcOff_ = false;  // small-signal matrix: ngspice's swacload stamps ROFF whatever the operating point
  bool switchState(const SwitchRef &w, const std::array<int, 2> &c, const std::vector<double> &x) const {
    if (swForce_) return *w.prev != 0;
    auto V = [&](int g) { return g > 0 && g <= (int)x.size() ? x[g - 1] : 0.0; };
    const double ctrl = w.s->ctrlName.empty() ? V(c[0]) - V(c[1]) : V(w.n[2]);
    return ctrl > w.s->vt + w.s->vh ? true : ctrl < w.s->vt - w.s->vh ? false : *w.on != 0;
  }
  // The accepted state moves to the last stamped one (ngspice sets no breakpoint at a toggle)
  bool acceptSwitches(const TensorNetlist &nl) const {
    bool toggled = false;
    forEachSwitch(nl, [&](const SwitchRef &w, const std::array<int, 2> &) { toggled = toggled || *w.on != *w.prev; *w.prev = *w.on; });
    return toggled;
  }

  // Independent source value: its waveform in transient, else DC, times the
  // source-stepping factor (1 except while solveDC ramps the sources).
  double srcFact_ = 1.0;
  // A fatal OSDI model error during the current analysis (the model's own text):
  // it fails the analysis at once instead of retrying ever smaller steps.
  mutable std::string osdiFatal_;
  mutable std::string osdiNonFinite_;
  double ptranGmin_ = 0.0;  // node-to-ground gmin while solveDC runs its transient OP  // the last non-finite OSDI evaluation: reported if the analysis fails
  template <class S> double sourceValue(const S &s, double dc, double h, double time) const {
    return srcFact_ * (h > 1e-15 ? waveformValue(s, dc, time) : dc);
  }

  // OSDI devices (compiled Verilog-A: BSIM4, BSIM-CMG, ...). An instance's local
  // nodes are its terminals, then the internal nodes left after collapsing; the
  // internal nodes are unknowns after the junction internals (BranchSlot::dev).
  static int osdiInternals(const BlockState &st) {
    int k = 0;
    for (const auto &in : st.osdiState) k += in ? (int)in->numInternal() : 0;
    return k;
  }
  static int deviceInternals(const TensorBlock &blk, const BlockState &st) {
    return junctionInternals(blk) + osdiInternals(st);
  }
  struct OsdiRef {
    const OsdiDevice *dev;
    osdi::Instance *in;
    std::vector<int> n;  // 1-based unknown of each local node (0 = ground)
  };
  // osdiCurrents() of every OSDI device at v, evaluated in parallel (an instance
  // writes only its own data; model evaluation was ~half of a BSIM4 transient), in
  // forEachOsdi order: osdiRefs_[i], osdiI_[i] and, with jac, osdiJ_[i]. Callers
  // accumulate serially, so stamps and residuals keep their order.
  // The KCL audit at a Newton iterate and the next iteration's stamp evaluate at the
  // same point (OSDI devices have no voltage limiting), so the last evaluation is
  // reused when v (bitwise), h, the method and the devices match; an accepted step
  // moves the charge history and clears it. Jacobians are always kept for that.
  mutable std::vector<OsdiRef> osdiRefs_;
  mutable std::vector<std::vector<double>> osdiI_, osdiJ_;
  mutable std::vector<double> osdiAtV_;
  mutable double osdiAtH_ = -1.0;
  mutable int osdiAtMethod_ = -1;
  mutable bool osdiAtValid_ = false;
  void osdiAll(const TensorNetlist &nl, const std::vector<double> &v, double h, bool /*jac*/, bool accept) const {
    // The device list (with its node vectors) is rebuilt only when the devices change:
    // walking the instances and comparing pointers allocates nothing (the rebuild did,
    // per device per call: ~25 % of a BSIM4 chain transient).
    size_t k = 0;
    bool listed = true;
    int total = 0;
    for (const auto &sl : branchSlots(nl, total))
      for (size_t i = 0; i < sl.blk->osdiDevices.size() && i < sl.st->osdiState.size(); ++i) {
        const osdi::Instance *in = sl.st->osdiState[i].get();
        if (!in) continue;
        listed = listed && k < osdiRefs_.size() && osdiRefs_[k].dev == &sl.blk->osdiDevices[i] && osdiRefs_[k].in == in;
        ++k;
      }
    if (!listed || k != osdiRefs_.size()) {
      osdiRefs_.clear();
      forEachOsdi(nl, [&](const OsdiRef &r) { osdiRefs_.push_back(r); });
      osdiAtValid_ = false;
    }
    const int method = integrator ? (int)integrator->getType() : -1;
    if (osdiAtValid_ && !accept && h == osdiAtH_ && method == osdiAtMethod_ && v == osdiAtV_) return;
    const int n = (int)osdiRefs_.size();
    osdiI_.resize(n);
    osdiJ_.resize(n);
    parallelFor(n, 2e4 * n,  // ~5 us an evaluation: a few devices are worth the pool
                [&](int i) { osdiI_[i] = osdiCurrents(osdiRefs_[i], v, h, &osdiJ_[i], accept); });
    osdiAtValid_ = !accept;
    osdiAtV_ = v;
    osdiAtH_ = h;
    osdiAtMethod_ = method;
  }
  template <class F> void forEachOsdi(const TensorNetlist &nl, F &&f) const {
    int total = 0;
    for (const auto &sl : branchSlots(nl, total)) {
      int next = sl.dev + junctionInternals(*sl.blk) + 1;
      for (size_t i = 0; i < sl.blk->osdiDevices.size() && i < sl.st->osdiState.size(); ++i) {
        osdi::Instance *in = sl.st->osdiState[i].get();
        if (!in) continue;
        OsdiRef r{&sl.blk->osdiDevices[i], in, {}};
        for (int t : r.dev->nodes) r.n.push_back(sl.map(t));
        for (uint32_t k = 0; k < in->numInternal(); ++k) r.n.push_back(next++);
        f(r);
      }
    }
  }

  // Currents into an OSDI device's local nodes at the unknowns v: the resistive
  // residual plus, in transient, its charges' current integrated from the last
  // accepted step (TRAP, else BE, as mosCurrents). One definition for the stamp,
  // the KCL audit and the history update. J: dI/dx = G + coef*C (row-major).
  // accept: store this point as the accepted step. A fatal model error gives NaN.
  static std::mutex &osdiErrorMutex() { static std::mutex m; return m; }
  std::vector<double> osdiCurrents(const OsdiRef &r, const std::vector<double> &v, double h,
                                   std::vector<double> *J = nullptr, bool accept = false) const {
    const size_t L = r.n.size();
    std::vector<double> x(L), I(L);
    for (size_t k = 0; k < L; ++k) x[k] = r.n[k] > 0 && r.n[k] <= (int)v.size() ? v[r.n[k] - 1] : 0.0;
    const bool tran = h > 1e-15 && integrator;
    osdi::Instance &in = *r.in;
    const uint32_t flags = tran ? osdi::ANALYSIS_TRAN : osdi::ANALYSIS_DC | osdi::ANALYSIS_STATIC;
    // Device bypass (ngspice's, voltage criterion): a device whose local voltages moved
    // less than RELTOL*|v| + VNTOL (1e-3, 1 uV) since its last evaluation, of the same
    // analysis, is not evaluated; its stored f, q extrapolate linearly with G, C
    // (error O(dv^2)). 98 % of the evaluations of a 1000-stage BSIM4 chain transient
    // were such. Never for the Jacobian at an operating point itself (AC, noise,
    // sensitivity run with limitJunctions_ off). TREESPICE_BYPASS=0 evaluates always.
    static const bool bypassOn = [] { const char *s = std::getenv("TREESPICE_BYPASS"); return !s || std::atoi(s) != 0; }();
    bool bypass = bypassOn && limitJunctions_ && in.evalFlags == flags && in.evalAt.size() == L;
    for (size_t k = 0; bypass && k < L; ++k)
      bypass = std::abs(x[k] - in.evalAt[k]) <= 1e-3 * std::max(std::abs(x[k]), std::abs(in.evalAt[k])) + 1e-6;
    in.gmin = gmin_;
    if (!bypass && !in.evaluate(x.data(), flags, 0.0)) {
      std::lock_guard<std::mutex> lock(osdiErrorMutex());  // devices evaluate in parallel
      osdiFatal_ = r.dev->instanceName + ": " + in.error();  // reported instead of a singular matrix
      std::fill(I.begin(), I.end(), NAN);
      if (J) J->assign(L * L, 0.0);
      return I;
    }
    std::vector<double> f(in.f()), q(in.q());  // at x: as evaluated, or extrapolated when bypassed
    if (bypass)
      for (size_t a = 0; a < L; ++a)
        for (size_t b = 0; b < L; ++b) {
          const double d = x[b] - in.evalAt[b];
          f[a] += in.G(a, b) * d;
          q[a] += in.C(a, b) * d;
        }
    for (size_t k = 0; k < L; ++k)
      if (!std::isfinite(f[k]) || !std::isfinite(q[k])) {
        // An extreme Newton iterate may do this and the homotopy recovers, so it
        // only explains a failed analysis (osdiNonFinite_), it does not end one.
        std::lock_guard<std::mutex> lock(osdiErrorMutex());
        osdiNonFinite_ = r.dev->instanceName + ": the model returned a non-finite current or charge "
                                               "(parameters outside its valid range?)";
        std::fill(I.begin(), I.end(), NAN);
        if (J) J->assign(L * L, 0.0);
        return I;
      }
    const bool trap = tran && integrator->getType() == IntegrationType::TRAPEZOIDAL;
    const double coef = !tran ? 0.0 : trap ? 2.0 / h : 1.0 / h;
    for (size_t k = 0; k < L; ++k) {
      const double iq = tran ? coef * (q[k] - in.qPrev[k]) - (trap ? in.iPrev[k] : 0.0) : 0.0;
      I[k] = f[k] + iq;
      if (accept) { in.qPrev[k] = q[k]; in.iPrev[k] = iq; }
    }
    if (accept) { in.vHist[2] = in.vHist[1]; in.vHist[1] = in.vHist[0]; in.vHist[0] = x; }
    if (J) {
      J->assign(L * L, 0.0);
      for (size_t a = 0; a < L; ++a)
        for (size_t b = 0; b < L; ++b) (*J)[a * L + b] = in.G(a, b) + coef * in.C(a, b);
    }
    // A branch-current unknown (V(a,b) <+ 0 branches, e.g. BSIM4's body network with
    // rbodymod off, which closes a loop of zero-volt branches) gets 1 uOhm in series,
    // so roundoff cannot drive a circulating current (1e-12 there gave 1e-4 A).
    // Internal nodes get nothing, as in ngspice: a floating one (an unconnected bulk)
    // is set by the model's own junction gmin, which a leak to ground would override.
    for (size_t k = in.terminals(); k < L; ++k)
      if (in.isFlow(k)) {
        I[k] += 1e-6 * x[k];
        if (J) (*J)[k * L + k] += 1e-6;
      }
    return I;
  }

  // Newton stamp of the currents I and Jacobian J of osdiCurrents() at v (a node
  // shared by two terminals sums).
  void stampOsdi(const OsdiRef &r, const std::vector<double> &I, const std::vector<double> &J,
                 const std::vector<double> &v) {
    const size_t L = r.n.size();
    auto V = [&](int g) { return g > 0 && g <= (int)v.size() ? v[g - 1] : 0.0; };
    for (size_t t = 0; t < L; ++t) {
      if (r.n[t] <= 0) continue;
      double ieq = I[t];
      for (size_t u = 0; u < L; ++u) {
        const double g = J[t * L + u];
        if (g == 0.0) continue;
        ieq -= g * V(r.n[u]);
        if (r.n[u] > 0) matrixBuilder.add(r.n[t] - 1, r.n[u] - 1, g);
      }
      rhsVector[r.n[t] - 1] -= ieq;
    }
  }

  // Charge branches (terminal pairs, q flows from -> to): diode A'->C; BJT
  // be (B'->E'), bc (B'->C'), bx (B->C'), substrate (S->C'; lateral: S->B', Junction::sub).
  static constexpr int diodeFrom[1] = {2}, diodeTo[1] = {1};
  static constexpr int bjtFrom[4] = {5, 5, 1, 3}, bjtTo[4] = {6, 4, 4, 4};
  static constexpr int jfetFrom[2] = {1, 1}, jfetTo[2] = {2, 0};  // JFET gs (G->S), gd (G->D)
  static double jfetType(const JFET &f) { return f.isNChannel ? 1.0 : -1.0; }

  // Branch voltages and charges (type domain) of j at terminal values x.
  // Junction-limiting shifts (junctionLimit): added to the junction voltages
  // (diode vd; BJT vbe, vbc) so the device is evaluated at its limited point.
  // seed: the shift carries derivative 1 (the G*(vl - v) correction pass).
  struct JShift { double d[2] = {0.0, 0.0}; bool seed = false; };
  template <typename T> static T shifted(const T &v, const JShift *sh, int k) {
    if (!sh || sh->d[k] == 0.0) return v;
    if constexpr (std::is_same_v<T, double>) return v + sh->d[k];
    else return v + T(sh->d[k], sh->seed ? sh->d[k] : 0.0);
  }

  template <typename T>
  int junctionCharges(const Junction &j, const std::array<T, 7> &x, T *v, T *q,
                      const JShift *sh = nullptr) const {
    if (j.d) {
      v[0] = shifted(T(x[2] - x[1]), sh, 0);
      q[0] = diode_iq(v[0], diodeModel(*j.d, gmin_)).second;
      return 1;
    }
    if (j.f) {
      const double t = jfetType(*j.f);
      v[0] = shifted(T(t * (x[1] - x[2])), sh, 0); v[1] = shifted(T(t * (x[1] - x[0])), sh, 1);
      const auto c = jfet_charges(v[0], v[1], jfetModel(*j.f, gmin_));
      q[0] = c[0]; q[1] = c[1];
      return 2;
    }
    const auto p = bjtModel(*j.q, gmin_);
    const double t = p.type;
    for (int k = 0; k < 3; ++k) v[k] = t * (x[bjtFrom[k]] - x[bjtTo[k]]);
    v[3] = t * j.subSign * (x[3] - x[j.sub]);
    v[0] = shifted(v[0], sh, 0); v[1] = shifted(v[1], sh, 1);
    const auto e = bjt_gp(v[0], v[1], v[2], v[3], p);
    for (int k = 0; k < 4; ++k) q[k] = e.q[k];
    return 4;
  }
  static bool junctionHasCharge(const Junction &j) {
    if (j.d) return j.d->card.cjo > 0.0 || j.d->card.cjsw > 0.0 || j.d->card.tt > 0.0;
    if (j.f) return j.f->card.cgs > 0.0 || j.f->card.cgd > 0.0;
    const auto &c = j.q->card;
    return c.cje > 0.0 || c.cjc > 0.0 || c.cjs > 0.0 || c.tf > 0.0 || c.tr > 0.0;
  }

  // Terminal currents into j at x: DC plus, in transient, the charge currents
  // integrated from the last accepted step (TRAP, else BE), as mosCurrents().
  // next != null: write the accepted charge state (history update).
  template <typename T>
  std::array<T, 7> junctionCurrents(const Junction &j, const std::array<T, 7> &x, double h,
                                    BlockState::DeviceCharge *next = nullptr, const JShift *sh = nullptr) const {
    std::array<T, 7> I{};
    for (auto &i : I) i = T(0.0);
    double type = 1.0;
    const int *from, *to;
    if (j.d) {
      const auto p = diodeModel(*j.d, gmin_);
      if (p.rs > 0.0) { const T ir = (x[0] - x[2]) / p.rs; I[0] += ir; I[2] -= ir; }
      const T id = diode_iq(shifted(T(x[2] - x[1]), sh, 0), p).first;
      I[2] += id; I[1] -= id;
      from = diodeFrom; to = diodeTo;
    } else if (j.f) {
      const auto p = jfetModel(*j.f, gmin_);
      type = p.type;
      const auto e = jfet_eval(shifted(T(type * (x[1] - x[2])), sh, 0), shifted(T(type * (x[1] - x[0])), sh, 1), p);
      I[0] += type * (e.id - e.igd); I[2] += type * (-e.id - e.igs); I[1] += type * (e.igs + e.igd);
      from = jfetFrom; to = jfetTo;
    } else {
      const auto p = bjtModel(*j.q, gmin_);
      type = p.type;
      auto res = [&](int a, int b, double r) {
        if (r > 0.0) { const T i = (x[a] - x[b]) / r; I[a] += i; I[b] -= i; }
      };
      double rbb = p.rb;  // RBM != RB: the base resistance at this iterate's bias, held in the Jacobian (ngspice's gx)
      if (p.varRb) {
        auto dv = [](const T &v) { if constexpr (std::is_same_v<T, double>) return v; else return v.val; };
        const auto ev = bjt_gp<double, double>(dv(shifted(T(type * (x[5] - x[6])), sh, 0)), dv(shifted(T(type * (x[5] - x[4])), sh, 1)),
                                               dv(T(type * (x[1] - x[4]))), dv(T(type * j.subSign * (x[3] - x[j.sub]))), p);
        rbb = bjt_rbb<double, double>(ev.qb, ev.cb, p);
      }
      res(0, 4, p.rc); res(1, 5, rbb); res(2, 6, p.re);
      const auto e = bjt_gp(shifted(T(type * (x[5] - x[6])), sh, 0), shifted(T(type * (x[5] - x[4])), sh, 1),
                            T(type * (x[1] - x[4])), T(type * j.subSign * (x[3] - x[j.sub])), p);
      I[4] += type * e.cc; I[5] += type * e.cb; I[6] -= type * (e.cc + e.cb);
      I[3] += type * j.subSign * e.cs; I[j.sub] -= type * j.subSign * e.cs;  // substrate junction
      from = bjtFrom; to = bjtTo;
    }
    const BlockState::DeviceCharge *st = j.st;
    if (!st || !(h > 1e-15) || !integrator || !junctionHasCharge(j)) return I;
    T v[5], q[5];
    const int K = junctionCharges(j, x, v, q, sh);
    if constexpr (std::is_same_v<T, double>)
      if (next) {  // shift the voltage history (next may alias st)
        std::copy(st->v[1], st->v[1] + 5, next->v[2]);
        std::copy(st->v[0], st->v[0] + 5, next->v[1]);
      }
    const bool trap = integrator->getType() == IntegrationType::TRAPEZOIDAL;
    for (int k = 0; k < K; ++k) {
      const T i = trap ? (2.0 / h) * (q[k] - st->q[k]) - st->i[k] : (q[k] - st->q[k]) / h;
      const bool sub = j.q && k == 3;  // substrate branch: to Junction::sub, SUBS sign
      I[from[k]] += (sub ? j.subSign : 1.0) * type * i;
      I[sub ? j.sub : to[k]] -= (sub ? j.subSign : 1.0) * type * i;
      if constexpr (std::is_same_v<T, double>)
        if (next) { next->v[0][k] = v[k]; next->q[k] = q[k]; next->i[k] = i; }
    }
    return I;
  }

  // Newton stamp of f(x) (terminal currents) over the terminal unknowns n.
  // Terminals may share an unknown (an internal node without its resistance),
  // so derivatives are seeded per unknown, not per terminal.
  template <size_t NT, class F>
  void stampByDual(const std::array<int, NT> &n, int terms, const std::vector<double> &v, F &&f) {
    auto V = [&](int k) { return n[k] > 0 && n[k] <= (int)v.size() ? v[n[k] - 1] : 0.0; };
    std::array<double, NT> I{};
    bool any = false;
    for (int u = 0; u < terms; ++u) {
      if (n[u] <= 0) continue;
      bool seen = false;
      for (int w = 0; w < u; ++w) seen = seen || n[w] == n[u];
      if (seen) continue;
      any = true;
      std::array<Dual<double>, NT> x;
      for (int k = 0; k < (int)NT; ++k) x[k] = Dual<double>(V(k), k < terms && n[k] == n[u] ? 1.0 : 0.0);
      const auto cur = f(x);
      for (int t = 0; t < terms; ++t) {
        I[t] = cur[t].val;
        if (n[t] <= 0 || cur[t].grad == 0.0) continue;
        matrixBuilder.add(n[t] - 1, n[u] - 1, cur[t].grad);
        rhsVector[n[t] - 1] += cur[t].grad * V(u);  // -(ieq) = -(I - J V)
      }
    }
    if (!any) return;  // every terminal grounded
    for (int t = 0; t < terms; ++t)
      if (n[t] > 0) rhsVector[n[t] - 1] -= I[t];
  }

  // ngspice junction limiting (DEVpnjlim in dioload/bjtload), device-local: the
  // raw junction voltages at x are limited against the last iteration's
  // (DeviceCharge::jv) and the device is linearised there; node voltages are
  // left alone. First iteration of a solve: vcrit (BJT vbc = 0), as MODEINITJCT.
  JShift junctionLimit(const Junction &j, const std::vector<double> &x) const {
    JShift sh;
    if (!j.st) return sh;
    auto V = [&](int k) { return j.n[k] > 0 && j.n[k] <= (int)x.size() ? x[j.n[k] - 1] : 0.0; };
    auto *st = j.st;
    if (j.d) {
      const auto m = diodeModel(*j.d, gmin_);
      const double vcrit = compute_vcrit(m.vte, m.is), raw = V(2) - V(1);
      double vl = vcrit;
      if (st->jvState) {
        vl = (m.breakdown && raw < std::min(0.0, -m.xbv + 10.0 * m.vte))
                 ? -(pnjlim(-(raw + m.xbv), -(st->jv[0] + m.xbv), m.vte, vcrit) + m.xbv)
                 : pnjlim(raw, st->jv[0], m.vte, vcrit);
      }
      st->jv[0] = vl; st->jvState = 1;
      sh.d[0] = vl - raw;
      return sh;
    }
    if (j.f) {  // both gate junctions start at -1 V (ngspice jfetload INITJCT)
      const auto m = jfetModel(*j.f, gmin_);
      const double vcrit = compute_vcrit(m.vt, m.is), raw[2] = {m.type * (V(1) - V(2)), m.type * (V(1) - V(0))};
      for (int k = 0; k < 2; ++k) {
        const double vl = st->jvState ? pnjlim(raw[k], st->jv[k], m.vt, vcrit) : -1.0;
        st->jv[k] = vl;
        sh.d[k] = vl - raw[k];
      }
      st->jvState = 1;
      return sh;
    }
    const auto m = bjtModel(*j.q, gmin_);
    const double vt = m.vtf / j.q->card.nf, vcrit = compute_vcrit(vt, m.is);
    const double raw[2] = {m.type * (V(5) - V(6)), m.type * (V(5) - V(4))};
    for (int k = 0; k < 2; ++k) {
      const double vl = st->jvState ? pnjlim(raw[k], st->jv[k], vt, vcrit) : (k == 0 ? vcrit : 0.0);
      st->jv[k] = vl;
      sh.d[k] = vl - raw[k];
    }
    st->jvState = 1;
    return sh;
  }
  // Limiting state from x (a Newton restart point) or back to MODEINITJCT.
  void syncJunctionLimits(const TensorNetlist &nl, const std::vector<double> &x) const {
    forEachJunction(nl, [&](const Junction &j) {
      if (!j.st) return;
      auto V = [&](int k) { return j.n[k] > 0 && j.n[k] <= (int)x.size() ? x[j.n[k] - 1] : 0.0; };
      const double t = j.q ? (j.q->isNPN ? 1.0 : -1.0) : j.f ? jfetType(*j.f) : 1.0;
      j.st->jv[0] = j.d ? V(2) - V(1) : j.f ? t * (V(1) - V(2)) : t * (V(5) - V(6));
      j.st->jv[1] = j.d ? 0.0 : j.f ? t * (V(1) - V(0)) : t * (V(5) - V(4));
      j.st->jvState = 1;
    });
  }
  void resetJunctionLimits(const TensorNetlist &nl) const {
    forEachJunction(nl, [&](const Junction &j) { if (j.st) j.st->jvState = 0; });
  }
  bool limitJunctions_ = true;  // off for the AC / sensitivity Jacobian at a converged point

  // Newton stamp of a junction device: I(x) ~ I(vl) + G (v(x) - vl) at the
  // limited junction voltages vl (ngspice's companion), i.e. stampByDual at the
  // shifted point plus the G*(vl - v) correction from one seeded pass.
  void stampJunction(const Junction &j, const std::vector<double> &v, double h) {
    const JShift sh = limitJunctions_ ? junctionLimit(j, v) : JShift{};
    stampByDual(j.n, j.terms, v, [&](const std::array<Dual<double>, 7> &x) { return junctionCurrents(j, x, h, nullptr, &sh); });
    if (sh.d[0] == 0.0 && sh.d[1] == 0.0) return;
    JShift seeded = sh;
    seeded.seed = true;
    std::array<Dual<double>, 7> x;
    for (int k = 0; k < 7; ++k) x[k] = Dual<double>(j.n[k] > 0 && j.n[k] <= (int)v.size() ? v[j.n[k] - 1] : 0.0, 0.0);
    const auto cur = junctionCurrents(j, x, h, nullptr, &seeded);
    for (int t = 0; t < j.terms; ++t)
      if (j.n[t] > 0) rhsVector[j.n[t] - 1] += cur[t].grad;
  }

  SolverStep solveDC(TensorNetlist &netlist);
  SolverStep stepTransient(TensorNetlist &netlist, double timeStep, double currentTime);

  // The one transient loop (session, GUI driver, Fourier). t = 0 state: the DC
  // operating point with .ic nodes forced, or with `uic` the .ic / IC= values.
  // Steps to `stop` at `step`; a failed step retries at h/4 (down to step*1e-9)
  // and regrows. `onPoint` gets each accepted step and returns false to cancel.
  // Returns an empty string on success, else the error.
  std::string runTransient(TensorNetlist &netlist, double step, double stop, bool uic,
                           const std::function<bool(const SolverStep &)> &onPoint);

  // The same loop in lockstep with an outside driver: beginTransient runs the t = 0
  // prologue (`stop` > 0 only sets ngspice's step caps; 0 = unknown, cap at `step`),
  // advanceTo(tEnd) steps until time() == tEnd (the last step is clipped to it, the
  // unclipped one carries to the next call), setSource re-drives a named V (or I)
  // source from its value now to `volts` over `rampS` (0 = a step at the instant).
  // The netlist must outlive the run and not be restructured. advanceTo and
  // beginTransient return "" on success, else the error.
  std::string beginTransient(TensorNetlist &netlist, double step, bool uic, double stop = 0.0);
  std::string advanceTo(TensorNetlist &netlist, double tEnd,
                        const std::function<bool(const SolverStep &)> &onPoint = {});
  bool setSource(TensorNetlist &netlist, const std::string &name, double volts, double rampS);
  double tranTime() const { return tran_.t; }
  double nodeVoltage(int node) const {  // node 0 = ground
    return node > 0 && node <= (int)lastSolution_.size() ? lastSolution_[node - 1] : 0.0;
  }

  // AC small-signal sweep around the operating point of the last solveDC().
  // Y(jw) = G + jwC: G is the Newton Jacobian at the OP (the DC stamps, so every
  // device, model card, subcircuit and MNA branch row), C the capacitors plus the
  // inductor/mutual L terms of the branch rows; excitation = AC mag/phase of the
  // V and I sources. Solved as the real 2N system [[G, -wC], [wC, G]] by the
  // contraction solver (one plan for the sweep). Returns x(f): nodes, then branches;
  // empty if solveDC() has not converged on this netlist.
  std::vector<std::vector<std::complex<double>>> solveAC(TensorNetlist &netlist,
                                                         const std::vector<double> &freqs);

  // Noise around the last solveDC() operating point (ngspice .noise semantics).
  // Generators: resistor thermal 4kT/R (incl. diode RS, BJT RB/RC/RE), diode shot 2q|Id|,
  // BJT shot 2q|Ic| (C'-E') and 2q|Ib| (B'-E'), MOSFET/JFET channel thermal 4kT*(2/3)*gm (MOS1). Per
  // frequency one adjoint solve gives every generator's transfer to
  // v(out)-v(ref) and one forward solve the gain from `inputSource` at unit AC;
  // totals integrate each generator with ngspice's power law (Nintegrate).
  // Densities in V/sqrt(Hz), totals in V rms; `error` is set on failure.
  struct NoiseResult {
    std::vector<double> onoise, inoise;
    double onoiseTotal = 0.0, inoiseTotal = 0.0;
    std::string error;
  };
  NoiseResult solveNoise(TensorNetlist &netlist, const std::vector<double> &freqs,
                         int outNode, int refNode, const std::string &inputSource);

  // Poles and zeros of the transfer function from (inP, inN) to v(outP) - v(outN) at the last solveDC()
  // operating point (ngspice .pz): poles are the roots of det(G + sC) with the input open (`volt` false) or
  // the input nodes shorted by an ideal voltage source (`volt`); zeros those of the system matrix bordered by
  // the input column and the output row. Nodes are global (0 = ground). `error` is set on failure.
  struct PoleZeroResult { std::vector<std::complex<double>> poles, zeros; std::string error; };
  PoleZeroResult solvePoleZero(TensorNetlist &netlist, int inP, int inN, int outP, int outN, bool volt,
                               bool wantPoles, bool wantZeros);

  // DC sensitivity of v(outNode) - v(refNode), or of i(outSource) when outSource is set, to every element value
  // at the last solveDC() operating point (ngspice .sens), by the adjoint method on the final Jacobian: one
  // solve J^T l = d(out)/dx, then dOut/dp = l . (-dF/dp) per parameter. Names are ngspice's vectors: r1 (d/dR),
  // r1_m and r1_scale, v1, i1 (d/dI), i1_m. Parameters of other devices are listed in `unsupported`, never as 0.
  struct SensResult {
    std::vector<std::string> names, unsupported;
    std::vector<double> values;
    std::string error;
  };
  SensResult solveSens(TensorNetlist &netlist, int outNode, int refNode, const std::string &outSource);

  // The grid of ngspice 47's .ac (probed point by point). lin: n points, but n = 2 gives only the first.
  // oct / dec: N = floor(log(f2/f1) * n) steps of 2^(1/n) / 10^(1/n). dec over more than one decade refits the
  // step to (f2/f1)^(1/N) so the last point is f2 itself; oct, and dec within one decade, stop at the last point <= f2.
  // .noise (noise = true) never refits and has no lin quirk.
  // "log" (tf_analysis): n points from f1 to f2, evenly spaced in log.
  static std::vector<double> acFrequencies(const std::string &sweep, int n, double f1, double f2, bool noise = false) {
    std::vector<double> f;
    if (n < 1 || !(f1 > 0.0 || (sweep == "lin" && f1 == 0.0)) || f2 < f1) return f;
    if (sweep == "lin" || sweep == "log") {
      const int pts = sweep == "lin" && n == 2 && !noise ? 1 : n;
      for (int k = 0; k < pts; ++k) {
        const double t = n == 1 ? 0.0 : double(k) / (n - 1);
        f.push_back(sweep == "lin" ? f1 + t * (f2 - f1) : f1 * std::pow(f2 / f1, t));
      }
      return f;
    }
    const bool oct = sweep == "oct";
    const double x = std::log(f2 / f1) / std::log(oct ? 2.0 : 10.0) * n;
    const int steps = (int)std::floor(x + 1e-9);
    const bool refit = !oct && !noise && x - n > 1e-9;
    const double ratio = refit ? std::pow(f2 / f1, 1.0 / steps) : std::pow(oct ? 2.0 : 10.0, 1.0 / n);
    for (int k = 0; k <= steps; ++k) f.push_back(k == steps && refit ? f2 : f1 * std::pow(ratio, k));
    return f;
  }
  

  // Phase 1.8: PVT Application
  void applyCorner(TensorNetlist &netlist, const acutesim::physics::CornerConfig& corner) {
      netlist.environment.process = (SimulationEnvironment::Corner)corner.process;
      
      // If voltage scale is 0 or negative, ignore (safety)
      if (corner.voltageScale > 0.1) {
          netlist.environment.global_voltage_scale = corner.voltageScale;
      }

  }
  
  // Deterministic RNG (FNV-1a Hash + Box-Muller)
  static uint64_t fnv1a(const std::string& s, uint64_t seed) {
      uint64_t hash = 14695981039346656037ULL;
      hash ^= seed;
      for (char c : s) {
          hash ^= static_cast<uint64_t>(c);
          hash *= 1099511628211ULL;
      }
      return hash;
  }
  


private:
  MatrixConstructor matrixBuilder;
  std::vector<double> rhsVector;
  TensorScheduler scheduler;
  
  // SoA Tensor Storage (Batch Physics)
  TensorizedBlock soaBlock;

  // =========================================================================
  // PHASE 1 HARDENING: Helper Methods (Decomposition of solveDC)
  // =========================================================================



  /**
   * coldStartSweep - SPICE3-accurate cold-start using dt_init = tstep/100 micro-steps.
   * Replaces the single Backward Euler first step with nSubsteps internal BE steps,
   * each of duration totalTime/nSubsteps, accumulating history so that the final
   * capacitor state matches what SPICE3 produces at t = tstep.
   *
   * @param netlist    Circuit netlist (AoS history is updated in-place per sub-step)
   * @param totalTime  Full first timestep duration (the tstep value)
   * @param nSubsteps  Number of internal micro-steps (typically COLD_START_SUBSTEPS=100)
   * @param finalVoltages  Output: converged voltages after the last micro-step
   * @return true if all micro-steps converged; false if any micro-step NR failed
   */
  bool coldStartSweep(TensorNetlist& netlist, double totalTime, int nSubsteps,
                      std::vector<double>& finalVoltages);

  /** Tensorize `netlist` into soaBlock and sync reactive-element history
   *  (global + instances) from the AoS state. */
  void loadSoAFromNetlist(const TensorNetlist& netlist);

  /** Capacitor + inductor history after an accepted step (global + instances),
   *  using the same companion formulas as the stamps and the KCL audit. */
  void updateReactiveHistory(TensorNetlist& netlist, const std::vector<double>& v, double h);
  void seedReactiveHistory(TensorNetlist& netlist, const std::vector<double>& x, bool uic);

  /** Last full solution x (node voltages + branch currents). */
  std::vector<double> lastSolution_;
  /** Newton starting point for a transient step: the last full solution when
   *  the topology matches, else the arbiter's node voltages, zero-padded. */
  std::vector<double> transientSeed(const TensorNetlist& netlist, int nTot) {
    if ((int)lastSolution_.size() == nTot) return lastSolution_;
    std::vector<double> x = arbiter.getPrevVoltages();
    x.resize(nTot, 0.0);
    return x;
  }

  // NR Parameters (Using centralized PhysicsConstants)
  // Note: Kept as local constexpr for backward compatibility; prefer PhysicsConstants::
  static constexpr int MAX_NR_ITER = PhysicsConstants::MAX_NR_ITER;
  static constexpr double PCG_TOL = 1e-10;
  static constexpr int PCG_MAX_ITER = 1000;

  /**
   * checkConvergence
   * Verifies BOTH voltage stability (approximate) AND physical charge
   * conservation (exact).
   */
  bool checkConvergence(const std::vector<double> &v_new,
                        const std::vector<double> &v_old,
                        double max_kcl_error,
                        double kcl_tol = 1e-6) {
    // Every test is written as "pass only if x <= tol" so that NaN (for which
    // all comparisons are false) fails instead of slipping through.
    if (v_new.size() != v_old.size()) return false;
    for (size_t i = 0; i < v_new.size(); ++i)
      if (!std::isfinite(v_new[i])) return false;

    // Both the update (RELTOL/VNTOL, as ngspice) and KCL must pass. A former
    // "KCL < 1 nA => converged" shortcut accepted stale voltages at
    // high-impedance nodes (a 555 output stage settled 0.5 V off, above VCC).

    // 1. Check Voltage Updates (Standard RELTOL/VNTOL)
    for (size_t i = 0; i < v_new.size(); ++i) {
      double diff = std::abs(v_new[i] - v_old[i]);
      double limit = reltol_ * std::max(std::abs(v_new[i]), std::abs(v_old[i])) + vntol_;
      if (!(diff <= limit)) return false;
    }

    // 2. Check Physical KCL Conservation (Research-Grade)
    // Relax ABSTOL slightly for Norton-stamped voltage sources (which are
    // stiff)
    return max_kcl_error <= kcl_tol;
  }

public:
  // ── MNA branch currents ────────────────────────────────────────────────
  // Unknowns x = [node voltages (N) | branch currents (B)]. V sources,
  // inductors, E (VCVS) and H (CCVS) each carry a branch current flowing from
  // their + node through the element to their - node; F (CCCS) and H read the
  // branch current of their controlling V source. Branch order: all V
  // sources, then inductors, then E, then H, each over the global block then
  // every instance (tensorizeNetlist order), so indices are fixed per topology.
  // These replace the 1 mOhm / 100 uOhm Norton approximations: sources and
  // shorts are now exact (an SoC power grid IS milliohms of metal).
  struct BranchSlot {
    const TensorBlock *blk;
    const BlockState *st;
    std::function<int(int)> map;   // local node -> global node
    int vs, ind, e, h, tl, bv, dev;  // first matrix row of each branch kind (tl: two per transmission line; bv: voltage behavioural sources); dev: device internal nodes
    const BlockInstance *inst = nullptr;
  };
  std::vector<BranchSlot> branchSlots(const TensorNetlist &nl, int &total) const {
    std::vector<BranchSlot> branchList;
    branchList.push_back({&nl.globalBlock, &nl.globalState, [](int x) { return x; }, 0, 0, 0, 0, 0, 0, 0, nullptr});
    for (size_t k = 0; k < nl.instances.size() && k < nl.instanceStates.size(); ++k) {
      auto it = nl.blockDefinitions.find(nl.instances[k].blockName);
      if (it == nl.blockDefinitions.end()) continue;
      const auto *inst = &nl.instances[k];
      branchList.push_back({&it->second, &nl.instanceStates[k], [inst](int x) {
        return (x <= 0 || x > (int)inst->nodeMapping.size()) ? 0 : inst->nodeMapping[x - 1];
      }, 0, 0, 0, 0, 0, 0, 0, inst});
    }
    int next = nl.numGlobalNodes;
    for (auto &s : branchList) { s.vs = next; next += (int)s.blk->voltageSources.size(); }
    for (auto &s : branchList) { s.ind = next; next += (int)s.blk->inductors.size(); }
    for (auto &s : branchList) { s.e = next; next += (int)s.blk->voltageControlledVoltageSources.size(); }
    for (auto &s : branchList) { s.h = next; next += (int)s.blk->currentControlledVoltageSources.size(); }
    for (auto &s : branchList) { s.tl = next; next += 2 * (int)s.blk->transmissionLines.size(); }
    for (auto &s : branchList) {
      s.bv = next;
      for (const auto &b : s.blk->behavioural) next += b.voltage;
    }
    for (auto &s : branchList) { s.dev = next; next += deviceInternals(*s.blk, *s.st); }
    total = next;
    return branchList;
  }
  int systemSize(const TensorNetlist &nl) const { int t = 0; branchSlots(nl, t); return t; }

  // The user-visible branch currents (V, L, E, H, B-voltage; ngspice's i(name)), in x order after
  // the nodes, named like ngspice does in subcircuits: v.x1.v1. Device internals follow, unnamed.
  int branchCount(const TensorNetlist &nl) const { int t = 0; return branchSlots(nl, t)[0].dev - nl.numGlobalNodes; }
  std::vector<std::string> branchNames(const TensorNetlist &nl) const {
    int t = 0;
    const auto sl = branchSlots(nl, t);
    std::vector<std::string> names(sl[0].dev - nl.numGlobalNodes);
    auto put = [&](const BranchSlot &s, int row, const std::string &n) {
      std::string full = s.inst ? n.substr(0, 1) + "." + s.inst->name + "." + n : n;
      for (auto &c : full) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      names[row - nl.numGlobalNodes] = full;
    };
    for (const auto &s : sl) {
      for (size_t i = 0; i < s.blk->voltageSources.size(); ++i) put(s, s.vs + i, s.blk->voltageSources[i].name);
      for (size_t i = 0; i < s.blk->inductors.size(); ++i) put(s, s.ind + i, s.blk->inductors[i].name);
      for (size_t i = 0; i < s.blk->voltageControlledVoltageSources.size(); ++i)
        put(s, s.e + i, s.blk->voltageControlledVoltageSources[i].name);
      for (size_t i = 0; i < s.blk->currentControlledVoltageSources.size(); ++i)
        put(s, s.h + i, s.blk->currentControlledVoltageSources[i].name);
      int row = s.bv;
      for (const auto &b : s.blk->behavioural) if (b.voltage) put(s, row++, b.name);
    }
    return names;
  }

  // Small-signal system at the last DC operating point: G (Newton Jacobian
  // without node gmin), reactive triplets C. Shared by solveAC and solveNoise.
  // dr/dc/dv/dtd: entries dv * e^(-jw dtd) (transmission lines), beside the jw*cv ones
  struct ACSystem { Csr_matrix G; std::vector<int> cr, cc; std::vector<double> cv; std::vector<int> dr, dc; std::vector<double> dv, dtd; int n = 0; };
  bool acSystem(TensorNetlist &netlist, ACSystem &sys);
  // R(Y) = [[G, -wC], [wC, G]] for Y = G + jwC (transpose: R(Y)^T = R(Y^H)).
  static Csr_matrix acMatrix(const ACSystem &s, double w, bool transpose);
  // Excitation: the sources' AC mag/phase, or unit AC at source `unit` only
  // (empty vector if no source has that name).
  std::vector<std::complex<double>> acExcitation(const TensorNetlist &nl, const std::string &unit = "") const;

  // used[n]: some element (any block, mapped to global nodes) touches node n.
  // Recomputed per call, O(elements): a cache keyed on counts went stale when a
  // CircuitSim was reused on a differently wired netlist.
  std::vector<char> usedNodes(const TensorNetlist &nl) const {
    int total = 0;
    std::vector<char> used(nl.numGlobalNodes + 1, 0);
    for (const auto &sl : branchSlots(nl, total)) {
      auto mark = [&](std::initializer_list<int> ns) {
        for (int local : ns) { int g = sl.map(local); if (g > 0 && g <= nl.numGlobalNodes) used[g] = 1; }
      };
      const auto &b = *sl.blk;
      for (const auto &e : b.resistors) mark({e.nodeTerminal1, e.nodeTerminal2});
      for (const auto &e : b.capacitors) mark({e.nodePlate1, e.nodePlate2});
      for (const auto &e : b.inductors) mark({e.nodeCoil1, e.nodeCoil2});
      for (const auto &e : b.voltageSources) mark({e.nodePositive, e.nodeNegative});
      for (const auto &e : b.currentSources) mark({e.nodePositive, e.nodeNegative});
      for (const auto &e : b.diodes) mark({e.anode, e.cathode});
      for (const auto &e : b.mosfets) mark({e.drain, e.gate, e.source, e.body});
      for (const auto &e : b.bjts) mark({e.nodeCollector, e.base, e.emitter, e.substrate});
      for (const auto &e : b.osdiDevices)
        for (int t : e.nodes) mark({t});
      for (const auto &e : b.jfets) mark({e.drain, e.gate, e.source});
      for (const auto &e : b.switches) mark({e.nodePos, e.nodeNeg, e.ctrlPos, e.ctrlNeg});
      for (const auto &e : b.voltageControlledVoltageSources) mark({e.outPos, e.outNeg, e.ctrlPos, e.ctrlNeg});
      for (const auto &e : b.voltageControlledCurrentSources) mark({e.outPos, e.outNeg, e.ctrlPos, e.ctrlNeg});
      for (const auto &e : b.currentControlledVoltageSources) mark({e.outPos, e.outNeg});
      for (const auto &e : b.currentControlledCurrentSources) mark({e.outPos, e.outNeg});
      for (const auto &e : b.transmissionLines) mark({e.nodePort1Pos, e.nodePort1Neg, e.nodePort2Pos, e.nodePort2Neg});
      for (const auto &e : b.behavioural) mark({e.outPos, e.outNeg});
      for (const auto &e : b.parasiticResistors) mark({e.node1, e.node2});
      for (const auto &e : b.parasiticCapacitors) mark({e.node1, e.node2});
    }
    return used;
  }

  // flux[a] = {(b, L_ab)}: self inductance plus K couplings M = k*sqrt(La*Lb).
  // Shared by the transient branch rows and the AC reactive matrix.
  static std::vector<std::vector<std::pair<int, double>>> fluxLinks(const TensorBlock &blk) {
    const auto &L = blk.inductors;
    std::vector<std::vector<std::pair<int, double>>> flux(L.size());
    for (size_t a = 0; a < L.size(); ++a) flux[a].push_back({(int)a, L[a].inductance_henries});
    for (const auto &k : blk.mutualInductors) {
      int a = k.inductor1_index, c = k.inductor2_index;
      if (a < 0 || c < 0 || a >= (int)L.size() || c >= (int)L.size() || a == c) continue;
      double M = k.couplingCoefficient_k * std::sqrt(L[a].inductance_henries * L[c].inductance_henries);
      flux[a].push_back({c, M});
      flux[c].push_back({a, M});
    }
    return flux;
  }

  // Outgoing waves of a lossless line at time t, linearly interpolated; before the first
  // point (t < 0) and past the last they hold (the step cap TD keeps t - TD inside the history).
  static void lineWaves(const BlockState::LineHistory &h, double t, double &w1, double &w2) {
    if (h.t.empty()) { w1 = w2 = 0.0; return; }
    const size_t k = std::upper_bound(h.t.begin(), h.t.end(), t) - h.t.begin();
    if (k == 0) { w1 = h.w1[0]; w2 = h.w2[0]; return; }
    if (k == h.t.size()) { w1 = h.w1.back(); w2 = h.w2.back(); return; }
    const double f = (t - h.t[k - 1]) / (h.t[k] - h.t[k - 1]);
    w1 = h.w1[k - 1] + f * (h.w1[k] - h.w1[k - 1]);
    w2 = h.w2[k - 1] + f * (h.w2[k] - h.w2[k - 1]);
  }
  // AC build: transmission lines stamp only their frequency-independent half (acSystem adds the delayed half)
  bool acLines_ = false;

  // All branch-element matrix entries + rhs. Stamping adds them to the matrix;
  // the KCL audit evaluates rhs - A x from the SAME triplets, so the two can
  // never disagree.
  struct BranchSystem { std::vector<int> r, c; std::vector<double> v, rhs; };
  void buildBranchSystem(const TensorNetlist &nl, double h, double time, BranchSystem &S) const {
    int total = 0;
    auto branchList = branchSlots(nl, total);
    S.r.clear(); S.c.clear(); S.v.clear(); S.rhs.assign(total, 0.0);
    auto add = [&](int r, int c, double v) {
      if (r >= 0 && c >= 0) { S.r.push_back(r); S.c.push_back(c); S.v.push_back(v); }
    };
    const bool tran = h > 1e-15 && integrator;
    const IntegrationType type = tran ? integrator->getType() : IntegrationType::GEAR_1_EULER;
    // Inductor companion: v = coef * sum_j L_aj * i_j + history (per method)
    const double coef = !tran ? 0.0 : type == IntegrationType::TRAPEZOIDAL ? 2.0 / h
                       : type == IntegrationType::GEAR_2 ? 1.5 / h : 1.0 / h;
    const double scale = nl.environment.global_voltage_scale;
    for (const auto &sl : branchList) {
      auto node = [&](int local) { return sl.map(local) - 1; };  // ground -> -1 (skipped)
      auto branch = [&](int b, int np, int nn) {  // KCL coupling + v(+) - v(-) in row b
        add(np, b, 1.0); add(nn, b, -1.0); add(b, np, 1.0); add(b, nn, -1.0);
      };
      const auto &blk = *sl.blk;
      for (size_t i = 0; i < blk.voltageSources.size(); ++i) {
        const auto &v = blk.voltageSources[i];
        const int b = sl.vs + (int)i;
        branch(b, node(v.nodePositive), node(v.nodeNegative));
        S.rhs[b] += scale * sourceValue(v, v.voltage_V, tran ? h : 0.0, time);
      }
      // Inductors (+ mutual coupling K: off-diagonal L_ab terms between branch rows)
      const auto &L = blk.inductors;
      const auto flux = fluxLinks(blk);
      for (size_t a = 0; a < L.size(); ++a) {
        const int b = sl.ind + (int)a;
        branch(b, node(L[a].nodeCoil1), node(L[a].nodeCoil2));  // DC: v1 - v2 = 0 (short)
        if (!tran) continue;
        for (auto [j, Laj] : flux[a]) {
          add(b, sl.ind + j, -coef * Laj);
          const auto &hj = sl.st->inductorState[j];
          double i1 = hj.i.empty() ? 0.0 : hj.i[0];
          double i2 = hj.i.size() > 1 ? hj.i[1] : i1;
          S.rhs[b] += (type == IntegrationType::GEAR_2) ? (Laj / h) * (-2.0 * i1 + 0.5 * i2)
                                                        : -coef * Laj * i1;
        }
        if (type == IntegrationType::TRAPEZOIDAL) {
          const auto &ha = sl.st->inductorState[a];
          S.rhs[b] -= ha.v.empty() ? 0.0 : ha.v[0];
        }
      }
      for (size_t k = 0; k < blk.voltageControlledVoltageSources.size(); ++k) {
        const auto &e = blk.voltageControlledVoltageSources[k];
        const int b = sl.e + (int)k;
        branch(b, node(e.outPos), node(e.outNeg));
        add(b, node(e.ctrlPos), -e.gain);
        add(b, node(e.ctrlNeg), e.gain);
      }
      for (size_t k = 0; k < blk.currentControlledVoltageSources.size(); ++k) {
        const auto &hs = blk.currentControlledVoltageSources[k];
        const int b = sl.h + (int)k;
        branch(b, node(hs.outPos), node(hs.outNeg));
        if (hs.ctrlSource >= 0 && hs.ctrlSource < (int)blk.voltageSources.size())
          add(b, sl.vs + hs.ctrlSource, -hs.rm);
      }
      int bv = sl.bv;
      for (const auto &bs : blk.behavioural)
        if (bs.voltage) branch(bv++, node(bs.outPos), node(bs.outNeg));  // f(x) itself: stampBehav
      for (const auto &f : blk.currentControlledCurrentSources) {
        if (f.ctrlSource < 0 || f.ctrlSource >= (int)blk.voltageSources.size()) continue;
        const int bc = sl.vs + f.ctrlSource;  // alpha * I(ctrl) leaves out+ through the source
        add(node(f.outPos), bc, f.alpha);
        add(node(f.outNeg), bc, -f.alpha);
      }
      // Lossless lines (ngspice TRA): branch currents i1, i2 flow into the + terminals. Transient:
      // v1 - Z0 i1 = w2(t - TD), v2 - Z0 i2 = w1(t - TD) with w = v + Z0 i. DC: a short, v1 = v2, i1 = -i2.
      // AC: the same equations with e^(-jw TD) on the far-port terms (acSystem adds those).
      for (size_t k = 0; k < blk.transmissionLines.size(); ++k) {
        const auto &t = blk.transmissionLines[k];
        const int b1 = sl.tl + 2 * (int)k, b2 = b1 + 1;
        const int p1 = node(t.nodePort1Pos), n1 = node(t.nodePort1Neg), p2 = node(t.nodePort2Pos), n2 = node(t.nodePort2Neg);
        add(p1, b1, 1.0); add(n1, b1, -1.0); add(p2, b2, 1.0); add(n2, b2, -1.0);
        if (tran || acLines_) {
          const double z0 = t.characteristicImpedance_Z0_ohms;
          add(b1, p1, 1.0); add(b1, n1, -1.0); add(b1, b1, -z0);
          add(b2, p2, 1.0); add(b2, n2, -1.0); add(b2, b2, -z0);
          if (tran && k < sl.st->lineState.size()) lineWaves(sl.st->lineState[k], time - t.propagationDelay_Td_s, S.rhs[b2], S.rhs[b1]);
        } else {
          add(b1, p1, 1.0); add(b1, n1, -1.0); add(b1, p2, -1.0); add(b1, n2, 1.0);
          add(b2, b1, 1.0); add(b2, b2, 1.0);
        }
      }
    }
  }
  void stampBranchSystem(const TensorNetlist &nl, double h, double time) {
    BranchSystem S;
    buildBranchSystem(nl, h, time, S);
    for (size_t t = 0; t < S.r.size(); ++t) matrixBuilder.add(S.r[t], S.c[t], S.v[t]);
    for (size_t i = 0; i < S.rhs.size() && i < rhsVector.size(); ++i) rhsVector[i] += S.rhs[i];
  }

  // Behavioural sources: f(x) over the unknowns x (node voltages, branch currents), plus df/dx_col for
  // every column it reads, from one Dual pass per V()/I() reference (the seed is that reference).
  struct BehavEnv {
    static constexpr bool strict = false;
    const std::vector<double> &rv;
    int seed;
    double t;
    double time() const { return t; }
    Dual<double> ref(int k) const { return Dual<double>(rv[k], k == seed ? 1.0 : 0.0); }
  };
  struct BehavEval { double f = 0.0; std::vector<std::pair<int, double>> d; };
  BehavEval behavEval(const Behavioural &b, const BranchSlot &sl, const std::vector<double> &x, double time, bool grad) const {
    const size_t n = b.refs.size();
    auto X = [&](int c) { return c >= 0 && c < (int)x.size() ? x[c] : 0.0; };
    std::vector<int> ca(n, -1), cb(n, -1);
    std::vector<double> rv(n);
    for (size_t k = 0; k < n; ++k) {
      const auto &r = b.refs[k];
      if (r.kind == 0) { ca[k] = sl.map(r.a) - 1; cb[k] = sl.map(r.b) - 1; }  // ground: -1, reads 0
      else ca[k] = r.kind == 1 ? sl.vs + r.a : r.kind == 2 ? sl.ind + r.a : sl.bv + r.a;
      rv[k] = X(ca[k]) - X(cb[k]);
    }
    BehavEnv env{rv, -1, time};
    const acutesim::numparam::Evaluator<BehavEnv> ev{*b.expr, env};
    BehavEval out;
    if (!grad || n == 0) { out.f = ev(b.expr->root).val; return out; }
    for (size_t k = 0; k < n; ++k) {
      env.seed = (int)k;
      const auto r = ev(b.expr->root);
      out.f = r.val;
      if (ca[k] >= 0) out.d.push_back({ca[k], r.grad});
      if (cb[k] >= 0) out.d.push_back({cb[k], -r.grad});
    }
    return out;
  }
  // f(x) of each behavioural source in every block; row: its branch row (voltage sources), else -1
  template <class F> void forEachBehav(const TensorNetlist &nl, F &&f) const {
    int total = 0;
    for (const auto &sl : branchSlots(nl, total)) {
      int row = sl.bv;
      for (const auto &b : sl.blk->behavioural) f(b, sl, b.voltage ? row++ : -1);
    }
  }
  // Newton stamp at x: f ~ f0 + g.(x' - x). A voltage source's row reads v+ - v- - g.x' = f0 - g.x; a
  // current source leaves + (enters -) with g.x' + f0 - g.x.
  void stampBehav(const Behavioural &b, const BranchSlot &sl, int row, const std::vector<double> &x, double time) {
    auto ev = behavEval(b, sl, x, time, true);
    // ponytail: outside a function's domain (ln(0) at the cold start, V = 0) the Newton step sees 0 here;
    // the audit (calculatePhysicalResiduals) keeps the exact value, so a real domain error cannot converge
    if (!std::isfinite(ev.f)) ev.f = 0.0;
    for (auto &[c, g] : ev.d) if (!std::isfinite(g)) g = 0.0;
    double rhs = ev.f;
    for (auto [c, g] : ev.d) rhs -= g * (c < (int)x.size() ? x[c] : 0.0);
    const int np = sl.map(b.outPos) - 1, nn = sl.map(b.outNeg) - 1;
    if (b.voltage) {
      for (auto [c, g] : ev.d) matrixBuilder.add(row, c, -g);
      rhsVector[row] += rhs;
      return;
    }
    for (auto [c, g] : ev.d) {
      if (np >= 0) matrixBuilder.add(np, c, g);
      if (nn >= 0) matrixBuilder.add(nn, c, -g);
    }
    if (np >= 0) rhsVector[np] -= rhs;
    if (nn >= 0) rhsVector[nn] += rhs;
  }

  /**
   * calculatePhysicalResiduals
   * Audits the solution by checking Kirchhoff's Current Law using EXACT
   * non-linear equations. Returns the maximum KCL violation (in Amperes) across
   * all nodes.
   */
  double calculatePhysicalResiduals(const TensorNetlist &netlist,
                                    const std::vector<double> &v_candidate,
                                    double h = 0.0, double time = 0.0,
                                    std::vector<double> *perNodeOut = nullptr) {
    int n = netlist.numGlobalNodes;
    if (n == 0)
      return 0.0;

    // Rows: node KCL (amps) then branch equations (volts)
    std::vector<double> node_residuals(systemSize(netlist), 0.0);

    // helper to get node voltage (Safe Access)
    auto getV = [&](int nodeIdx) {
      if (nodeIdx <= 0 || nodeIdx > (int)v_candidate.size())
        return 0.0;
      return v_candidate[nodeIdx - 1];
    };

    // Helper to accumulate residual (Safe Access)
    auto addRes = [&](int nodeIdx, double val) {
      if (nodeIdx > 0 && nodeIdx <= n)
        node_residuals[nodeIdx - 1] += val;
    };

    // 1. Resistors (Ohm's Law: I = V/R)
    for (const auto &r : netlist.globalBlock.resistors) {
      double v1 = getV(r.nodeTerminal1);
      double v2 = getV(r.nodeTerminal2);
      double i_r = (v1 - v2) / r.resistance_ohms;
      addRes(r.nodeTerminal1, -i_r); // Current LEAVING node
      addRes(r.nodeTerminal2, i_r);  // Current ENTERING node
    }

    // 1.5 The transient OP's node gmin (addDiagonalConditioning(ptranGmin_)): the audit must
    // see the current the matrix holds, or a node at volts never passes KCL there (10 V at
    // 1e-12 S is 10 pA: a floating node behind a 1 ms RC failed every step at t = 0)
    if (ptranGmin_ + gshunt_ > 0.0)
      for (int i = 1; i <= n; ++i) addRes(i, -(ptranGmin_ + gshunt_) * getV(i));

    // 2. V sources, inductors, E/F/H: branch-current elements, audited from
    //    the same triplets they are stamped with (see buildBranchSystem below)

    // 2.5. Current Sources (Ideal Current Injection)
    // Current sources inject fixed current regardless of voltage
    for (const auto &cs : netlist.globalBlock.currentSources) {
      double is = sourceValue(cs, cs.current_A, h, time);
      // Current flows from + to - inside source, leaves + externally
      addRes(cs.nodePositive, -is); // Current leaving + node
      addRes(cs.nodeNegative, is);  // Current entering - node
    }

    // 2.7 Dependent Sources (Global): G only; E/F/H are branch elements
    for (const auto &vccs : netlist.globalBlock.voltageControlledCurrentSources) {
        double v_ctrl = getV(vccs.ctrlPos) - getV(vccs.ctrlNeg);
        double i_vccs = vccs.gm * v_ctrl;
        addRes(vccs.outPos, -i_vccs);
        addRes(vccs.outNeg, i_vccs);
    }

    // 3.55 MOSFETs, global block and every instance: the stamped mosCurrents()
    forEachMosfet(netlist, [&](const Mosfet &m, const std::array<int, 4> &t,
                               const BlockState::DeviceCharge *st) {
        const auto I = mosCurrents<double>({getV(t[0]), getV(t[1]), getV(t[2]), getV(t[3])},
                                           mos1Params(m, gmin_), st, h);
        for (int k = 0; k < 4; ++k) addRes(t[k], -I[k]);
    });
    // 3.6 Diodes and BJTs, global block and every instance: the stamped
    //     junctionCurrents(), including their internal-node rows
    forEachJunction(netlist, [&](const Junction &j) {
        std::array<double, 7> x{};
        for (int k = 0; k < j.terms; ++k) x[k] = getV(j.n[k]);
        const auto I = junctionCurrents(j, x, h);
        for (int k = 0; k < j.terms; ++k)
          if (j.n[k] > 0 && j.n[k] <= (int)node_residuals.size()) node_residuals[j.n[k] - 1] -= I[k];
    });
    // 3.62 Switches: the stamped state's conductance; a state the iterate would change is not converged
    forEachSwitch(netlist, [&](const SwitchRef &w, const std::array<int, 2> &c) {
        const double i = (getV(w.n[0]) - getV(w.n[1])) / (*w.on ? w.s->ron : w.s->roff);
        addRes(w.n[0], -i);
        addRes(w.n[1], i);
        if (switchState(w, c, v_candidate) != (*w.on != 0)) addRes(w.n[0], 1.0);
    });
    // 3.65 OSDI devices: the stamped osdiCurrents(), internal-node rows included
    osdiAll(netlist, v_candidate, h, false, false);
    for (size_t i = 0; i < osdiRefs_.size(); ++i) {
        const auto &r = osdiRefs_[i];
        for (size_t k = 0; k < r.n.size(); ++k)
          if (r.n[k] > 0 && r.n[k] <= (int)node_residuals.size()) node_residuals[r.n[k] - 1] -= osdiI_[i][k];
    }
    
    // 3.7 Hierarchical Instances — audited in Section 5 (parallel scheduler path)
    
    // 4. Capacitors (Transient Only)
    // If h > 0, we are in transient and must audit the displacement current
    if (h > 1e-15 && integrator) {
      for (size_t i = 0; i < netlist.globalBlock.capacitors.size(); ++i) {
        const auto &c = netlist.globalBlock.capacitors[i];
        const auto &hist = netlist.globalState.capacitorState[i];
          
        double v_curr = getV(c.nodePlate1) - getV(c.nodePlate2);
        double i_c = 0.0;
        IntegrationType type = integrator->getType();

        if (hist.v.empty())
          continue;

        if (type == IntegrationType::TRAPEZOIDAL) {
          double g_eq = (2.0 * c.capacitance_farads) / h;
          i_c = g_eq * (v_curr - hist.v[0]) - hist.i[0];
        } else if (type == IntegrationType::GEAR_1_EULER) {
          i_c = (c.capacitance_farads / h) * (v_curr - hist.v[0]);
        } else if (type == IntegrationType::GEAR_2) {
          if (hist.v.size() >= 2) {
            i_c = (c.capacitance_farads / h) *
                  (1.5 * v_curr - 2.0 * hist.v[0] +
                   0.5 * hist.v[1]);
          } else {
            i_c = (c.capacitance_farads / h) * (v_curr - hist.v[0]);
          }
        }
        addRes(c.nodePlate1, -i_c);
        addRes(c.nodePlate2, i_c);
      }
      
    }

    // 5. Hierarchical Instances
    // Sequential accumulation required for bit-identical determinism.
    // Parallel atomic updates are non-deterministic due to non-associative float addition.
    for (int i = 0; i < (int)scheduler.jobs.size(); ++i) {
      const auto &job = scheduler.jobs[i];
      const TensorBlock &block = *job.blockTmpl;
      for (int instanceIdx : job.instanceIds) {
        const auto &instance = netlist.instances[instanceIdx];
        auto mapNode = [&](int localNode) {
          return (localNode == 0) ? 0 : instance.nodeMapping[localNode - 1];
        };

        // Resistors
        // Resistors
        for (const auto &r : block.resistors) {
          double i_r = (getV(mapNode(r.nodeTerminal1)) -
                        getV(mapNode(r.nodeTerminal2))) /
                       r.resistance_ohms;
          addRes(mapNode(r.nodeTerminal1), -i_r);
          addRes(mapNode(r.nodeTerminal2), i_r);
        }

        // Current sources inside the subcircuit (V sources are branch elements)
        for (const auto &cs : block.currentSources) {
          double is = sourceValue(cs, cs.current_A, h, time);
          addRes(mapNode(cs.nodePositive), -is);
          addRes(mapNode(cs.nodeNegative), is);
        }

        // Dependent Sources: G only (E/F/H are branch elements)
        for (const auto &vccs : block.voltageControlledCurrentSources) {
            double v_ctrl = getV(mapNode(vccs.ctrlPos)) - getV(mapNode(vccs.ctrlNeg));
            double i_vccs = vccs.gm * v_ctrl;
            addRes(mapNode(vccs.outPos), -i_vccs);
            addRes(mapNode(vccs.outNeg), i_vccs);
        }

        // Capacitors (Transient)
        if (h > 1e-15 && integrator) {
          const auto &instState = netlist.instanceStates[instanceIdx];
          for (size_t c_idx = 0; c_idx < block.capacitors.size(); ++c_idx) {
            const auto &c = block.capacitors[c_idx];
            const auto &hist = instState.capacitorState[c_idx];

            double v_curr =
                getV(mapNode(c.nodePlate1)) - getV(mapNode(c.nodePlate2));
            double i_c = 0.0;
            IntegrationType type = integrator->getType();
            if (hist.v.empty())
              continue;

            if (type == IntegrationType::TRAPEZOIDAL) {
              double g_eq = (2.0 * c.capacitance_farads) / h;
              i_c = g_eq * (v_curr - hist.v[0]) - hist.i[0];
            } else if (type == IntegrationType::GEAR_1_EULER) {
              i_c = (c.capacitance_farads / h) * (v_curr - hist.v[0]);
            } else if (type == IntegrationType::GEAR_2) {
              if (hist.v.size() >= 2) {
                i_c = (c.capacitance_farads / h) *
                      (1.5 * v_curr - 2.0 * hist.v[0] +
                       0.5 * hist.v[1]);
              } else {
                i_c = (c.capacitance_farads / h) * (v_curr - hist.v[0]);
              }
            }
            addRes(mapNode(c.nodePlate1), -i_c);
            addRes(mapNode(c.nodePlate2), i_c);
          }
        }
      }
    }

    // Branch elements: residual = rhs - A x over exactly the stamped triplets
    {
      BranchSystem S;
      buildBranchSystem(netlist, h, time, S);
      auto xAt = [&](int c) { return c < (int)v_candidate.size() ? v_candidate[c] : 0.0; };
      for (size_t t = 0; t < S.r.size(); ++t) node_residuals[S.r[t]] -= S.v[t] * xAt(S.c[t]);
      for (size_t i = 0; i < S.rhs.size(); ++i) node_residuals[i] += S.rhs[i];
    }

    // Behavioural sources: the exact f(x), not the linearisation just stamped
    forEachBehav(netlist, [&](const Behavioural &b, const BranchSlot &sl, int row) {
      const double f = behavEval(b, sl, v_candidate, time, false).f;
      if (b.voltage) { node_residuals[row] += f; return; }
      addRes(sl.map(b.outPos), -f);
      addRes(sl.map(b.outNeg), f);
    });

    // 6. Find Max Residual
    double max_err = 0.0;
    for (double val : node_residuals) {
      max_err = std::max(max_err, std::abs(val));
    }
    if (perNodeOut) *perNodeOut = std::move(node_residuals);
    return max_err;
  }

  /**
   * kclExcess — the NR convergence metric: KCL error beyond each node's
   * floating-point noise floor, max_i max(0, |r_i| - KCL_RELTOL*(|A||v|)_i),
   * where A is the Jacobian just solved. Raw max|r_i| made correct solutions
   * with stiff stamps (ideal op-amps) report "not converged".
   */
  double kclExcess(const TensorNetlist &netlist, const std::vector<double> &v,
                   const Csr_matrix &A, double h = 0.0, double time = 0.0) {
    std::vector<double> r;
    calculatePhysicalResiduals(netlist, v, h, time, &r);
    double worst = 0.0;
    for (int i = 0; i < (int)r.size(); ++i) {
      if (!std::isfinite(r[i])) return INFINITY;  // NaN must fail, not vanish in max()
      double scale = 0.0;
      if (i < A.rows)
        for (int k = A.row_pointer[i]; k < A.row_pointer[i + 1]; ++k)
          scale += std::abs(A.values[k]) * std::abs(v[A.col_indices[k]]);
      worst = std::max(worst, std::abs(r[i]) - PhysicsConstants::KCL_RELTOL * scale);
    }
    return worst;
  }

  /**
   * calculatePhysicalResiduals (per-node overload) — Phase 2.9
   * Like the scalar overload but also populates perNodeResiduals with the
   * raw KCL violation at each node. Used for RMS/worst-node computation.
   * Calling this on every NR iteration is expensive — invoke only at final step.
   */
  double calculatePhysicalResiduals(const TensorNetlist &netlist,
                                    const std::vector<double> &v_candidate,
                                    std::vector<double> &perNodeResiduals,
                                    double h = 0.0) {
    // Same audit as the scalar overload (it used to be a partial re-implementation)
    return calculatePhysicalResiduals(netlist, v_candidate, h, 0.0, &perNodeResiduals);
  }

  /**
   * calculatePhysicalResidualsDetailed
   * Like calculatePhysicalResiduals but additionally records per-device
   * attribution for the worst KCL violations. Returns max KCL error.
   * Populates worstDevices with sorted contributions to the worst node.
   */
  double calculatePhysicalResidualsDetailed(
      const TensorNetlist &netlist,
      const std::vector<double> &v_candidate,
      std::vector<ResidualContribution> &worstDevices,
      int &worstNodeOut) {

    int n = netlist.numGlobalNodes;
    worstDevices.clear();
    worstNodeOut = 0;
    if (n == 0) return 0.0;

    std::vector<double> node_residuals(n, 0.0);

    auto getV = [&](int nodeIdx) {
      if (nodeIdx <= 0 || nodeIdx > (int)v_candidate.size()) return 0.0;
      return v_candidate[nodeIdx - 1];
    };
    auto addRes = [&](int nodeIdx, double val) {
      if (nodeIdx > 0 && nodeIdx <= n)
        node_residuals[nodeIdx - 1] += val;
    };

    // Per-device attribution: (deviceName, deviceType, affectedNodes[], current)
    struct DeviceAttribution {
        std::string name;
        std::string type;
        std::vector<std::pair<int, double>> nodeCurrents; // (1-based node, current contributed)
    };
    std::vector<DeviceAttribution> attributions;

    // 1. Resistors
    for (const auto &r : netlist.globalBlock.resistors) {
      double v1 = getV(r.nodeTerminal1);
      double v2 = getV(r.nodeTerminal2);
      double i_r = (v1 - v2) / r.resistance_ohms;
      addRes(r.nodeTerminal1, -i_r);
      addRes(r.nodeTerminal2, i_r);
      attributions.push_back({r.name, "Resistor",
          {{r.nodeTerminal1, -i_r}, {r.nodeTerminal2, i_r}}});
    }

    // 2. Branch elements (V, L, E, F, H): node-row terms of the shared branch
    //    system; their currents are unknowns in v_candidate (DC: h = 0)
    {
      BranchSystem S;
      buildBranchSystem(netlist, 0.0, 0.0, S);
      for (size_t t = 0; t < S.r.size(); ++t)
        if (S.r[t] < n && S.c[t] < (int)v_candidate.size())
          node_residuals[S.r[t]] -= S.v[t] * v_candidate[S.c[t]];
    }

    // Current Sources
    for (const auto &cs : netlist.globalBlock.currentSources) {
      const double is = sourceValue(cs, cs.current_A, 0.0, 0.0);
      addRes(cs.nodePositive, -is);
      addRes(cs.nodeNegative, is);
      attributions.push_back({cs.type.empty() ? "I?" : cs.type, "CurrentSource",
          {{cs.nodePositive, -is}, {cs.nodeNegative, is}}});
    }


    // 3. Diodes and BJTs (every block): DC junctionCurrents() at the external
    //    nodes (internal-node rows are not node KCL rows here)
    forEachJunction(netlist, [&](const Junction &j) {
      std::array<double, 7> x{};
      for (int k = 0; k < j.terms; ++k) x[k] = getV(j.n[k]);
      const auto I = junctionCurrents(j, x, 0.0);
      DeviceAttribution a{j.d ? j.d->instanceName : j.f ? j.f->instanceName : j.q->instanceName, j.d ? "Diode" : j.f ? "JFET" : "BJT", {}};
      for (int k = 0; k < j.terms; ++k) {
        addRes(j.n[k], -I[k]);
        if (j.n[k] > 0 && j.n[k] <= n) a.nodeCurrents.push_back({j.n[k], -I[k]});
      }
      attributions.push_back(std::move(a));
    });
    // OSDI devices, likewise (DC, external nodes)
    osdiAll(netlist, v_candidate, 0.0, false, false);
    for (size_t i = 0; i < osdiRefs_.size(); ++i) {
      const auto &r = osdiRefs_[i];
      const auto &I = osdiI_[i];
      DeviceAttribution a{r.dev->instanceName, "OSDI", {}};
      for (size_t k = 0; k < r.dev->nodes.size(); ++k) {
        addRes(r.n[k], -I[k]);
        if (r.n[k] > 0 && r.n[k] <= n) a.nodeCurrents.push_back({r.n[k], -I[k]});
      }
      attributions.push_back(std::move(a));
    }

    // MOSFETs
    for (const auto &m : netlist.globalBlock.mosfets) {
      const auto c = mos1_currents(getV(m.drain), getV(m.gate), getV(m.source), getV(m.body),
                                   mos1Params(m, gmin_));
      addRes(m.drain, -c.d);
      addRes(m.source, -c.s);
      addRes(m.body, -c.b);
      attributions.push_back({m.instanceName, "MOSFET",
          {{m.drain, -c.d}, {m.source, -c.s}, {m.body, -c.b}}});
    }

    // Find worst node
    double max_err = 0.0;
    int worst_idx = 0;
    for (int i = 0; i < n; ++i) {
      double abs_val = std::abs(node_residuals[i]);
      if (abs_val > max_err) {
        max_err = abs_val;
        worst_idx = i;
      }
    }
    worstNodeOut = worst_idx + 1; // 1-based

    // Build per-device attribution for worst node
    for (const auto &attr : attributions) {
      for (const auto &[node, current] : attr.nodeCurrents) {
        if (node == worstNodeOut && std::abs(current) > 1e-15) {
          ResidualContribution rc;
          rc.deviceName = attr.name;
          rc.deviceType = attr.type;
          rc.nodeIndex = node;
          rc.currentContribution = current;
          rc.percentOfTotal = (max_err > 0.0) ? (std::abs(current) / max_err * 100.0) : 0.0;
          worstDevices.push_back(rc);
        }
      }
    }

    // Sort by contribution magnitude (descending)
    std::sort(worstDevices.begin(), worstDevices.end(),
              [](const ResidualContribution &a, const ResidualContribution &b) {
                return std::abs(a.currentContribution) > std::abs(b.currentContribution);
              });

    return max_err;
  }

  // Phase 2.5: Differentiable Physics
  enum class ComponentType { RESISTOR, CAPACITOR, INDUCTOR, DIODE, MOSFET, BJT, JFET, VSOURCE, ISOURCE };

  // b = -dF/dp (size nTot) of one instance parameter at the converged point, F the DC residual (node KCL
  // "current leaving", then the branch equations). Empty if the parameter is not implemented.
  std::vector<double> sensRhs(TensorNetlist &netlist, ComponentType type, int index, const std::string& paramName,
                              const std::vector<double>& v_converged, int nTot) {
      std::vector<double> b_sens(nTot, 0.0);
      
      auto getV = [&](int n) { return (n>0 && n<=(int)v_converged.size()) ? v_converged[n-1] : 0.0; };

      try {
          if (type == ComponentType::VSOURCE) {  // F_branch = v+ - v- - V
              if (index < 0 || index >= (int)netlist.globalBlock.voltageSources.size()) return {};
              int total = 0;
              b_sens[branchSlots(netlist, total)[0].vs + index] = 1.0;
          }
          else if (type == ComponentType::ISOURCE) {  // the source current leaves n+ and enters n-
              if (index < 0 || index >= (int)netlist.globalBlock.currentSources.size()) return {};
              const auto& i = netlist.globalBlock.currentSources[index];
              if (i.nodePositive > 0) b_sens[i.nodePositive - 1] -= 1.0;
              if (i.nodeNegative > 0) b_sens[i.nodeNegative - 1] += 1.0;
          }
          else if (type == ComponentType::RESISTOR) {
              if (index < 0 || index >= (int)netlist.globalBlock.resistors.size()) return {};
              const auto& r = netlist.globalBlock.resistors[index];
              
              // Only 'resistance' is supported for now
              if (paramName != "resistance" && paramName != "R") return {};
              
              // I_leaving_n1 = (V1 - V2) / R
              // dI/dR = -(V1 - V2) / R^2
              double v_diff = getV(r.nodeTerminal1) - getV(r.nodeTerminal2);
              double R = r.resistance_ohms;
              double didp = -v_diff / (R * R);
              
              // F_n1 += I_leaving. dF_n1/dp = dI/dp
              // b[n1] = -dF/dp = -dI/dp
              if (r.nodeTerminal1 > 0) b_sens[r.nodeTerminal1-1] -= didp;
              if (r.nodeTerminal2 > 0) b_sens[r.nodeTerminal2-1] += didp; // I_leaving_n2 is -I -> d(-I)/dp = -dI/dp -> b = +dI/dp
          } 
          else if (type == ComponentType::DIODE || type == ComponentType::BJT) {
              // dI/dp of the same junctionCurrents() the solve used, by central
              // difference on the instance's parameter (global-block devices).
              // ponytail: finite difference (relative step 1e-6); Dual parameters if precision matters.
              const bool isD = type == ComponentType::DIODE;
              const size_t count = isD ? netlist.globalBlock.diodes.size() : netlist.globalBlock.bjts.size();
              if (index < 0 || index >= (int)count) return {};
              Diode d; BJT q;
              double *par = nullptr;
              const void *orig = nullptr;
              if (isD) {
                  d = netlist.globalBlock.diodes[index]; orig = &netlist.globalBlock.diodes[index];
                  par = paramName == "Is" ? &d.saturationCurrent_I_S_A : paramName == "N" ? &d.emissionCoefficient_N
                      : paramName == "Vt" ? &d.thermalVoltage_V_T_V : nullptr;
              } else {
                  q = netlist.globalBlock.bjts[index]; orig = &netlist.globalBlock.bjts[index];
                  par = paramName == "Is" ? &q.saturationCurrent_I_S_A
                      : (paramName == "BetaF" || paramName == "Beta") ? &q.betaF
                      : paramName == "BetaR" ? &q.betaR : paramName == "Vt" ? &q.thermalVoltage_V_T_V : nullptr;
              }
              if (!par) return {};
              Junction jj;
              forEachJunction(netlist, [&](const Junction &j) { if ((const void *)(isD ? (const void *)j.d : (const void *)j.q) == orig) jj = j; });
              jj.d = isD ? &d : nullptr; jj.q = isD ? nullptr : &q; jj.st = nullptr;
              std::array<double, 7> x{};
              for (int k = 0; k < jj.terms; ++k) x[k] = getV(jj.n[k]);
              const double p0 = *par, dp = 1e-6 * std::max(std::abs(p0), 1e-30);
              *par = p0 + dp; const auto Ip = junctionCurrents(jj, x, 0.0);
              *par = p0 - dp; const auto Im = junctionCurrents(jj, x, 0.0);
              for (int k = 0; k < jj.terms; ++k)
                  if (jj.n[k] > 0 && jj.n[k] <= (int)b_sens.size()) b_sens[jj.n[k] - 1] -= (Ip[k] - Im[k]) / (2.0 * dp);
          }
          else if (type == ComponentType::MOSFET) {
              if (index < 0 || index >= (int)netlist.globalBlock.mosfets.size()) return {};
              const auto& m = netlist.globalBlock.mosfets[index];

              // dI/dp of the same mos1_currents() the solve used: Vth (VTO), W, L
              const Mos1Params<double> p = mos1Params(m, gmin_);
              if (p.level != 1) return {};  // ponytail: levels 2/3 have no parameter sensitivity
              Mos1Params<Dual<double>> pd{p.polarity, p.beta, p.vto, p.lambda, p.gamma, p.phi, p.isat, p.vt};
              const double leff = m.l - 2.0 * m.card.ld;
              if (paramName == "Vth") pd.vto.grad = 1.0;
              if (paramName == "W") pd.beta.grad = p.beta / m.w;
              if (paramName == "L") pd.beta.grad = -p.beta / leff;
              const auto c = mos1_currents(Dual<double>(getV(m.drain)), Dual<double>(getV(m.gate)),
                                           Dual<double>(getV(m.source)), Dual<double>(getV(m.body)), pd);
              if (m.drain > 0) b_sens[m.drain-1] -= c.d.grad;
              if (m.source > 0) b_sens[m.source-1] -= c.s.grad;
              if (m.body > 0) b_sens[m.body-1] -= c.b.grad;
          }
          // CAPACITOR SENSITIVITY (DC: dI/dC = 0, AC needs omega)
          else if (type == ComponentType::CAPACITOR) {
              if (index < 0 || index >= (int)netlist.globalBlock.capacitors.size()) return {};
              // DC: Capacitors are open circuit, no current, so dI/dC = 0.
              // For AC sensitivity, user would need to provide omega. Placeholder for now.
          }
          // INDUCTOR SENSITIVITY (DC: dI/dL = 0, AC needs omega)
          else if (type == ComponentType::INDUCTOR) {
              if (index < 0 || index >= (int)netlist.globalBlock.inductors.size()) return {};
              // DC: Inductors are short, no L term in DC model. dI/dL = 0.
          }
      } catch(...) {
          return {};
      }
      return b_sens;
  }

  /**
   * calculateSensitivity
   * Computes the gradient of node voltages with respect to a specific component parameter (dV/dp).
   * Solves: J * (dV/dp) = - (dF/dp)
   */
  std::vector<double> calculateSensitivity(TensorNetlist &netlist, 
                                           ComponentType type, 
                                           int index, 
                                           const std::string& paramName, 
                                           const std::vector<double>& v_converged) {
      if (v_converged.empty()) return {};

      // 1. Re-build Jacobian (J) at convergence point
      // We assume standard stamping (double params) gives us the correct J = dF/dV
      scheduler.schedule(netlist);
      matrixBuilder.clear();
      const int nTot = systemSize(netlist);  // nodes + branch currents
      matrixBuilder.setDimensions(nTot, nTot);
      rhsVector.assign(nTot, 0.0);
      std::vector<double> x_converged = v_converged;
      if ((int)lastSolution_.size() == nTot) x_converged = lastSolution_;  // with branch currents
      x_converged.resize(nTot, 0.0);
      limitJunctions_ = false;  // linearise at the converged point itself
      stampAllElements(netlist, 0.0, x_converged, 0.0); // h=0, time=0 for DC sensitivity
      limitJunctions_ = true;
      // (no node gmin: the Jacobian of the converged OP has only junction GMIN)
      Csr_matrix J = matrixBuilder.createCsr(); /* Explicitly J */
      
      // 2. RHS vector b = - dF/dp
      std::vector<double> b_sens = sensRhs(netlist, type, index, paramName, v_converged, nTot);
      if (b_sens.empty()) return {};

      // 3. Solve J * x = b
      // J is nonsymmetric (PCG assumed symmetry); solve exactly, return node rows
      SolverResult res = solveLinear(J, b_sens);
      res.solution.resize(netlist.numGlobalNodes);
      return res.solution;
  }

  // Phase 2: Numerical Integration Strategy




  /**
   * stampBlockNonLinear
   * Linearizes and stamps non-linear elements (Diodes, MOSFETs, BJTs) for a specific
   * block instance.
   * Uses Forward-Mode Automatic Differentiation (AD) via Dual<double>.
   */
  /**
   * stampSoABlock
   * Uses Structure-of-Arrays (SoA) tensors for batch processing.
   * Calls specialized SIMD-friendly physics kernels and then stamps the results.
   */
  void stampSoABlock(TensorizedBlock& block, const std::vector<double>& v_guess, double h = 0.0,
                     IntegrationType intType = IntegrationType::TRAPEZOIDAL) {
    // 0. Linear Components
    
    // Resistors
    size_t nR = block.resistors.size();
    if (nR > 0) {
        std::vector<int> rows, cols;
        std::vector<double> vals;
        rows.reserve(nR * 4); cols.reserve(nR * 4); vals.reserve(nR * 4);
        for (size_t i = 0; i < nR; ++i) {
            int n1 = block.resistors.nodes1[i];
            int n2 = block.resistors.nodes2[i];
            double g = 1.0 / block.resistors.R[i];
            if (n1 != 0 && n2 != 0) {
                rows.push_back(n1-1); cols.push_back(n1-1); vals.push_back(g);
                rows.push_back(n2-1); cols.push_back(n2-1); vals.push_back(g);
                rows.push_back(n1-1); cols.push_back(n2-1); vals.push_back(-g);
                rows.push_back(n2-1); cols.push_back(n1-1); vals.push_back(-g);
            } else if (n1 != 0) {
                rows.push_back(n1-1); cols.push_back(n1-1); vals.push_back(g);
            } else if (n2 != 0) {
                rows.push_back(n2-1); cols.push_back(n2-1); vals.push_back(g);
            }
        }
        matrixBuilder.addBatch(rows, cols, vals);
    }

    // V sources / inductors: branch-current elements (stampBranchSystem)

    // Current Sources
    size_t nIS = block.currentSources.size();
    for (size_t i = 0; i < nIS; ++i) {
        int nP = block.currentSources.nodesPos[i];
        int nN = block.currentSources.nodesNeg[i];
        double I = srcFact_ * block.currentSources.currents[i];  // (refreshed to the waveform in transient)
        if (nP != 0) rhsVector[nP-1] -= I;
        if (nN != 0) rhsVector[nN-1] += I;
    }

    // Capacitors (Transient)
    size_t nC = block.capacitors.size();
    if (nC > 0 && h > 0.0) {
        std::vector<int> rows, cols;
        std::vector<double> vals;
        rows.reserve(nC * 4); cols.reserve(nC * 4); vals.reserve(nC * 4);
        for (size_t i = 0; i < nC; ++i) {
            int n1 = block.capacitors.plates1[i];
            int n2 = block.capacitors.plates2[i];
            double g_eq, i_hist;
            // GEAR_1_EULER (Backward Euler): g=C/h, no i_prev term.
            // GEAR_2 (BDF2): g=1.5C/h, I=C(2v1-0.5v2)/h — must match the KCL
            //   audit and history update; the old BE fallback here made every
            //   transient that switched to Gear2 fail NR at all step sizes.
            // TRAPEZOIDAL (default): g=2C/h, companion current uses i_prev.
            if (intType == IntegrationType::GEAR_1_EULER) {
                g_eq   = block.capacitors.C[i] / h;
                i_hist = g_eq * block.capacitors.v_prev[i];
            } else if (intType == IntegrationType::GEAR_2) {
                g_eq   = 1.5 * block.capacitors.C[i] / h;
                i_hist = block.capacitors.C[i] / h *
                         (2.0 * block.capacitors.v_prev[i] - 0.5 * block.capacitors.v_prev2[i]);
            } else {
                g_eq   = (2.0 * block.capacitors.C[i]) / h;
                i_hist = g_eq * block.capacitors.v_prev[i] + block.capacitors.i_prev[i];
            }

            if (n1 != 0 && n2 != 0) {
                rows.push_back(n1-1); cols.push_back(n1-1); vals.push_back(g_eq);
                rows.push_back(n2-1); cols.push_back(n2-1); vals.push_back(g_eq);
                rows.push_back(n1-1); cols.push_back(n2-1); vals.push_back(-g_eq);
                rows.push_back(n2-1); cols.push_back(n1-1); vals.push_back(-g_eq);
            } else if (n1 != 0) {
                rows.push_back(n1-1); cols.push_back(n1-1); vals.push_back(g_eq);
            } else if (n2 != 0) {
                rows.push_back(n2-1); cols.push_back(n2-1); vals.push_back(g_eq);
            }
            if (n1 != 0) rhsVector[n1-1] += i_hist;
            if (n2 != 0) rhsVector[n2-1] -= i_hist;
        }
        matrixBuilder.addBatch(rows, cols, vals);
    }
    // Diodes, BJTs and MOSFETs: stampAllElements() (stampByDual, both paths)


  }


  /**
   * stampAllElements
   * High-level entry point for matrix assembly.
   */
  void stampAllElements(const TensorNetlist &netlist, double h,
                        const std::vector<double> &v_guess, double time) {
    {
        // SoA Pathway (Flattened). tensorizeNetlist() stores DC values only, so
        // in transient re-evaluate every source at `time`, walking blocks in the
        // same order tensorizeNetlist() flattens them (global, then instances).
        if (h > 1e-15) {
            size_t ki = 0;
            auto refresh = [&](const TensorBlock &blk) {
                for (const auto &i : blk.currentSources)
                    if (ki < soaBlock.currentSources.currents.size())
                        soaBlock.currentSources.currents[ki++] = waveformValue(i, i.current_A, time);
            };
            refresh(netlist.globalBlock);
            for (const auto &inst : netlist.instances) {
                auto it = netlist.blockDefinitions.find(inst.blockName);
                if (it != netlist.blockDefinitions.end()) refresh(it->second);
            }
        }
        stampSoABlock(soaBlock, v_guess, h, integrator ? integrator->getType() : IntegrationType::TRAPEZOIDAL);

        // Devices tensorizeNetlist() has no SoA tensors for (G, J) were
        // silently dropped here. Stamp them with the same helpers the AoS path
        // uses, so both paths share one implementation.
        auto getV = [&](int n) { return (n > 0 && n <= (int)v_guess.size()) ? v_guess[n - 1] : 0.0; };
        auto stampUncovered = [&](const TensorBlock &blk, const std::function<int(int)> &mapNode) {
            for (const auto &g : blk.voltageControlledCurrentSources)
                stampVCCS(mapNode(g.ctrlPos), mapNode(g.ctrlNeg), mapNode(g.outPos), mapNode(g.outNeg),
                          g.gm, matrixBuilder, rhsVector);
        };
        stampUncovered(netlist.globalBlock, [](int n) { return n; });
        for (const auto &inst : netlist.instances) {
            auto it = netlist.blockDefinitions.find(inst.blockName);
            if (it == netlist.blockDefinitions.end()) continue;
            stampUncovered(it->second, [&](int n) {  // same remap as tensorizeNetlist()
                return (n <= 0 || n > (int)inst.nodeMapping.size()) ? 0 : inst.nodeMapping[n - 1];
            });
        }
    }

    // Branch-current elements (V, L+K, E, F, H), MOSFETs, diodes and BJTs: one implementation for both paths
    stampBranchSystem(netlist, h, time);
    forEachBehav(netlist, [&](const Behavioural &b, const BranchSlot &sl, int row) { stampBehav(b, sl, row, v_guess, time); });
    forEachMosfet(netlist, [&](const Mosfet &m, const std::array<int, 4> &n,
                               const BlockState::DeviceCharge *st) { stampMos1(n, mos1Params(m, gmin_), st, h, v_guess); });
    forEachJunction(netlist, [&](const Junction &j) { stampJunction(j, v_guess, h); });
    forEachSwitch(netlist, [&](const SwitchRef &w, const std::array<int, 2> &c) {
      *w.on = !swAcOff_ && switchState(w, c, v_guess);
      stampResistor(w.n[0], w.n[1], *w.on ? w.s->ron : w.s->roff, matrixBuilder);
    });
    osdiAll(netlist, v_guess, h, true, false);  // stamped in device order: the same sequence every iteration
    for (size_t i = 0; i < osdiRefs_.size(); ++i) stampOsdi(osdiRefs_[i], osdiI_[i], osdiJ_[i], v_guess);

    // A node no element touches is not a circuit node (ngspice drops it after
    // flattening, e.g. an X pin wired to a port the .subckt never uses): pin it
    // to 0 V. A real floating node (series caps, an undriven gate) still fails.
    const std::vector<char> used = usedNodes(netlist);
    for (int n = 1; n <= netlist.numGlobalNodes; ++n)
      if (!used[n]) matrixBuilder.add(n - 1, n - 1, 1.0);
  }


  void addDiagonalConditioning(int n, double gmin) {
    // gmin <= 0: no node-to-ground conditioning. Like ngspice, once converged
    // the only gmin is the 1e-12 S across each junction (device models); a
    // node-to-ground leak settles floating junction nodes differently
    // (aac_10_bridge). Mid-ladder DC steps still pass gmin > 0 for conditioning.
    gmin += gshunt_;  // .options rshunt: a resistor from every node to ground, in every analysis
    if (gmin <= 0.0) return;
    for (int i = 0; i < n; ++i)
      matrixBuilder.add(i, i, gmin);
  }


  // Placeholder for audit logic (implemented in audit file)
#include "circuitsim_audit.inc"
};

// =============================================================================
// SOLVER IMPLEMENTATIONS
// =============================================================================
// solveDC and stepTransient are implemented in circuitsim_solver.cpp
// to reduce header size and improve compile times.
