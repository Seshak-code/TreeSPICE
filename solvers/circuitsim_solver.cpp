/**
 * circuitsim_solver.cpp
 *
 * Implementation of CircuitSim solver methods:
 * - solveDC: DC Operating Point Analysis (Newton-Raphson)
 * - stepTransient: Transient Analysis (Adaptive Time-Stepping)
 *
 * Extracted from circuitsim.h for modularity.
 *
 * Phase 1 Hardening: Refactored for NASA-style safety.
 */

#include <sstream>

#include "../infrastructure/safe_utils.h"
#include "../math/pole_zero.h"
#include "../physics/circuitsim.h"
#include "../physics/device_physics.h"
#include "../physics/physics_constants.h"
#include "../solvers/convergence_engine.h"
#if !defined(__EMSCRIPTEN__) && defined(ACUTESIM_HAS_DAWN_NATIVE)
#include "gpu_context_manager.h"
#endif

// Matches SPICE3's dt_init = tstep/100: divides the first transient step into
// 100 internal Backward Euler micro-steps for cold-start accuracy.
static constexpr int COLD_START_SUBSTEPS = 100;

// =============================================================================
// MAIN SOLVER METHODS
// =============================================================================

static std::string lower(std::string s) {
  for (auto &c : s) c = (char)std::tolower((unsigned char)c);
  return s;
}

SolverStep CircuitSim::solveDC(TensorNetlist &netlist) {
  if (netlist.numGlobalNodes == 0)
    return {0.0, {}, {0, 0.0, true}, 0};
  forEachSwitch(netlist, [&](const SwitchRef &w, const std::array<int, 2> &) { *w.on = *w.prev; });

  // GPU mode currently solves on the CPU path: the former WebGPU "hybrid" loop
  // stamped and solved on the CPU anyway with its own copy of the device
  // physics (voltage-only, no branch currents). Phase 4 replaces it with GPU
  // device evaluation + batched front contraction.

  // .nodeset (ngspice): a first solve holds the nodes at their values (1e10 S, as .ic does), then the
  // circuit is solved from that point with them released. Only a cold start: a .dc sweep's later points
  // continue from the previous one.
  if (!netlist.nodeSets.empty() && prevVoltages_.empty()) {
    TensorNetlist held = netlist;
    held.nodeSets.clear();
    for (const auto &[n, v] : netlist.nodeSets) {
      held.addResistor(n, 0, 1e-10);
      held.addCurrentSource(0, n, v * 1e10);
    }
    if (solveDC(held).stats.converged) prevVoltages_ = lastSolution_;
  }

  struct RestoreSources { double &f; ~RestoreSources() { f = 1.0; } } restoreSources{srcFact_};
  osdiFatal_.clear();
  osdiNonFinite_.clear();

  // Ensure topology hash is fresh
  netlist.globalBlock.computeTopologyHash();

  // Initialize SoA Block for this solve
  soaBlock = tensorizeNetlist(netlist);

  // Unknowns: node voltages then MNA branch currents (V, L, E, H)
  const int nTot = systemSize(netlist);
  std::vector<double> voltageEstimate(nTot, 0.0);

  // Start: the last converged operating point when there is one (a .dc sweep's
  // previous point; ngspice continues from it with its junction-limiting state),
  // else all zeros, as ngspice. A heuristic seed (sources' values on their nodes,
  // 0.6 V anodes, mid-rail current-source nodes, Vbe ladders) put cross-coupled
  // latches on their metastable midpoint (QA3 BUG-5; 7 more IHP flip-flops matched
  // ngspice without it, and nothing needed it).
  const std::vector<double> coldSeed(nTot, 0.0);
  const bool warm = !prevVoltages_.empty() && prevVoltages_.size() == voltageEstimate.size();
  voltageEstimate = warm ? prevVoltages_ : coldSeed;
  if (warm) syncJunctionLimits(netlist, voltageEstimate);
  else resetJunctionLimits(netlist);  // first iteration: junctions at vcrit (ngspice MODEINITJCT)

  bool converged = false;
  int total_iterations = 0;
  double final_kcl_error = 0.0;
  bool singularMatrixDetected = false;
  int singularRow = -1;  // unknown with no pivot (ngspice: "singular matrix: check node N")
  std::vector<ConvergenceStep> convergenceHistory;
  // Phase 2.9: Numerical Intelligence state
  bool stagnationDetected = false;
  bool divergenceDetected = false;
  bool rankDeficientDetected = false;
  int rankDeficientRow = -1;
  double smallestPivotOverall = 1e300;
  RegularizationInfo regulInfo;

  // SPICE order: plain Newton with NO node gmin first (junction GMIN in the
  // device models keeps floating junction nodes referenced, like ngspice — a
  // node-to-ground leak settles them differently: aac_10_bridge). Only if that
  // fails, ngspice's dynamic gmin stepping (cktop.c dynamic_gmin): node gmin from
  // 1e-3 down to 1e-12, the step factor grown after easy steps and shrunk after
  // hard ones, and a failed step restarts from the last good solution with a
  // smaller decrement; then a final plain Newton at gmin = 0. The fixed ladder it
  // replaces gave up at the first failed step, so long inverter chains (>= 50
  // stages), which ngspice solves this way, never converged.
  applyOptions(netlist.environment);
  const int ITL1 = itl1_;                 // ngspice: plain/final DC Newton limit
  const int ITL2 = itl2_;                 // ngspice: per gmin step (50)
  constexpr double GMIN_FACTOR = 10.0;
  const double GMIN_TARGET = gmin_ > 0.0 ? gmin_ : 1e-12;  // the ladder's end; gmin=0 still ends it
  double gmin_factor = 0.0;               // current node gmin (0 = plain Newton)
  double stepFactor = GMIN_FACTOR, oldGmin = 1e-2;
  bool finalStep = true, gminSteppingUsed = false;
  // Source stepping (ngspice gillespie_src), when gmin stepping fails too: all
  // independent sources scaled by srcFact_ from 0 up to 1, the increment grown
  // after easy steps, cut tenfold after a failed one (from the last good point).
  bool sourceSteppingUsed = false;
  // A node with no DC path (a floating gate, a capacitor-only node) leaves the
  // final gmin = 0 solve singular. ngspice 47 then runs its transient OP: from all
  // nodes at 0 V and every capacitor uncharged, sources applied, to steady state,
  // so such a node keeps zero net charge (a capacitive divider decides it).
  bool finalSingular = false, floatingNodes = false;
  double lastPivot = std::numeric_limits<double>::infinity();  // of the latest solve
  int lastPivotRow = -1;
  double srcConv = 0.0, srcRaise = 1e-3;
  const std::vector<double> &seededGuess = coldSeed;  // gmin stepping restarts cold
  std::vector<double> lastGood;
  for (int si = 0; si < 1000; ++si) {  // ponytail: hard cap; ngspice has none (factor -> 1 ends it)
    bool gmin_converged = false;

    // Phase 2.9: Track regularization
    if (gmin_factor > 0.0) {
      regulInfo.applied = true;
      regulInfo.gminValue = gmin_factor;
      regulInfo.injections = static_cast<uint32_t>(netlist.numGlobalNodes);
    }

    int iteration = 0; // Declare outside loop for scope access
    for (; iteration < (finalStep ? ITL1 : ITL2); ++iteration) {
      // Unified Stamping (Phase 1: Structural Abstraction)
      scheduler.schedule(netlist);
      matrixBuilder.clear();
      matrixBuilder.setDimensions(nTot, nTot);
      rhsVector.assign(nTot, 0.0);

      swForce_ = iteration == 0;
      stampAllElements(netlist, 0.0, voltageEstimate, 0.0);
      swForce_ = false;
      addDiagonalConditioning(netlist.numGlobalNodes, gmin_factor);
      if (gmin_factor > 0.0) {  // device-internal nodes are nodes too (ngspice CKTdiagGmin)
        int totalRows = 0;
        for (const auto &sl : branchSlots(netlist, totalRows))
          for (int r = sl.dev, e = sl.dev + deviceInternals(*sl.blk, *sl.st); r < e; ++r)
            matrixBuilder.add(r, r, gmin_factor);
      }

      // Sparse Merge (D5): Use cached pattern for O(N) assembly on iterations >
      // 0
      Csr_matrix matrix;
      bool patternValid = false;
      if (cachedPattern_ && cachedPattern_->rows == nTot)
        matrix = matrixBuilder.assembleFast(*cachedPattern_, &patternValid);
      if (!patternValid) {
        cachedPattern_ =
            std::make_unique<CachedCsrPattern>(matrixBuilder.buildPattern());
        matrix = matrixBuilder.createCsr();

        // Topology changed: drop history tied to the old unknown vector
        prevVoltages_.clear();
        // Gap 4: Reset seeder history on topology change
        neuralSeeder_.reset();
      }

      // Tensor-contraction direct solver for every circuit (the legacy
      // treewidth / MPO / Schur routes solved a voltage-only system)
      SolverResult result = solveLinear(matrix, rhsVector);
      lastPivot = result.smallestPivot;
      lastPivotRow = result.rankDeficientRow;
      if (iteration == 0 && si == 0) {
        std::cerr << "[TN-ROUTE] CONTRACTION (unknowns=" << nTot
                  << ", branches=" << nTot - netlist.numGlobalNodes
                  << ", fronts=" << contractionPlan_->fronts.size()
                  << ", levels=" << contractionPlan_->levels.size() << ")\n";
      }
      // Phase 2.9: Capture pivot diagnostics from LU
      if (result.smallestPivot < smallestPivotOverall) {
        smallestPivotOverall = result.smallestPivot;
        rankDeficientRow = result.rankDeficientRow;
      }
      if (result.rankDeficient) rankDeficientDetected = true;

      if (!result.converged) {
        // Matrix is singular - circuit topology error or floating node (reported
        // in error_detail if the solve fails; homotopy steps may hit it and recover)
        final_kcl_error = 1e9; // Flag as bad residual
        singularMatrixDetected = true;
        singularRow = result.rankDeficientRow;
        if (finalStep && gminSteppingUsed && !sourceSteppingUsed) finalSingular = true;
        // If singular, retrying with same Jacobian won't help. Abort this GMIN
        // step.
        total_iterations += iteration + 1;
        break;
      }

      std::vector<double> v_new = result.solution;

      // NaN Protection using SafeUtils
      SafeUtils::sanitize_vector(v_new, 0.0);


      // Check 1 + 2: Voltage Convergence AND Physical KCL Accuracy
      double kcl_error = kclExcess(netlist, v_new, matrix);

      // No residual-based damping (ngspice has none): the residual is measured at
      // the raw node voltages, where junction exponentials are not limited, so
      // damping on it fought the device-local junction limiting. Without it a cold
      // 60-stage RTL chain converges in 37 iterations instead of 131.
      // Record convergence step for diagnostics
      {
        ConvergenceStep cs;
        cs.iteration = iteration;
        cs.residualNorm = kcl_error;
        cs.dampingFactor = 1.0;  // full Newton steps
        cs.linearSolverResidual = result.finalResidual; // Phase 2.9

        // Compute max voltage delta
        double maxDelta = 0.0;
        for (size_t i = 0; i < v_new.size(); ++i) {
          double d = std::abs(v_new[i] - voltageEstimate[i]);
          if (d > maxDelta)
            maxDelta = d;
        }
        cs.maxVoltageDelta = maxDelta;
        cs.worstNode = ConvergenceEngine::findWorstNode(std::vector<double>(
            v_new.size(), kcl_error)); // Simplified; worst node from residual
        convergenceHistory.push_back(cs);
      }

      // Phase 2.9: Stagnation and divergence detection
      {
        // Stagnation: residual plateaued but not converged
        if (iteration > 0 && std::abs(final_kcl_error - kcl_error) < 1e-15 &&
            kcl_error > abstol_) {
          stagnationDetected = true;
        }
        // Divergence: last 3 iterations show monotonic increase
        size_t sz = convergenceHistory.size();
        if (sz >= 3) {
          if (convergenceHistory[sz - 1].residualNorm >
                  convergenceHistory[sz - 2].residualNorm &&
              convergenceHistory[sz - 2].residualNorm >
                  convergenceHistory[sz - 3].residualNorm) {
            divergenceDetected = true;
          }
        }
      }

      final_kcl_error = kcl_error; // Store for next iteration comparison

      // During GMIN stepping, we relax the KCL check because GMIN itself adds
      // residual. We only enforce strict KCL at the final (nominal GMIN) step.
      double effective_kcl_tol = finalStep ? abstol_ : 1.0;

      if (checkConvergence(v_new, voltageEstimate, kcl_error,
                           effective_kcl_tol)) {
        voltageEstimate = v_new;
        gmin_converged = true;
        total_iterations += iteration + 1;
        break;
      }

      voltageEstimate = v_new;
    }

    if (!osdiFatal_.empty()) break;  // a fatal model error does not go away with homotopy
    // If we finished the loop without converging, accumulate iterations
    if (!gmin_converged) {
      total_iterations += iteration + 1;
    }

    if (sourceSteppingUsed) {
      if (gmin_converged) {
        lastGood = voltageEstimate;
        srcConv = srcFact_;
        if (srcConv >= 1.0) { converged = true; break; }
        if (iteration + 1 <= ITL2 / 4) srcRaise *= 1.5;
        if (iteration + 1 > 3 * ITL2 / 4) srcRaise *= 0.5;
      } else {
        if (srcFact_ == 0.0 || srcRaise < 1e-7) break;  // zero sources fail, or the ramp stalled
        srcRaise /= 10.0;
        voltageEstimate = lastGood;
        syncJunctionLimits(netlist, voltageEstimate);
      }
      srcFact_ = std::min(1.0, srcConv + srcRaise);
      continue;
    }
    // Only a gmin = 0 solve marks the simulation as converged
    if (finalStep && gmin_converged) {
      converged = true;
      break;
    }
    if (finalStep && (gminSteppingUsed || !netlist.environment.gmin_stepping)) {  // gmin stepping failed (or off): sources from zero
      sourceSteppingUsed = true;
      std::cerr << "Note: starting source stepping\n";  // as ngspice
      srcFact_ = 0.0;
      voltageEstimate = lastGood = std::vector<double>(nTot, 0.0);
      resetJunctionLimits(netlist);
      stagnationDetected = divergenceDetected = false;
      continue;
    }

    if (!gminSteppingUsed) {  // direct attempt failed: start from the seed
      gminSteppingUsed = true;
      voltageEstimate = lastGood = seededGuess;
      resetJunctionLimits(netlist);  // ngspice dynamic_gmin starts in MODEINITJCT
      stagnationDetected = divergenceDetected = false;
      gmin_factor = oldGmin / stepFactor;
    } else if (gmin_converged) {
      const int iters = iteration + 1;
      lastGood = voltageEstimate;
      if (gmin_factor <= GMIN_TARGET) {
        gmin_factor = 0.0;
      } else {
        if (iters <= ITL2 / 4) stepFactor = std::min(stepFactor * std::sqrt(stepFactor), GMIN_FACTOR);
        if (iters > 3 * ITL2 / 4) stepFactor = std::sqrt(stepFactor);
        oldGmin = gmin_factor;
        if (gmin_factor < stepFactor * GMIN_TARGET) {
          stepFactor = gmin_factor / GMIN_TARGET;
          gmin_factor = GMIN_TARGET;
        } else {
          gmin_factor /= stepFactor;
        }
      }
    } else if (stepFactor < 1.00005) {
      gmin_factor = 0.0;  // ladder failed: ngspice still tries the final solve
    } else {              // back off: smaller decrement from the last good point
      stepFactor = std::sqrt(std::sqrt(stepFactor));
      gmin_factor = oldGmin / stepFactor;
      voltageEstimate = lastGood;
      syncJunctionLimits(netlist, voltageEstimate);
    }
    finalStep = gmin_factor == 0.0;
  } // End GMIN stepping loop

  // A DC solution needs a sound final factorization. Newton may cross near-singular
  // iterates (a 100-stage RTL chain's source-stepping start has 1e-92 pivots and converges),
  // but an answer factored through a pivot at or below ngspice's PIVTOL is no solution: a
  // floating network with a current forced into it, split over fronts, left a 1e-18
  // rounding residue and "converged" at 1e15 V (BUG-6: chains, stars, grids).
  if (converged && !(lastPivot > PhysicsConstants::PIVTOL)) {
    converged = false;
    singularMatrixDetected = true;
    singularRow = lastPivotRow;
  }

  bool anySwitch = false;
  forEachSwitch(netlist, [&](const SwitchRef &, const std::array<int, 2> &) { anySwitch = true; });
  if (!converged && (finalSingular || anySwitch)) {  // ngspice: a switch circuit without a DC state falls back to its transient OP
    // ngspice's defaults: 100 ns steps to 10 us (optran ... 100n 10u)
    srcFact_ = 1.0;  // source stepping may have left the sources scaled down
    TensorNetlist pseudo = netlist;
    // ngspice keeps a node gmin during its transient OP: a node with no capacitance
    // either (an isolated MOS1 gate) would leave even the transient singular.
    struct GminOn { double &g; ~GminOn() { g = 0.0; } } gminOn{ptranGmin_};
    ptranGmin_ = gmin_;
    if (runTransient(pseudo, 100e-9, 10e-6, /*uic=*/true, [](const SolverStep &) { return true; }).empty()) {
      voltageEstimate = lastSolution_;
      voltageEstimate.resize(nTot, 0.0);
      // As ngspice 47: the 10 us value is the operating point, settled or not (1 mA into a
      // 1 uF cap-only node gives I*10us/C = 0.01 V there too; a floating node behind a 1 ms
      // RC is mid-charge). BUG-7 asked for "singular" here; ngspice does not say that.
      converged = floatingNodes = true;
    }
  }

  // Phase 2.9: Final per-node residual for RMS + worst-node tracking
  std::vector<double> perNodeResiduals;
  final_kcl_error =
      calculatePhysicalResiduals(netlist, voltageEstimate, perNodeResiduals);
  perNodeResiduals.resize(netlist.numGlobalNodes);  // worst NODE (not branch row)

  // Compute RMS and worst node
  double rmsResidual = 0.0;
  uint32_t worstNodeIndex = 0;
  double worstNodeVoltage = 0.0;
  if (!perNodeResiduals.empty()) {
    double sumSq = 0.0;
    double maxAbs = 0.0;
    for (size_t i = 0; i < perNodeResiduals.size(); ++i) {
      double v = std::abs(perNodeResiduals[i]);
      sumSq += v * v;
      if (v > maxAbs) {
        maxAbs = v;
        worstNodeIndex = static_cast<uint32_t>(i + 1); // 1-based
        worstNodeVoltage =
            (i < voltageEstimate.size()) ? voltageEstimate[i] : 0.0;
      }
    }
    rmsResidual =
        std::sqrt(sumSq / static_cast<double>(perNodeResiduals.size()));
  }

  std::string details;
  if (final_kcl_error > 1.0 || !converged) {
    details = auditPhysicalResiduals(netlist, voltageEstimate);
  }

  // NR Convergence Diagnostics
  ConvergenceFailureType failureType = ConvergenceEngine::classifyFailure(
      convergenceHistory, singularMatrixDetected);

  std::vector<ResidualContribution> worstDevices;
  int worstNode = 0;
  if (!converged) {
    calculatePhysicalResidualsDetailed(netlist, voltageEstimate, worstDevices,
                                       worstNode);
  }

  std::string justification = ConvergenceEngine::generateJustification(
      failureType, convergenceHistory, worstDevices);

  // Phase 1.7.9 + Phase 2.9: Build trust-enriched SolverStats
  SolverStats stats;
  stats.iterations = total_iterations;
  stats.residual = final_kcl_error;
  stats.converged = converged;
  if (!converged && singularRow >= 0) {
    std::string what;
    if (singularRow < netlist.numGlobalNodes) {
      const std::string &nm = singularRow + 1 < (int)netlist.nodeNames.size() ? netlist.nodeNames[singularRow + 1] : "";
      what = "Singular matrix: check node " + (nm.empty() ? std::to_string(singularRow + 1) : "'" + lower(nm) + "'") +
             " (no DC path to ground?)";
    } else {  // a branch row: name its element
      int total = 0;
      std::string who = std::to_string(singularRow - netlist.numGlobalNodes + 1);
      for (const auto &sl : branchSlots(netlist, total)) {
        auto hit = [&](int first, const auto &v) { return singularRow >= first && singularRow < first + (int)v.size() ? v[singularRow - first].name : std::string(); };
        auto lineHit = [&] { const int k = (singularRow - sl.tl) / 2; return singularRow >= sl.tl && k < (int)sl.blk->transmissionLines.size() ? sl.blk->transmissionLines[k].name : std::string(); };
        std::string nm = hit(sl.vs, sl.blk->voltageSources);
        if (nm.empty()) nm = hit(sl.ind, sl.blk->inductors);
        if (nm.empty()) nm = lineHit();
        if (!nm.empty()) { who = lower(nm) + (sl.blk->name == "GLOBAL" ? "" : " in " + lower(sl.blk->name)); break; }
      }
      what = "Singular matrix: check branch " + who + " (a loop of voltage sources, inductors or 0-ohm resistors?)";
    }
    details = what + (details.empty() ? "" : "\n" + details);
  }
  if (!converged && !osdiFatal_.empty()) details = "OSDI model error: " + osdiFatal_;
  else if (!converged && !osdiNonFinite_.empty()) details = "OSDI model error: " + osdiNonFinite_ + (details.empty() ? "" : "\n" + details);
  if (!converged && details.empty())
    details = std::string("DC operating point did not converge (Newton") +
              (gminSteppingUsed ? ", gmin stepping" : "") + (sourceSteppingUsed ? ", source stepping" : "") + " failed)";
  stats.error_detail = details;
  stats.method = "Newton-Raphson";
  stats.integrationMethod = ""; // DC has no integration
  stats.gminSteppingUsed = gminSteppingUsed;  // the direct attempt failed: ladder walked
  stats.sourceSteppingUsed = sourceSteppingUsed;
  stats.failureType = failureType;
  stats.convergenceHistory = std::move(convergenceHistory);
  stats.worstDevices = std::move(worstDevices);
  stats.convergenceJustification = floatingNodes
      ? std::string("Operating point has nodes without a DC path (singular matrix): taken from a transient "
                    "from zero to 10 us, as ngspice's transient OP")
      : justification;
  stats.convergenceTolerance = 1e-6;
  // Phase 2.9: Numerical Intelligence fields
  stats.rmsResidual = rmsResidual;
  stats.worstNodeIndex = worstNodeIndex;
  stats.worstNodeVoltage = worstNodeVoltage;
  stats.stagnationDetected = stagnationDetected;
  stats.divergenceDetected = divergenceDetected;
  stats.rankDeficient = rankDeficientDetected || singularMatrixDetected;
  stats.rankDeficientRow = rankDeficientRow;
  stats.smallestPivot = smallestPivotOverall;
  stats.regularization = regulInfo;

  // Gap 3: Save converged voltages for next solve's adaptive precision routing
  // Gap 4: Record solution in neural seeder history
  if (converged) {
    acceptSwitches(netlist);
    prevVoltages_ = voltageEstimate;
    neuralSeeder_.recordSolution(voltageEstimate, final_kcl_error,
                                 total_iterations);
  }

  lastSolution_ = voltageEstimate;  // full x (incl. branch currents) seeds transient
  std::vector<double> branches(voltageEstimate.begin() + std::min<size_t>(netlist.numGlobalNodes, voltageEstimate.size()),
                               voltageEstimate.begin() + std::min<size_t>(netlist.numGlobalNodes + branchCount(netlist), voltageEstimate.size()));
  voltageEstimate.resize(netlist.numGlobalNodes);  // API returns node voltages
  return {0.0, voltageEstimate, stats, netlist.globalBlock.topologyHash, std::move(branches)};
}

// =============================================================================
// COLD-START MICRO-STEPPING (SPICE3 dt_init = tstep/100)
// =============================================================================

// Re-tensorize and carry the AoS companion-model history (global block AND
// subcircuit instances, in tensorizeNetlist()'s flattening order) into the SoA
// arrays. Previously only global devices were synced, so capacitors inside
// subcircuits restarted from zero history every step.
void CircuitSim::loadSoAFromNetlist(const TensorNetlist &netlist) {
  soaBlock = tensorizeNetlist(netlist);
  size_t kc = 0;
  auto sync = [&](const TensorBlock &blk, const BlockState &st) {
    for (size_t i = 0; i < blk.capacitors.size(); ++i, ++kc) {
      if (i >= st.capacitorState.size() || kc >= soaBlock.capacitors.size()) continue;
      const auto &hist = st.capacitorState[i];
      if (!hist.v.empty()) soaBlock.capacitors.v_prev[kc] = hist.v[0];
      if (hist.v.size() > 1) soaBlock.capacitors.v_prev2[kc] = hist.v[1];
      if (!hist.i.empty()) soaBlock.capacitors.i_prev[kc] = hist.i[0];
    }
  };
  sync(netlist.globalBlock, netlist.globalState);
  for (size_t k = 0; k < netlist.instances.size(); ++k) {
    auto it = netlist.blockDefinitions.find(netlist.instances[k].blockName);
    if (it != netlist.blockDefinitions.end() && k < netlist.instanceStates.size())
      sync(it->second, netlist.instanceStates[k]);
  }
}

// Reactive-element history after an accepted step, for the global block AND
// every subcircuit instance. Capacitors use exactly the formulas of
// stampSoABlock() and the KCL audit; inductors record their SOLVED branch
// current (previously inductor history was never written back at all, so any
// transient with an inductor failed to converge).
void CircuitSim::updateReactiveHistory(TensorNetlist &netlist, const std::vector<double> &v,
                                        double h) {
  const IntegrationType type = integrator->getType();
  auto getV = [&](int n) { return (n > 0 && n <= (int)v.size()) ? v[n - 1] : 0.0; };
  int total = 0;
  const auto branchList = branchSlots(netlist, total);  // same block order as below
  size_t slot = 0;
  auto update = [&](const TensorBlock &blk, BlockState &st, const std::function<int(int)> &mapNode) {
    const auto *sl = slot < branchList.size() ? &branchList[slot++] : nullptr;
    const int indBase = sl ? sl->ind : -1;
    for (size_t a = 0; a < blk.inductors.size() && a < st.inductorState.size() && indBase >= 0; ++a) {
      auto &hist = st.inductorState[a];
      if (hist.v.empty()) hist.resize(3);
      const auto &L = blk.inductors[a];
      const int b = indBase + (int)a;
      integrator->updateHistory(hist.v, hist.i,
                                getV(mapNode(L.nodeCoil1)) - getV(mapNode(L.nodeCoil2)),
                                b < (int)v.size() ? v[b] : 0.0);
    }
    for (size_t i = 0; i < blk.capacitors.size() && i < st.capacitorState.size(); ++i) {
      const auto &c = blk.capacitors[i];
      auto &hist = st.capacitorState[i];
      if (hist.v.empty()) hist.resize(3);
      double v_curr = getV(mapNode(c.nodePlate1)) - getV(mapNode(c.nodePlate2));
      double C = c.capacitance_farads, i_curr;
      if (type == IntegrationType::TRAPEZOIDAL)
        i_curr = (2.0 * C / h) * (v_curr - hist.v[0]) - hist.i[0];
      else if (type == IntegrationType::GEAR_2 && hist.v.size() >= 2)
        i_curr = (C / h) * (1.5 * v_curr - 2.0 * hist.v[0] + 0.5 * hist.v[1]);
      else
        i_curr = (C / h) * (v_curr - hist.v[0]);
      integrator->updateHistory(hist.v, hist.i, v_curr, i_curr);
    }
    for (size_t k = 0; sl && k < blk.transmissionLines.size() && k < st.lineState.size(); ++k) {
      const auto &t = blk.transmissionLines[k];
      const int b = sl->tl + 2 * (int)k;
      const double z0 = t.characteristicImpedance_Z0_ohms;
      pushLine(st.lineState[k], h,
               getV(mapNode(t.nodePort1Pos)) - getV(mapNode(t.nodePort1Neg)) + z0 * (b < (int)v.size() ? v[b] : 0.0),
               getV(mapNode(t.nodePort2Pos)) - getV(mapNode(t.nodePort2Neg)) + z0 * (b + 1 < (int)v.size() ? v[b + 1] : 0.0),
               t.propagationDelay_Td_s);
    }
  };
  update(netlist.globalBlock, netlist.globalState, [](int n) { return n; });
  // MOSFET charges: the state mosCurrents() integrated at v (netlist is non-const here)
  forEachMosfet(netlist, [&](const Mosfet &m, const std::array<int, 4> &n, const BlockState::DeviceCharge *st) {
    if (st) mosCurrents<double>({getV(n[0]), getV(n[1]), getV(n[2]), getV(n[3])}, mos1Params(m, gmin_), st, h,
                                const_cast<BlockState::DeviceCharge *>(st));
  });
  acceptSwitches(netlist);
  // Diode / BJT charges, likewise (junctionCurrents with next = st)
  forEachJunction(netlist, [&](const Junction &j) {
    if (!j.st) return;
    std::array<double, 7> x{};
    for (int k = 0; k < j.terms; ++k) x[k] = getV(j.n[k]);
    junctionCurrents(j, x, h, j.st);
  });
  // OSDI charges, likewise (osdiCurrents with accept)
  osdiAll(netlist, v, h, false, true);
  for (size_t k = 0; k < netlist.instances.size() && k < netlist.instanceStates.size(); ++k) {
    const auto &inst = netlist.instances[k];
    auto it = netlist.blockDefinitions.find(inst.blockName);
    if (it == netlist.blockDefinitions.end()) continue;
    update(it->second, netlist.instanceStates[k], [&](int n) {
      return (n <= 0 || n > (int)inst.nodeMapping.size()) ? 0 : inst.nodeMapping[n - 1];
    });
  }
}

void CircuitSim::pushLine(BlockState::LineHistory &h, double dt, double w1, double w2, double td) {
  const size_t n = h.t.size();
  const double t = (n ? h.t.back() : 0.0) + dt;
  // A kink: the slope changes by more than half, by more than reltol*|w| + vntol over this step.
  // Smooth curvature stays below that; corners land on step boundaries, so the break is at t[n-1].
  for (auto [y, w] : {std::pair{&h.w1, w1}, std::pair{&h.w2, w2}}) {
    if (n < 2) break;
    const double s1 = ((*y)[n - 1] - (*y)[n - 2]) / (h.t[n - 1] - h.t[n - 2]), s2 = (w - (*y)[n - 1]) / dt;
    if (std::abs(s2 - s1) * dt > 1e-3 * std::max(std::abs(w), std::abs((*y)[n - 1])) + 1e-6 &&
        std::abs(s2 - s1) > 0.5 * std::max(std::abs(s1), std::abs(s2)))
      h.breaks.insert(h.t[n - 1] + td);
  }
  h.t.push_back(t); h.w1.push_back(w1); h.w2.push_back(w2);
  while (h.t.size() > 3 && h.t[1] <= t - 2.0 * td) {  // only t - TD onwards is read again
    h.t.erase(h.t.begin()); h.w1.erase(h.w1.begin()); h.w2.erase(h.w2.begin());
  }
  h.breaks.erase(h.breaks.begin(), h.breaks.upper_bound(t));
}

// t = 0 reactive state from x (DC operating point, or the UIC node voltages):
// capacitors hold their voltage at zero current, inductors their current at
// zero voltage, so the first step starts in equilibrium. With UIC an element's
// own IC= wins (SPICE3: unspecified nodes are 0 V, inductor currents 0 A).
void CircuitSim::seedReactiveHistory(TensorNetlist &netlist, const std::vector<double> &x,
                                     bool uic) {
  auto getV = [&](int n) { return (n > 0 && n <= (int)x.size()) ? x[n - 1] : 0.0; };
  int total = 0;
  const auto branchList = branchSlots(netlist, total);  // same block order as below
  size_t slot = 0;
  auto seed = [&](const TensorBlock &blk, BlockState &st, const std::function<int(int)> &mapNode) {
    const auto *sl = slot < branchList.size() ? &branchList[slot++] : nullptr;
    const int indBase = sl ? sl->ind : -1;
    for (size_t a = 0; a < blk.inductors.size() && a < st.inductorState.size(); ++a) {
      const auto &L = blk.inductors[a];
      const int b = indBase + (int)a;
      const double i0 = uic ? (L.icGiven ? L.ic_A : 0.0)
                            : (indBase >= 0 && b < (int)x.size() ? x[b] : 0.0);
      st.inductorState[a].v.assign(3, 0.0);
      st.inductorState[a].i.assign(3, i0);
    }
    for (size_t i = 0; i < blk.capacitors.size() && i < st.capacitorState.size(); ++i) {
      const auto &c = blk.capacitors[i];
      const double v0 = (uic && c.icGiven) ? c.ic_V
                                           : getV(mapNode(c.nodePlate1)) - getV(mapNode(c.nodePlate2));
      st.capacitorState[i].v.assign(3, v0);
      st.capacitorState[i].i.assign(3, 0.0);
    }
    // Lines: the waves at t = 0 (held before it); the line's first kink, the start of the drive, reaches the far end at TD
    for (size_t k = 0; sl && k < blk.transmissionLines.size() && k < st.lineState.size(); ++k) {
      const auto &t = blk.transmissionLines[k];
      const int b = sl->tl + 2 * (int)k;
      const double z0 = t.characteristicImpedance_Z0_ohms;
      auto &h = st.lineState[k];
      h = {};
      if (uic && t.icGiven) h.w1 = {t.ic[0] + z0 * t.ic[1]}, h.w2 = {t.ic[2] + z0 * t.ic[3]};
      else h.w1 = {getV(mapNode(t.nodePort1Pos)) - getV(mapNode(t.nodePort1Neg)) + z0 * (b < (int)x.size() ? x[b] : 0.0)},
           h.w2 = {getV(mapNode(t.nodePort2Pos)) - getV(mapNode(t.nodePort2Neg)) + z0 * (b + 1 < (int)x.size() ? x[b + 1] : 0.0)};
      h.t = {0.0};
      h.breaks = {t.propagationDelay_Td_s};
    }
  };
  seed(netlist.globalBlock, netlist.globalState, [](int n) { return n; });
  // MOSFET charges at x, zero current (Meyer charges count from here: only their change matters)
  forEachMosfet(netlist, [&](const Mosfet &m, const std::array<int, 4> &n, const BlockState::DeviceCharge *st) {
    if (!st) return;
    const double zero[5] = {};
    const auto c = mos1_charges(getV(n[0]), getV(n[1]), getV(n[2]), getV(n[3]), mos1Params(m, gmin_), zero, zero, zero);
    auto *s = const_cast<BlockState::DeviceCharge *>(st);  // netlist is non-const here
    *s = {};
    for (auto &vh : s->v) std::copy(c.v, c.v + 5, vh);
    std::copy(c.q, c.q + 5, s->q);
    std::copy(c.half, c.half + 3, s->half);
  });
  // Diode / BJT charges at x, zero current
  forEachJunction(netlist, [&](const Junction &j) {
    if (!j.st) return;
    std::array<double, 7> xv{};
    for (int k = 0; k < j.terms; ++k) xv[k] = getV(j.n[k]);
    double v[5], q[5];
    const int K = junctionCharges(j, xv, v, q);
    *j.st = {};
    for (auto &vh : j.st->v) std::copy(v, v + K, vh);
    std::copy(q, q + K, j.st->q);
  });
  // OSDI charges at x, zero current
  forEachOsdi(netlist, [&](const OsdiRef &r) {
    osdiCurrents(r, x, 0.0, nullptr, true);
    r.in->vHist[1] = r.in->vHist[2] = r.in->vHist[0];
  });
  osdiAtValid_ = false;  // the charge history moved
  for (size_t k = 0; k < netlist.instances.size() && k < netlist.instanceStates.size(); ++k) {
    const auto &inst = netlist.instances[k];
    auto it = netlist.blockDefinitions.find(inst.blockName);
    if (it == netlist.blockDefinitions.end()) continue;
    seed(it->second, netlist.instanceStates[k], [&](int n) {
      return (n <= 0 || n > (int)inst.nodeMapping.size()) ? 0 : inst.nodeMapping[n - 1];
    });
  }
}

std::string CircuitSim::runTransient(TensorNetlist &netlist, double step, double stop, bool uic,
                                     const std::function<bool(const SolverStep &)> &onPoint) {
  if (step == 0.0) step = stop / 50.0;  // .tran 0 tstop: ngspice's default TMAX
  if (!(step > 0.0) || !(stop > 0.0)) return "Transient needs step >= 0 and stop time > 0.";
  const std::string err = beginTransient(netlist, step, uic, stop);
  return err.empty() ? advanceTo(netlist, stop, onPoint) : err;
}

// Waveform sources whose corners the steps must land on (SPICE breakpoints).
// ponytail: scans every waveform source per step; a merged corner queue is
// the upgrade if designs carry thousands of PULSE sources.
void CircuitSim::tranCorners(const TensorNetlist &netlist) {
  auto &corners = tran_.corners;
  corners.clear();
  auto addCorners = [&](const TensorBlock &b) {
    for (const auto &v : b.voltageSources)
      if (v.type == "PULSE" || v.type == "SINE" || v.type == "PWL" || v.type == "EXP" || v.type == "AM") corners.push_back([&v](double x) { return nextBreakpoint(v, x); });
    for (const auto &i : b.currentSources)
      if (i.type == "PULSE" || i.type == "SINE" || i.type == "PWL" || i.type == "EXP" || i.type == "AM") corners.push_back([&i](double x) { return nextBreakpoint(i, x); });
  };
  addCorners(netlist.globalBlock);
  for (const auto &[name, blk] : netlist.blockDefinitions) addCorners(blk);
  auto addLines = [&](const BlockState &st) {
    for (const auto &l : st.lineState)
      corners.push_back([&b = l.breaks](double x) {
        const auto it = b.upper_bound(x);
        return it == b.end() ? std::numeric_limits<double>::infinity() : *it;
      });
  };
  addLines(netlist.globalState);
  for (const auto &st : netlist.instanceStates) addLines(st);
}

std::string CircuitSim::beginTransient(TensorNetlist &netlist, double step, bool uic, double stop) {
  if (!(step > 0.0)) return "Transient needs step > 0.";
  osdiFatal_.clear();
  osdiNonFinite_.clear();
  applyOptions(netlist.environment);
  setIntegrationMethod(tranMethod_);
  forEachSwitch(netlist, [&](const SwitchRef &w, const std::array<int, 2> &) { *w.on = *w.prev = w.s->init; });
  const int nTot = systemSize(netlist);
  std::vector<double> x0(nTot, 0.0);
  if (uic) {
    for (const auto &[n, v] : netlist.nodeICs)
      if (n > 0 && n <= netlist.numGlobalNodes) x0[n - 1] = v;
  } else {
    // DC bias. SPICE3 forces .ic nodes during it through 1e10 S to the value
    // (Norton: 1e-10 ohm to ground plus v*1e10 A in). UIC skips it, which an
    // op-amp integrator without DC feedback needs (its OP is the open-loop rail).
    TensorNetlist forced;
    if (!netlist.nodeICs.empty()) {
      forced = netlist;
      for (const auto &[n, v] : netlist.nodeICs) {
        forced.addResistor(n, 0, 1e-10);
        forced.addCurrentSource(0, n, v * 1e10);
      }
    }
    SolverStep dc = solveDC(netlist.nodeICs.empty() ? netlist : forced);
    if (!dc.stats.converged)
      return "DC bias failed." + (dc.stats.error_detail.empty() ? "" : " " + dc.stats.error_detail);
    x0 = lastSolution_;
    x0.resize(nTot, 0.0);
  }
  arbiterInitialized = false;  // this run's cold start (the DC bias may itself have run a transient OP)
  lteHold_ = 0;
  seedReactiveHistory(netlist, x0, uic);
  lastSolution_ = x0;  // seeds the first Newton iteration

  // A failed step returns converged=false at the SAME time: shrink h and
  // retry, give up below dtMin, regrow toward the user step.
  tran_ = {};
  tran_.step = step;
  tran_.dtMin = step * 1e-9;
  tran_.endTol = step * 1e-9;  // absorb FP drift at the final step
  // The step after a corner uses backward Euler (SPICE: order 1 after a
  // breakpoint): trapezoidal rings +-dv across a derivative jump (an L driven
  // by a ramp that stops) and LTE rejection then shrinks h until it gives up.
  // Step size as ngspice dctran: never above min(tstep, tstop/50), first step
  // min(tstep, tstop/100)/100, then doubling under LTE rejection. Starting at tstep
  // sampled a 100 MHz sine only at its zeros and stepped an LC tank one period at a
  // time; LTE has no history to reject the first steps (QA3 BUG-6).
  tran_.hMax = stop > 0.0 ? std::min(step, stop / 50.0) : step;
  tran_.h = (stop > 0.0 ? std::min(step, stop / 100.0) : step) / 100.0;
  tranCorners(netlist);
  // A line is read TD back: a step longer than TD would need the value it is solving for
  auto capByDelay = [&](const TensorBlock &b) { for (const auto &t : b.transmissionLines) tran_.hMax = std::min(tran_.hMax, t.propagationDelay_Td_s); };
  capByDelay(netlist.globalBlock);
  for (const auto &[name, blk] : netlist.blockDefinitions) capByDelay(blk);
  tran_.h = std::min(tran_.h, tran_.hMax);
  return "";
}

std::string CircuitSim::advanceTo(TensorNetlist &netlist, double stop,
                                  const std::function<bool(const SolverStep &)> &onPoint) {
  auto &[step, dtMin, endTol, hMax, h, t, landedOnCorner, tinySteps, accepted, corners] = tran_;
  while (stop - t > endTol) {
    const bool endClip = stop - t < h;
    double hTry = std::min(h, stop - t);
    const double intended = hTry;  // before clipping to a corner (ngspice's CKTsaveDelta)
    bool toCorner = false;
    for (const auto &next : corners)
      if (next(t + endTol) - t <= hTry) { hTry = next(t + endTol) - t; toCorner = true; }
    // The second step is backward Euler too (ngspice stays at order 1 until LTE allows 2): the
    // first step's i history is the start inrush, and a trapezoid step on it rings +-dI forever
    // (OSDI devices at a UIC start, +-0.1 A on i(vdd)); BE stores the relaxed current instead.
    const bool euler = landedOnCorner || accepted == 1;
    if (accepted == 1) lteHold_ = std::max(lteHold_, 1);
    if (euler) setIntegrationMethod(IntegrationType::GEAR_1_EULER);
    SolverStep s = stepTransient(netlist, hTry, t);
    if (euler) setIntegrationMethod(tranMethod_);
    if (!osdiFatal_.empty()) return "OSDI model error at t=" + std::to_string(t) + " s: " + osdiFatal_;
    if (!s.stats.converged || !(s.time > t)) {
      h = hTry * 0.25;
      if (h < dtMin) {
        std::ostringstream msg;  // default float format: 1e-14, not 0.000000
        msg << "Transient step failed to converge at t=" << t << " s (timestep below " << dtMin
            << " s).";
        return msg.str();
      }
      continue;
    }
    const double taken = s.time - t;
    const bool tickEnd = endClip && taken >= hTry * (1.0 - 1e-9);  // the tick, not the step size, ended this step
    // Stalled: a run of accepted steps far below the user step (an ill-posed model
    // that converges only at femtosecond steps) would take millions of them.
    // ponytail: 1000 consecutive steps below step*1e-4 is a heuristic bound.
    if (!tickEnd) tinySteps = taken < step * 1e-4 ? tinySteps + 1 : 0;
    if (tinySteps > 1000) {
      std::ostringstream msg;
      msg << "Transient stalled at t=" << t << " s: 1000 steps below " << step * 1e-4 << " s (timestep too small).";
      return msg.str();
    }
    if (taken < dtMin) {  // accepted, but LTE control cut it below dtMin: no progress (ngspice: delmin)
      std::ostringstream msg;
      msg << "Transient timestep too small at t=" << t << " s (" << taken << " s < " << dtMin << " s).";
      return msg.str();
    }
    landedOnCorner = toCorner && taken >= hTry * (1.0 - 1e-9);
    t = s.time;
    ++accepted;
    if (onPoint && !onPoint(s)) return "Cancelled.";
    const double hKept = h;
    h = std::min(hMax, taken * 2.0);
    if (tickEnd && !toCorner) h = std::max(h, hKept);  // a clipped tick says nothing about the step size
    // After a corner, as ngspice dctran: at most 0.1 * min(the step that was intended
    // before it was clipped to the corner, the gap to the next corner). A source jump (a
    // PWL repeat) taken with one 4.5 ns backward-Euler step left an RC 13 mV low against
    // the exact response. The clipped step itself can be femtoseconds, and 0.1 of it made
    // Newton fail on roundoff (an inverter's PWL input, Xyce tranInverterBinning).
    if (lteHold_ > 0) --lteHold_;
    if (landedOnCorner) {
      lteHold_ = 3;
      double gap = std::numeric_limits<double>::infinity();
      for (const auto &next : corners) gap = std::min(gap, next(t + endTol) - t);
      h = std::min(h, 0.1 * std::min(intended, gap));
    }
  }
  return "";
}

bool CircuitSim::setSource(TensorNetlist &netlist, const std::string &name, double volts, double rampS) {
  std::string key = name;
  for (auto &c : key) c = (char)std::toupper((unsigned char)c);
  const double t = tran_.t;
  auto drive = [&](auto &srcs, auto dc) {  // V: voltage_V, I: current_A
    for (auto &src : srcs) {
      if (src.name != key) continue;
      const double now = waveformValue(src, dc(src), t);
      src.type = "PWL";
      src.pwl_t = {t, t + std::max(rampS, 0.0)};
      src.pwl_v = {now, volts};
      src.pwl_td = 0.0;
      src.pwl_r = -1.0;
      return true;
    }
    return false;
  };
  auto &g = netlist.globalBlock;  // ponytail: top level only; sources inside a .subckt are not reachable by name
  if (!drive(g.voltageSources, [](auto &v) { return v.voltage_V; }) &&
      !drive(g.currentSources, [](auto &i) { return i.current_A; }))
    return false;
  tranCorners(netlist);
  // The jump is a corner: the next step is backward Euler and short, as after any breakpoint
  tran_.landedOnCorner = true;
  lteHold_ = 3;
  tran_.h = std::min(tran_.h, 0.1 * tran_.hMax);
  return true;
}

bool CircuitSim::acSystem(TensorNetlist &netlist, ACSystem &sys) {
  sys = {};
  sys.n = systemSize(netlist);
  if ((int)lastSolution_.size() != sys.n) return false;
  // G: the DC Newton Jacobian at the operating point (same stamps as solveDC),
  // without the node-to-ground gmin: like ngspice, a 1e-12 S leak on every node
  // would load 1e10-ohm references by 1 % (aac_09 transformer secondaries).
  matrixBuilder.clear();
  matrixBuilder.setDimensions(sys.n, sys.n);
  rhsVector.assign(sys.n, 0.0);
  applyOptions(netlist.environment);
  limitJunctions_ = false;  // the Jacobian at the operating point itself
  swAcOff_ = true;
  acLines_ = true;
  stampAllElements(netlist, 0.0, lastSolution_, 0.0);
  acLines_ = false;
  swAcOff_ = false;
  addDiagonalConditioning(netlist.numGlobalNodes, 0.0);  // rshunt only
  limitJunctions_ = true;
  int rtotal = 0;
  for (const auto &sl : branchSlots(netlist, rtotal))  // AC=: the resistance .ac sees instead of R
    for (const auto &r : sl.blk->resistors) {
      if (std::isnan(r.ac_ohms)) continue;
      const double dg = 1.0 / r.ac_ohms - 1.0 / r.resistance_ohms;
      const int a = sl.map(r.nodeTerminal1), b = sl.map(r.nodeTerminal2);
      if (a > 0) matrixBuilder.add(a - 1, a - 1, dg);
      if (b > 0) matrixBuilder.add(b - 1, b - 1, dg);
      if (a > 0 && b > 0) { matrixBuilder.add(a - 1, b - 1, -dg); matrixBuilder.add(b - 1, a - 1, -dg); }
    }
  sys.G = matrixBuilder.createCsr();
  auto addC = [&](int r, int c, double v) {
    if (r >= 0 && c >= 0) { sys.cr.push_back(r); sys.cc.push_back(c); sys.cv.push_back(v); }
  };
  int total = 0;
  for (const auto &sl : branchSlots(netlist, total)) {
    auto node = [&](int local) { return sl.map(local) - 1; };  // ground -> -1
    for (const auto &c : sl.blk->capacitors) {
      const int a = node(c.nodePlate1), b = node(c.nodePlate2);
      const double C = c.capacitance_farads;
      addC(a, a, C); addC(b, b, C); addC(a, b, -C); addC(b, a, -C);
    }
    for (size_t k = 0; k < sl.blk->transmissionLines.size(); ++k) {  // row b1: ... - e^(-jw TD) (v2 + Z0 i2), row b2 likewise
      const auto &t = sl.blk->transmissionLines[k];
      const int b1 = sl.tl + 2 * (int)k, b2 = b1 + 1;
      auto addD = [&](int r, int c, double v) {
        if (r >= 0 && c >= 0) { sys.dr.push_back(r); sys.dc.push_back(c); sys.dv.push_back(v); sys.dtd.push_back(t.propagationDelay_Td_s); }
      };
      const double Z = t.characteristicImpedance_Z0_ohms;
      addD(b1, node(t.nodePort2Pos), -1.0); addD(b1, node(t.nodePort2Neg), 1.0); addD(b1, b2, -Z);
      addD(b2, node(t.nodePort1Pos), -1.0); addD(b2, node(t.nodePort1Neg), 1.0); addD(b2, b1, -Z);
    }
    const auto flux = fluxLinks(*sl.blk);  // inductor row: v1 - v2 - jw*sum L_aj i_j = 0
    for (size_t a = 0; a < flux.size(); ++a)
      for (auto [j, Laj] : flux[a]) addC(sl.ind + (int)a, sl.ind + j, -Laj);
  }
  // MOSFETs: C_tu = dQ_t/dV_u of the terminal charges at the operating point
  // (Meyer 2*half + overlap, junction dq/dv: ngspice mos1acld), from Dual.
  forEachMosfet(netlist, [&](const Mosfet &m, const std::array<int, 4> &n, const BlockState::DeviceCharge *) {
    const auto p = mos1Params(m, gmin_);
    if (!p.caps) return;
    auto V = [&](int k) { return n[k] > 0 ? lastSolution_[n[k] - 1] : 0.0; };
    const double zero[5] = {};
    const auto op = mos1_charges(V(0), V(1), V(2), V(3), p, zero, zero, zero);
    for (int u = 0; u < 4; ++u) {
      std::array<Dual<double>, 4> x;
      for (int k = 0; k < 4; ++k) x[k] = Dual<double>(V(k), k == u ? 1.0 : 0.0);
      const auto c = mos1_charges(x[0], x[1], x[2], x[3], p, op.v, zero, op.half);
      double Q[4] = {};
      for (int k = 0; k < 5; ++k) {
        Q[mosFrom[k]] += p.polarity * c.q[k].grad;
        Q[mosTo[k]] -= p.polarity * c.q[k].grad;
      }
      for (int t = 0; t < 4; ++t) if (Q[t] != 0.0) addC(n[t] - 1, n[u] - 1, Q[t]);
    }
  });
  // Diodes / BJTs: C_tu = dQ_t/dV_u of the junction charges (depletion + transit
  // time, XTF/VTF cross terms included) at the operating point, from Dual.
  forEachJunction(netlist, [&](const Junction &j) {
    if (!junctionHasCharge(j)) return;
    auto V = [&](int k) { return j.n[k] > 0 ? lastSolution_[j.n[k] - 1] : 0.0; };
    const int *from = j.d ? diodeFrom : j.f ? jfetFrom : bjtFrom, *to = j.d ? diodeTo : j.f ? jfetTo : bjtTo;
    const double type = j.q ? (j.q->isNPN ? 1.0 : -1.0) : j.f ? jfetType(*j.f) : 1.0;
    for (int u = 0; u < j.terms; ++u) {
      if (j.n[u] <= 0) continue;
      bool seen = false;
      for (int w = 0; w < u; ++w) seen = seen || j.n[w] == j.n[u];
      if (seen) continue;
      std::array<Dual<double>, 7> x;
      for (int k = 0; k < 7; ++k) x[k] = Dual<double>(V(k), k < j.terms && j.n[k] == j.n[u] ? 1.0 : 0.0);
      Dual<double> v[5], q[5];
      const int K = junctionCharges(j, x, v, q);
      double Q[7] = {};
      for (int k = 0; k < K; ++k) {
        const double s = (j.q && k == 3 ? j.subSign : 1.0) * type;  // substrate: to Junction::sub
        Q[from[k]] += s * q[k].grad;
        Q[j.q && k == 3 ? j.sub : to[k]] -= s * q[k].grad;
      }
      for (int t = 0; t < j.terms; ++t) if (Q[t] != 0.0) addC(j.n[t] - 1, j.n[u] - 1, Q[t]);
    }
  });
  // OSDI devices: the reactive Jacobian dq/dx at the operating point
  forEachOsdi(netlist, [&](const OsdiRef &r) {
    osdiCurrents(r, lastSolution_, 0.0);
    for (size_t a = 0; a < r.n.size(); ++a)
      for (size_t b = 0; b < r.n.size(); ++b)
        if (const double c = r.in->C(a, b); c != 0.0) addC(r.n[a] - 1, r.n[b] - 1, c);
  });
  return true;
}

Csr_matrix CircuitSim::acMatrix(const ACSystem &s, double w, bool transpose) {
  const int n = s.n;
  MatrixConstructor M;
  M.setDimensions(2 * n, 2 * n);
  auto put = [&](int r, int c, double v) { transpose ? M.add(c, r, v) : M.add(r, c, v); };
  for (int r = 0; r < n; ++r)
    for (int p = s.G.row_pointer[r]; p < s.G.row_pointer[r + 1]; ++p) {
      put(r, s.G.col_indices[p], s.G.values[p]);
      put(r + n, s.G.col_indices[p] + n, s.G.values[p]);
    }
  for (size_t t = 0; t < s.cv.size(); ++t) {
    put(s.cr[t], s.cc[t] + n, -w * s.cv[t]);
    put(s.cr[t] + n, s.cc[t], w * s.cv[t]);
  }
  for (size_t t = 0; t < s.dv.size(); ++t) {  // dv * (cos(w td) - j sin(w td))
    const double re = s.dv[t] * std::cos(w * s.dtd[t]), im = -s.dv[t] * std::sin(w * s.dtd[t]);
    put(s.dr[t], s.dc[t], re);
    put(s.dr[t] + n, s.dc[t] + n, re);
    put(s.dr[t], s.dc[t] + n, -im);
    put(s.dr[t] + n, s.dc[t], im);
  }
  return M.createCsr();
}

std::vector<std::complex<double>> CircuitSim::acExcitation(const TensorNetlist &nl,
                                                           const std::string &unit) const {
  int total = 0;
  const auto blocks = branchSlots(nl, total);
  std::vector<std::complex<double>> e(total);
  bool found = unit.empty();
  auto phasor = [&](const std::string &name, double mag, double phaseDeg) -> std::complex<double> {
    if (!unit.empty()) return name == unit ? (found = true, 1.0) : 0.0;
    return std::polar(mag, phaseDeg * M_PI / 180.0);
  };
  for (const auto &sl : blocks) {
    for (size_t i = 0; i < sl.blk->voltageSources.size(); ++i) {
      const auto &v = sl.blk->voltageSources[i];
      e[sl.vs + i] += phasor(v.name, v.ac_mag, v.ac_phase);
    }
    for (const auto &cs : sl.blk->currentSources) {  // leaves n+ externally (as in DC)
      const auto p = phasor(cs.name, cs.ac_mag, cs.ac_phase);
      const int a = sl.map(cs.nodePositive) - 1, b = sl.map(cs.nodeNegative) - 1;
      if (a >= 0) e[a] -= p;
      if (b >= 0) e[b] += p;
    }
  }
  return found ? e : std::vector<std::complex<double>>{};
}

std::vector<std::vector<std::complex<double>>>
CircuitSim::solveAC(TensorNetlist &netlist, const std::vector<double> &freqs) {
  ACSystem sys;
  if (!acSystem(netlist, sys)) return {};
  const int n = sys.n;
  const auto e = acExcitation(netlist);
  std::vector<double> b(2 * n);
  for (int i = 0; i < n; ++i) { b[i] = e[i].real(); b[i + n] = e[i].imag(); }
  std::vector<std::vector<std::complex<double>>> out;
  for (double f : freqs) {
    const SolverResult s = solveLinear(acMatrix(sys, 2.0 * M_PI * f, false), b);
    std::vector<std::complex<double>> x(n);
    for (int i = 0; i < n && i + n < (int)s.solution.size(); ++i) x[i] = {s.solution[i], s.solution[i + n]};
    out.push_back(std::move(x));
  }
  return out;
}

CircuitSim::NoiseResult CircuitSim::solveNoise(TensorNetlist &netlist, const std::vector<double> &freqs,
                                               int outNode, int refNode, const std::string &inputSource) {
  NoiseResult r;
  ACSystem sys;
  if (!acSystem(netlist, sys)) { r.error = "noise needs a converged DC operating point"; return r; }
  const int n = sys.n;
  const auto ein = acExcitation(netlist, inputSource);
  if (ein.empty()) { r.error = "noise input source '" + inputSource + "' not found"; return r; }
  if (outNode < 1 || outNode > netlist.numGlobalNodes || refNode < 0 || refNode > netlist.numGlobalNodes) {
    r.error = "noise output node out of range";
    return r;
  }

  // Generators: current noise of `psd` [A^2/Hz] between global nodes a, b (0 = ground)
  struct Gen { int a, b; double psd; std::function<double(double)> psdAt = nullptr; };  // psdAt(f): coloured
  std::vector<Gen> gens;
  const double kT4 = 4.0 * PhysicsConstants::K_BOLTZMANN * netlist.environment.ambient_temp_K;
  const double q2 = 2.0 * PhysicsConstants::Q_ELECTRON;
  auto V = [&](int g) { return g > 0 ? lastSolution_[g - 1] : 0.0; };
  int total = 0;
  for (const auto &sl : branchSlots(netlist, total)) {
    const auto &blk = *sl.blk;
    for (const auto &rr : blk.resistors)
      gens.push_back({sl.map(rr.nodeTerminal1), sl.map(rr.nodeTerminal2), kT4 / rr.resistance_ohms});
    for (const auto &sw : blk.switches)  // thermal noise of the conductance the small-signal matrix holds (ROFF)
      gens.push_back({sl.map(sw.nodePos), sl.map(sw.nodeNeg), kT4 / sw.roff});
    for (const auto &m : blk.mosfets) {
      const int d = sl.map(m.drain), g = sl.map(m.gate), s = sl.map(m.source), b = sl.map(m.body);
      // channel gm = dI_drain/dV_gate (the bulk junctions do not see the gate)
      const double gm = mos1_currents(Dual<double>(V(d)), Dual<double>(V(g), 1.0), Dual<double>(V(s)),
                                      Dual<double>(V(b)), mos1Params(m, gmin_)).d.grad;
      gens.push_back({d, s, kT4 * (2.0 / 3.0) * std::abs(gm)});
      if (m.card.kf != 0.0) {  // flicker (ngspice mos1noise, as probed): 1e-3 KF |Id| / (Cox W Leff M f^AF)
        const auto p = mos1Params(m, gmin_);
        const double id = std::abs(mos1_currents(V(d), V(g), V(s), V(b), p).d);
        gens.push_back({d, s, 0.0, [k = 1e-3 * m.card.kf * id / p.cox, af = m.card.af](double f) { return k / std::pow(f, af); }});
      }
    }
  }
  // Diodes / BJTs (ngspice dionoise, bjtnoise): series-resistance thermal noise,
  // shot noise of the junction currents between the internal nodes
  forEachJunction(netlist, [&](const Junction &j) {
    if (j.d) {
      const auto m = diodeModel(*j.d, gmin_);
      if (m.rs > 0.0) gens.push_back({j.n[0], j.n[2], kT4 / m.rs});
      const double id = diode_iq(V(j.n[2]) - V(j.n[1]), m).first;
      gens.push_back({j.n[2], j.n[1], q2 * std::abs(id)});
      if (j.d->card.kf != 0.0)  // flicker: KF |Id|^AF / f
        gens.push_back({j.n[2], j.n[1], 0.0, [kf = j.d->card.kf, af = j.d->card.af, id = std::abs(id)](double f) {
                          return kf * std::pow(std::max(id, 1e-38), af) / f; }});
      return;
    }
    if (j.f) {  // ngspice jfetnoi: channel 8kT gm/3 and flicker KF |Id|^AF / f between D and S (RD, RS are resistors)
      const auto m = jfetModel(*j.f, gmin_);
      const double t = m.type;
      auto eval = [&](double dv) {  // dv: vgs and vgd move together, i.e. vds is fixed (gm)
        return jfet_eval(Dual<double>(t * (V(j.n[1]) - V(j.n[2])), dv), Dual<double>(t * (V(j.n[1]) - V(j.n[0])), dv), m);
      };
      const auto e = eval(0.0);
      gens.push_back({j.n[0], j.n[2], kT4 * (2.0 / 3.0) * std::abs(eval(1.0).id.grad)});
      if (j.f->card.kf != 0.0)
        gens.push_back({j.n[0], j.n[2], 0.0, [kf = j.f->card.kf, af = j.f->card.af, id = std::abs(e.id.val)](double f) {
                          return kf * std::pow(std::max(id, 1e-38), af) / f; }});
      return;
    }
    const auto m = bjtModel(*j.q, gmin_);
    if (m.rc > 0.0) gens.push_back({j.n[0], j.n[4], kT4 / m.rc});
    if (m.re > 0.0) gens.push_back({j.n[2], j.n[6], kT4 / m.re});
    const double t = m.type;
    const auto e = bjt_gp(t * (V(j.n[5]) - V(j.n[6])), t * (V(j.n[5]) - V(j.n[4])), t * (V(j.n[1]) - V(j.n[4])),
                          t * j.subSign * (V(j.n[3]) - V(j.n[j.sub])), m);
    if (m.rb > 0.0) gens.push_back({j.n[1], j.n[5], kT4 / (m.varRb ? bjt_rbb(e.qb, e.cb, m) : m.rb)});
    if (j.q->card.kf != 0.0)  // flicker: KF |Ib|^AF / f between B' and E'
      gens.push_back({j.n[5], j.n[6], 0.0, [kf = j.q->card.kf, af = j.q->card.af, ib = std::abs(e.cb)](double f) {
                        return kf * std::pow(std::max(ib, 1e-38), af) / f; }});
    gens.push_back({j.n[4], j.n[6], q2 * std::abs(e.cc)});
    gens.push_back({j.n[5], j.n[6], q2 * std::abs(e.cb)});
  });
  // OSDI devices: the model's own noise sources (white and flicker) at the OP
  forEachOsdi(netlist, [&](const OsdiRef &r) {
    std::vector<double> x(r.n.size());
    for (size_t k = 0; k < x.size(); ++k) x[k] = V(r.n[k]);
    r.in->gmin = gmin_;
    r.in->evaluate(x.data(), osdi::ANALYSIS_NOISE | osdi::CALC_NOISE, 0.0);
    const auto srcs = r.in->noise(1.0);
    for (size_t s = 0; s < srcs.size(); ++s) {
      auto node = [&](int l) { return l < 0 ? 0 : r.n[l]; };
      gens.push_back({node(srcs[s].a), node(srcs[s].b), 0.0,
                      [in = r.in, s](double f) { return in->noise(f)[s].psd; }});
    }
  });

  // ngspice Nintegrate: integral of a density varying as a power of f between points
  constexpr double kMinLog = 1e-38;
  auto nint = [&](double dens, double lnD, double lnLast, double lnF, double lnF0, double dF) {
    double ex = (lnD - lnLast) / (lnF - lnF0);
    if (std::abs(ex) < kMinLog) return dens * dF;
    const double a = std::exp(lnD - ex * lnF);
    ex += 1.0;
    if (std::abs(ex) < kMinLog) return a * (lnF - lnF0);
    return a * (std::exp(ex * lnF) - std::exp(ex * lnF0)) / ex;
  };

  std::vector<double> bin(2 * n, 0.0), bout(2 * n, 0.0), lnLast(gens.size());
  for (int i = 0; i < n; ++i) { bin[i] = ein[i].real(); bin[i + n] = ein[i].imag(); }
  bout[outNode - 1] = 1.0;
  if (refNode > 0) bout[refNode - 1] = -1.0;
  std::unique_ptr<ContractionPlan> adjointPlan;  // R^T has its own pattern
  double onTot = 0.0, inTot = 0.0, lnF0 = 0.0, f0 = 0.0;
  for (size_t k = 0; k < freqs.size(); ++k) {
    const double f = freqs[k], w = 2.0 * M_PI * f;
    const SolverResult fw = solveLinear(acMatrix(sys, w, false), bin);
    const Csr_matrix RT = acMatrix(sys, w, true);
    if (!adjointPlan) adjointPlan = std::make_unique<ContractionPlan>(planContraction(RT));
    const SolverResult z = solveContraction(RT, bout, *adjointPlan);
    if ((int)fw.solution.size() != 2 * n || (int)z.solution.size() != 2 * n) {
      r.error = "noise: linear solve failed at f=" + std::to_string(f);
      return r;
    }
    auto cx = [&](const std::vector<double> &x, int g) {
      return g > 0 ? std::complex<double>(x[g - 1], x[g - 1 + n]) : 0.0;
    };
    const std::complex<double> H = cx(fw.solution, outNode) - cx(fw.solution, refNode);
    const double gainInv = 1.0 / std::max(std::norm(H), kMinLog), lnGainInv = std::log(gainInv);
    const double lnF = std::log(std::max(f, kMinLog));
    double S = 0.0;
    for (size_t g = 0; g < gens.size(); ++g) {
      const double dens = std::norm(cx(z.solution, gens[g].a) - cx(z.solution, gens[g].b)) *
                          (gens[g].psdAt ? gens[g].psdAt(f) : gens[g].psd);
      const double lnD = std::log(std::max(dens, kMinLog));
      if (k > 0) {
        onTot += nint(dens, lnD, lnLast[g], lnF, lnF0, f - f0);
        inTot += nint(dens * gainInv, lnD + lnGainInv, lnLast[g] + lnGainInv, lnF, lnF0, f - f0);
      }
      lnLast[g] = lnD;
      S += dens;
    }
    r.onoise.push_back(std::sqrt(S));
    r.inoise.push_back(std::sqrt(S * gainInv));
    lnF0 = lnF;
    f0 = f;
  }
  r.onoiseTotal = std::sqrt(onTot);
  r.inoiseTotal = std::sqrt(inTot);
  return r;
}

CircuitSim::PoleZeroResult CircuitSim::solvePoleZero(TensorNetlist &netlist, int inP, int inN, int outP, int outN,
                                                     bool volt, bool wantPoles, bool wantZeros) {
  PoleZeroResult r;
  ACSystem sys;
  if (!acSystem(netlist, sys)) { r.error = "pole-zero needs a converged DC operating point"; return r; }
  const int n = sys.n, N = n + 1;
  // (G + sC) x = 0 bordered: column `in` (the input source's injection), row `row` (shorted input / nulled output)
  auto roots = [&](const std::vector<std::pair<int, double>> &col, const std::vector<std::pair<int, double>> &row,
                   double corner, std::vector<std::complex<double>> &out) {
    std::vector<double> A(N * N, 0.0), B(N * N, 0.0);
    for (int i = 0; i < n; ++i)
      for (int p = sys.G.row_pointer[i]; p < sys.G.row_pointer[i + 1]; ++p) A[i * N + sys.G.col_indices[p]] += sys.G.values[p];
    for (size_t t = 0; t < sys.cv.size(); ++t) B[sys.cr[t] * N + sys.cc[t]] += sys.cv[t];
    for (auto [i, v] : col) A[i * N + n] += v;
    for (auto [j, v] : row) A[n * N + j] += v;
    A[n * N + n] = corner;
    return acutesim::finiteRoots(std::move(A), std::move(B), N, out);
  };
  auto pair = [&](int a, int b) {
    std::vector<std::pair<int, double>> v;
    if (a > 0) v.push_back({a - 1, 1.0});
    if (b > 0) v.push_back({b - 1, -1.0});
    return v;
  };
  for (int nd : {inP, inN, outP, outN})
    if (nd < 0 || nd > netlist.numGlobalNodes) { r.error = "pole-zero node out of range"; return r; }
  // Open input: border with a unit diagonal, which adds no root. Shorted input: the bordered system is singular
  // when an ideal source already holds the pair (the short is then a no-op), so that case is the open matrix too.
  if (wantPoles) {
    if (volt) r.error = roots(pair(inP, inN), pair(inP, inN), 0.0, r.poles);
    if (!volt || !r.error.empty()) r.error = roots({}, {}, 1.0, r.poles = {});
  }
  if (r.error.empty() && wantZeros) {
    r.error = roots(pair(inP, inN), pair(outP, outN), 0.0, r.zeros);
    if (!r.error.empty() && volt) {  // a source already holds the input pair: its value is the input (rhs of its branch row)
      int total = 0;
      for (const auto &sl : branchSlots(netlist, total))
        for (size_t i = 0; i < sl.blk->voltageSources.size(); ++i) {
          const int p = sl.map(sl.blk->voltageSources[i].nodePositive), m = sl.map(sl.blk->voltageSources[i].nodeNegative);
          if ((p == inP && m == inN) || (p == inN && m == inP)) {
            r.zeros.clear();
            return r.error = roots({{sl.vs + (int)i, 1.0}}, pair(outP, outN), 0.0, r.zeros), r;
          }
        }
    }
  }
  return r;
}

CircuitSim::SensResult CircuitSim::solveSens(TensorNetlist &netlist, int outNode, int refNode, const std::string &outSource) {
  SensResult r;
  ACSystem sys;
  if (!acSystem(netlist, sys)) { r.error = "sensitivity needs a converged DC operating point"; return r; }
  const int n = sys.n;
  auto &g = netlist.globalBlock;
  std::vector<double> e(n, 0.0);
  if (!outSource.empty()) {
    int total = 0;
    const int vs = branchSlots(netlist, total)[0].vs;
    const auto it = std::find_if(g.voltageSources.begin(), g.voltageSources.end(), [&](const VoltageSource &v) { return v.name == outSource; });
    if (it == g.voltageSources.end()) { r.error = ".sens i(" + outSource + "): no such voltage source at the top level"; return r; }
    e[vs + (it - g.voltageSources.begin())] = 1.0;
  } else {
    if (outNode > 0) e[outNode - 1] += 1.0;
    if (refNode > 0) e[refNode - 1] -= 1.0;
  }
  MatrixConstructor T;  // J^T
  T.setDimensions(n, n);
  for (int i = 0; i < n; ++i)
    for (int p = sys.G.row_pointer[i]; p < sys.G.row_pointer[i + 1]; ++p) T.add(sys.G.col_indices[p], i, sys.G.values[p]);
  const SolverResult adj = solveLinear(T.createCsr(), e);
  if ((int)adj.solution.size() != n) { r.error = "sensitivity: adjoint solve failed"; return r; }
  const std::vector<double> v(lastSolution_.begin(), lastSolution_.begin() + netlist.numGlobalNodes);
  auto lower = [](std::string s) { for (auto &c : s) c = (char)std::tolower((unsigned char)c); return s; };
  auto d = [&](ComponentType t, int idx) {
    const std::vector<double> b = sensRhs(netlist, t, idx, "R", v, n);
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += adj.solution[i] * b[i];
    return s;
  };
  auto add = [&](const std::string &name, double val) { r.names.push_back(lower(name)); r.values.push_back(val); };
  for (size_t i = 0; i < g.resistors.size(); ++i) {
    const double dR = d(ComponentType::RESISTOR, (int)i), R = g.resistors[i].resistance_ohms;
    add(g.resistors[i].name, dR);
    add(g.resistors[i].name + "_m", -R * dR);
    add(g.resistors[i].name + "_scale", R * dR);
  }
  for (size_t i = 0; i < g.voltageSources.size(); ++i) add(g.voltageSources[i].name, d(ComponentType::VSOURCE, (int)i));
  for (size_t i = 0; i < g.currentSources.size(); ++i) {
    const double dI = d(ComponentType::ISOURCE, (int)i);
    add(g.currentSources[i].name, dI);
    add(g.currentSources[i].name + "_m", g.currentSources[i].current_A * dI);
  }
  // ponytail: only top-level R, V and I values; every other device (and any subcircuit instance) is counted, not guessed.
  auto note = [&](const char *kind, size_t count) { if (count) r.unsupported.push_back(std::to_string(count) + " " + kind); };
  note("switch(es)", g.switches.size()); note("controlled source(s)", g.voltageControlledVoltageSources.size() +
      g.voltageControlledCurrentSources.size() + g.currentControlledVoltageSources.size() + g.currentControlledCurrentSources.size());
  note("behavioural source(s)", g.behavioural.size()); note("diode(s)", g.diodes.size());
  note("MOSFET(s)", g.mosfets.size()); note("BJT(s)", g.bjts.size()); note("JFET(s)", g.jfets.size());
  note("OSDI device(s)", g.osdiDevices.size()); note("transmission line(s)", g.transmissionLines.size());
  note("subcircuit instance(s)", netlist.instances.size());
  return r;
}

bool CircuitSim::coldStartSweep(TensorNetlist &netlist, double totalTime,
                                int nSubsteps,
                                std::vector<double> &finalVoltages) {
  if (nSubsteps <= 0 || totalTime <= 0.0 || netlist.numGlobalNodes == 0)
    return false;

  // All micro-steps use Backward Euler (GEAR_1_EULER).
  // The caller already set this before invoking coldStartSweep.
  const double dt_sub = totalTime / static_cast<double>(nSubsteps);

  // Seed from the last full solution (node voltages + branch currents)
  const int nTot = systemSize(netlist);
  std::vector<double> nrEstimate = transientSeed(netlist, nTot);

  for (int k = 0; k < nSubsteps; ++k) {
    // Sync soaBlock capacitor/inductor history from AoS netlist state.
    // This must happen at the start of each sub-step so stampAllElements
    // uses the correct companion-model source term for this micro-step.
    loadSoAFromNetlist(netlist);

    integrator->prepareStep(dt_sub);

    // Inner NR loop — identical structure to the one in stepTransient.
    syncJunctionLimits(netlist, nrEstimate);
    bool nrConverged = false;
    SolverResult subResult{};
    for (int iteration = 0; iteration < nrLimit(); ++iteration) {
      scheduler.schedule(netlist);
      matrixBuilder.clear();
      matrixBuilder.setDimensions(nTot, nTot);
      rhsVector.assign(nTot, 0.0);

      stampAllElements(netlist, dt_sub, nrEstimate, dt_sub * (k + 1));
      addDiagonalConditioning(netlist.numGlobalNodes, ptranGmin_);  // 0 except in a transient OP
      // (no node-to-ground gmin in transient: junction GMIN in the device
      // models keeps floating junction nodes referenced, like ngspice)

      Csr_matrix A = matrixBuilder.createCsr();

      subResult = solveLinear(A, rhsVector);

      std::vector<double> v_new = subResult.solution;

      double kcl_error = kclExcess(netlist, v_new, A, dt_sub, dt_sub * (k + 1));
      if (checkConvergence(v_new, nrEstimate, kcl_error,
                           abstol_)) {
        nrEstimate = v_new;
        nrConverged = true;
        break;
      }
      nrEstimate = v_new;
    }

    if (!nrConverged) {
      // NR failed for this micro-step — abort the sweep.
      return false;
    }

    updateReactiveHistory(netlist, nrEstimate, dt_sub);
    lteController_.recordStep(dt_sub);
    // Note: Inductor history write-back is not performed here, consistent with
    // stepTransient which also does not write back inductorState in its main
    // history-update block.  The soaBlock re-sync at the top of each sub-step
    // reads the current hist.i[0] and hist.v[0] values correctly.
  }

  finalVoltages = nrEstimate;
  return true;
}

SolverStep CircuitSim::stepTransient(TensorNetlist &netlist, double timeStep,
                                     double currentTime) {
  if (netlist.numGlobalNodes == 0)
    return {currentTime, {}, {0, 0.0, true}, 0};

  bool isFirstTransientStep = !arbiterInitialized;
  // When coldStartSweep() succeeds it updates all capacitor history internally,
  // so we must skip the main NR while-loop and the post-convergence
  // history-update block below.  This flag is set to true only when the sweep
  // converges.
  bool isFirstTransientStepHandled = false;

  std::vector<double> finalVoltages;
  SolverResult lastResult{}; // zero-initialize: iterations=0,
                             // finalResidual=0.0, converged=false
  bool simulationConverged = false;
  double next_dt_suggestion = timeStep;

  if (!arbiterInitialized) {
    arbiter.initialize(netlist);
    arbiterInitialized = true;
    lteController_.reset();

    // M4: Enforce Formal Passivity
    if (!arbiter.checkPassivity(netlist)) {
      // Halt simulation if physically impossible device parameters found
      return {currentTime, {}, {0, 0.0, false}};
    }

    // Use Backward Euler for all cold-start micro-steps.  TRAPEZOIDAL's
    // companion current I_eq = -G_eq*v_prev - i_prev is zero when both
    // hist.v and hist.i are zero (cold start / UIC ic=0), which gives a
    // systematically wrong first-step solution.  BE's companion model
    // I_eq = 0 (purely a conductance C/h) is correct for a zero-IC start
    // and doesn't depend on i_prev at all.
    setIntegrationMethod(IntegrationType::GEAR_1_EULER);

    // SPICE3-accurate cold-start: divide the first transient step into
    // COLD_START_SUBSTEPS internal BE micro-steps (dt_init = tstep/100).
    // Each micro-step updates the AoS capacitor history so that by the end
    // the companion-model state matches what SPICE3 produces at t = tstep.
    std::vector<double> sweepVoltages;
    bool sweepOk =
        coldStartSweep(netlist, timeStep, COLD_START_SUBSTEPS, sweepVoltages);
    if (sweepOk) {
      finalVoltages = sweepVoltages;
      simulationConverged = true;
      // Restore TRAPEZOIDAL now — hist.i has been recorded with BE formula
      // for all 100 micro-steps, so the next step can use the high-accuracy
      // second-order method with a valid hist.i.
      setIntegrationMethod(tranMethod_);
      isFirstTransientStepHandled = true;
    }
    // Fallback: if sweep failed (e.g. NR divergence on a stiff circuit),
    // isFirstTransientStepHandled remains false and the main NR loop below
    // executes a single BE step at timeStep — the original cold-start behavior.
  }

  // Initialize SoA Block for transient steps.
  // tensorizeNetlist() zero-inits v_prev/i_prev; we must carry the AoS history
  // (updated by the previous step's history-update loop) into the SoA arrays so
  // that stampSoABlock() uses the correct companion-model source term.
  loadSoAFromNetlist(netlist);

  double actualTimeStep = timeStep;
  int attempt = 0;
  const int MAX_ATTEMPTS = 4;

  // finalVoltages, lastResult, simulationConverged, next_dt_suggestion are
  // declared above (before the !arbiterInitialized block) so coldStartSweep can
  // populate them.

  if (!isFirstTransientStepHandled) {
    while (attempt < MAX_ATTEMPTS) {
      // each attempt starts from the accepted switch states
      forEachSwitch(netlist, [&](const SwitchRef &w, const std::array<int, 2> &) { *w.on = *w.prev; });
      integrator->prepareStep(actualTimeStep);

      const int nTot = systemSize(netlist);
      std::vector<double> nrEstimate = transientSeed(netlist, nTot);

      // Gap 4: Neural seeder dV/dt extrapolation for transient steps.
      // Uses the two most recent converged solutions to predict the next state.
      if (!prevVoltages_.empty() && prevVoltages_.size() == nrEstimate.size()) {
        auto seed =
            neuralSeeder_.predict(nrEstimate, prevVoltages_, actualTimeStep);
        if (!seed.empty()) {
          nrEstimate = std::move(seed);
        }
      }

      // ─── TAA: History Rejection ──────────────────────────────────────
      // Reject polynomial predictors for nodes undergoing violent switching.
      // Replaces their guess with first-order Euler to prevent NR divergence
      // without globally cutting the timestep.
      arbiter.rejectHistory(nrEstimate, actualTimeStep);

      bool nrConverged = false;
      syncJunctionLimits(netlist, nrEstimate);  // limit relative to this step's start
      for (int iteration = 0; iteration < nrLimit(); ++iteration) {
        scheduler.schedule(netlist);
        matrixBuilder.clear();
        matrixBuilder.setDimensions(nTot, nTot);
        rhsVector.assign(nTot, 0.0);

        stampAllElements(netlist, actualTimeStep, nrEstimate,
                         currentTime + actualTimeStep);
        addDiagonalConditioning(netlist.numGlobalNodes, ptranGmin_);  // 0 except in a transient OP
        // (no node-to-ground gmin in transient: junction GMIN in the device
        // models keeps floating junction nodes referenced, like ngspice)

        Csr_matrix A = matrixBuilder.createCsr();

        // --- Solver Strategy Optimization ---
        // LU is O(N^3) but robust for non-symmetric active circuits.
        // PCG is O(N*sqrt(K)) but requires SPD (Resistor/Capacitor meshes).
        lastResult = solveLinear(A, rhsVector);

        std::vector<double> v_new = lastResult.solution;

        double kcl_error =
            kclExcess(netlist, v_new, A, actualTimeStep, currentTime + actualTimeStep);
        if (checkConvergence(v_new, nrEstimate, kcl_error,
                             abstol_)) {
          nrEstimate = v_new;
          nrConverged = true;
          break;
        }
        nrEstimate = v_new;
      }

      if (!nrConverged) {
        actualTimeStep *= 0.5;
        attempt++;
        continue;
      }

      // A switch toggled in a step much longer than the step/200 resolution: bisect to the
      // crossing. ngspice (whose steps are what the user's tstep allows) toggles at the end of
      // whichever step crosses; this lands within step/200 of the threshold instead.
      bool toggled = false;
      forEachSwitch(netlist, [&](const SwitchRef &w, const std::array<int, 2> &) { toggled = toggled || *w.on != *w.prev; });
      if (toggled && tran_.step > 0.0 && actualTimeStep > tran_.step / 200.0) {
        actualTimeStep *= 0.5;
        continue;
      }

      // ─── Gap 5: LTE-Based Timestep Control ─────────────────────────────
      // Estimate local truncation error BEFORE the arbiter's stability check.
      // If LTE exceeds tolerance, reject the step and retry with smaller dt.
      auto lteResult = lteController_.estimateLTE(
          netlist, nrEstimate, actualTimeStep, integrator->getType(),
          [&](const std::function<void(double, const std::array<double, 3> &, double)> &visit) {
            // diode / BJT charge branches, internal nodes included (as the MOS branches)
            auto getV = [&](int n) { return n > 0 && n <= (int)nrEstimate.size() ? nrEstimate[n - 1] : 0.0; };
            forEachJunction(netlist, [&](const Junction &j) {
              if (!j.st || !junctionHasCharge(j)) return;
              std::array<double, 7> x{};
              for (int k = 0; k < j.terms; ++k) x[k] = getV(j.n[k]);
              double v[5], q[5];
              const int K = junctionCharges(j, x, v, q);
              for (int k = 0; k < K; ++k) {
                const double dv = std::abs(v[k] - j.st->v[0][k]);
                visit(v[k], {j.st->v[0][k], j.st->v[1][k], j.st->v[2][k]}, dv > 1e-9 ? std::abs(q[k] - j.st->q[k]) / dv : 0.0);
              }
            });
            // OSDI devices: their local node voltages
            forEachOsdi(netlist, [&](const OsdiRef &r) {
              const auto &h = r.in->vHist;
              for (size_t k = 0; k < r.n.size(); ++k) visit(getV(r.n[k]), {h[0][k], h[1][k], h[2][k]}, -1.0);
            });
          });

      if (lteResult.requiresRejection && (lteHold_ == 0 || lteResult.deviceRejects)) {
        // LTE too large — timestep was too aggressive
        actualTimeStep = lteResult.suggestedDt;
        attempt++;
        continue;
      }

      // ponytail: no automatic TRAP -> GEAR_2 switch on "ringing" (here or for
      // arbiter suggestion 2 below). Every GEAR_2 formula (stamps, audit, history,
      // inductor rows) has constant-step BDF2 coefficients, and the switch came
      // when h was changing: each switch injected a jump (~40 mV in
      // hier_rc_ladder_pulse). Variable-step BDF2 coefficients are the upgrade.

      // Original arbiter stability analysis (energy monitor, Lyapunov check)
      // The arbiter tracks node voltages only (it re-initialises on a size change)
      int suggestion = arbiter.analyzeStability(
          netlist,
          std::vector<double>(nrEstimate.begin(), nrEstimate.begin() + netlist.numGlobalNodes),
          actualTimeStep, next_dt_suggestion);

      if (suggestion == 1) {
        actualTimeStep *= 0.5;
        attempt++;
        continue;
      }

      // Use LTE-based timestep suggestion if it's more conservative
      double lte_dt =
          lteController_.suggestTimestep(actualTimeStep, lteResult.maxLTE,
                                         lteTolerance_, integrator->getType());
      if (lte_dt < next_dt_suggestion) {
        next_dt_suggestion = lte_dt;
      }

      finalVoltages = nrEstimate;
      simulationConverged = true;
      timeStep = next_dt_suggestion;
      break;
    }
  } // end if (!isFirstTransientStepHandled)

  // Hang Protection: Detect if timeStep has underflowed
  if (timeStep < 1e-18) {
    // Prevent infinite loop by returning failing step
    return {currentTime,
            std::vector<double>(netlist.numGlobalNodes, 0.0),
            {lastResult.iterations, lastResult.finalResidual, false}};
  }

  // When coldStartSweep() handled the first step, all capacitor history was
  // updated inside the sweep loop.  Skip the history-update block to avoid
  // double-updating the companion-model state.
  if (simulationConverged && !isFirstTransientStepHandled) {
    updateReactiveHistory(netlist, finalVoltages, actualTimeStep);

    lteController_.recordStep(actualTimeStep);

    // Restore TRAPEZOIDAL after the first step (which used Backward Euler
    // for cold-start accuracy).  The restore must happen AFTER the cap
    // history update above so that hist.i is recorded with the BE formula,
    // not the TRAPEZOIDAL formula — otherwise i_prev would be doubled.
    if (isFirstTransientStep &&
        integrator->getType() == IntegrationType::GEAR_1_EULER) {
      setIntegrationMethod(tranMethod_);
    }

  }

  if (!simulationConverged)
    return {currentTime,
            std::vector<double>(netlist.numGlobalNodes, 0.0),
            {lastResult.iterations, lastResult.finalResidual, false},
            netlist.globalBlock.topologyHash};

  // Gap 4: Record converged transient solution for neural seeder
  prevVoltages_ = finalVoltages;
  neuralSeeder_.recordSolution(finalVoltages, lastResult.finalResidual,
                               lastResult.iterations);
  lastSolution_ = finalVoltages;                    // full x seeds the next step
  std::vector<double> branches(finalVoltages.begin() + std::min<size_t>(netlist.numGlobalNodes, finalVoltages.size()),
                               finalVoltages.begin() + std::min<size_t>(netlist.numGlobalNodes + branchCount(netlist), finalVoltages.size()));
  finalVoltages.resize(netlist.numGlobalNodes);     // API returns node voltages

  return {
      currentTime + actualTimeStep,
      finalVoltages,
      {lastResult.iterations, lastResult.finalResidual, simulationConverged},
      netlist.globalBlock.topologyHash,
      std::move(branches)};
}

