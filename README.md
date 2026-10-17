# TreeSPICE (Tensor Recursive Elimination Engine)

**TREE — Tensor Recursive Elimination Engine.** A SPICE-equivalent circuit simulation engine built
on a contraction-tree direct solver.

> **Status: in development.** TreeSPICE solves standard SPICE netlists (DC, sweeps, transient,
> AC, noise) and is checked against ngspice and Xyce on every change. It is not yet a
> production simulator: device-model coverage stops at MOSFET level 1, and the GPU path is
> still being built. Everything below marked *planned* or *open* does not exist yet.

TreeSPICE is the simulation kernel of the OpenVLSI chip-design environment, developed here as a standalone library. Its long-term
goal is one application where you can design an SoC, attach RAM, run firmware on it and see
the program's output, with transistor-level simulation available wherever it matters.

---

## What TreeSPICE is

TreeSPICE solves the same equations as SPICE. Each Newton step of a modified nodal analysis
(MNA) system is linearised and solved exactly. The **way** that linear system is solved is
the project's core idea.

### The solver: superposition up an elimination tree

The circuit graph is split recursively by nested dissection into a tree:

```
                 [root separator]            ← dense Schur complement, solved last
                 /              \
        [separator]            [separator]   ← each adds its children's Schur complements
         /      \               /      \
     [block]  [block]       [block]  [block] ← independent: eliminated in parallel
```

- Each leaf block eliminates its internal unknowns and passes a dense **Schur complement**
  (its effect on the boundary nodes) up to its parent.
- A parent **adds** its children's contributions. This is superposition: independent
  subcircuits combine linearly at their shared boundary.
- All fronts on one tree level are independent and run in parallel.
- The pipeline is: MC64-style matching, then nested-dissection ordering, then threshold partial pivoting
  (u = 0.01) with delayed pivots, then iterative refinement.
- The plan is computed once per sparsity pattern and reused for every Newton iteration and
  time step.

In numerical linear algebra this is a **multifrontal factorization**. It is also the same
object as a tensor-network contraction tree: both are a tree decomposition of the circuit
graph, and cotengra-style contraction planners use the same recursive bipartitioning as
nested dissection. TreeSPICE schedules elimination as a contraction tree. That tree is the
backbone every scaling technique on the roadmap builds on: latency reuse, isomorphism
reuse, PDN factor caching, compressed separators and batched GPU fronts.

**What "contraction" does *not* mean here.** Each intermediate in TreeSPICE is a dense
*w × w* Schur complement over a separator of *w* unknowns. That costs O(w²) memory and
O(w³) flops. It is **not** an O(2ʷ) tensor as in quantum-circuit simulation. Ideas built for
discrete tensor networks therefore do not carry over to TreeSPICE: index slicing, tensor-train
Jacobians, and tensor decision diagrams. Their real counterparts are out-of-core
factorization, low-rank (Sherman–Morrison–Woodbury) updates, and instancing (see
[References](#references)).

### Measured today (Apple M4 Pro, CPU)

| Resistor-grid size | DC solve |
|---|---|
| 10⁴ nodes | 0.02 s |
| 10⁵ nodes | 0.11 s |
| 10⁶ nodes | 3.0 s |

From 10⁵ to 10⁶ nodes the time grows with an exponent of about 1.43. That matches the
O(N^1.5) nested-dissection bound for planar graphs (Lipton–Rose–Tarjan 1979). This is the
honest expected scaling, not a claim of beating it.

---

## Current capabilities

| Area | Supported |
|---|---|
| Analyses | `.op`, `.dc` sweeps, `.tran` (trapezoidal, LTE step control, breakpoints, `uic` / `.ic`), `.ac`, `.noise`, `.pz` (poles/zeros, dense, up to 500 unknowns), `.sens` (DC adjoint sensitivity to top-level R, V and I values; device parameters and AC `.sens` are reported as not supported) |
| Elements | R, C, L, K (mutual inductance), V/I sources (DC, PULSE, SIN, AC), E/F/G/H controlled sources, J (Shichman–Hodges), X subcircuits (nested, local node scoping) |
| Diode | SPICE level 1: IS, N, RS (internal node), CJO/VJ/M/FC, TT, BV/IBV breakdown, instance AREA/M |
| BJT | Gummel–Poon: IS, BF/BR, NF/NR, VAF/VAR, IKF/IKR, ISE/NE, ISC/NC, RB/RC/RE (internal nodes), CJE/CJC with XCJC, CJS (substrate node), TF/XTF/VTF/ITF, TR, AREA |
| MOSFET | Level 1 complete: source/drain swap, body effect, bulk junctions, Meyer gate charge, overlap and junction capacitances |
| Compiled Verilog-A | BSIM4, BSIM-CMG and other OpenVAF models through OSDI (`.osdi <file>`, ngspice's `N` element): DC, transient, AC, noise (flicker included) |
| Netlist | SPICE syntax with SI suffixes, `.model`, `.param` substitution, `.subckt`, `.include`, `.lib <file> <corner>`. Unknown or unsupported cards are **hard errors**, never silently dropped |
| Convergence | Plain Newton first, then ngspice's dynamic gmin stepping, then source stepping; gmin across each PN junction as in ngspice; responses fail closed on NaN or non-convergence |
| Determinism | No `-ffast-math`, round-to-nearest enforced, deterministic parallel solve |

**Not yet supported:** `.param` arithmetic expressions, bias-dependent base resistance (IRB/RBM), excess phase (PTF), flicker noise (KF) of built-in devices, OSDI models that keep state or use `$limit`,
EXP/PWL/SFFM sources, transmission lines, PZ analysis. The GPU mode currently runs the CPU
path.

### How it is verified

Correctness is established by **running** reference simulators. Checklists do not count.

- **ngspice differential harness:** over 100 netlists covering op, dc, tran, ac and noise.
  Each one is compared waveform-by-waveform against ngspice, on both the CPU and GPU modes.
- **Xyce regression subset:** 78 device and analysis cases, 0 failures.
- **IBM power-grid benchmarks** (Nassif, ASP-DAC 2008): 6 of 6 match the published
  solutions. The largest, ibmpg6, solves in 37 s.
- Every engine fix adds a regression check derived from an analytic or hand-derived reference.

Known issues are tracked with reproducer netlists, and each fix is retested before it is marked done.

---

## Where it is going: the roadmap

The scale target is stated honestly:

- Flat, SPICE-accurate simulation of 10⁹ transistors does not fit on one GPU. The matrix
  alone needs roughly 60 GB before fill, and at O(N^1.5) one factorization would take on the
  order of a day.
- No commercial tool claims flat SPICE accuracy at that size. The largest documented
  near-SPICE run is HSIM's 284-million-transistor DRAM, which relied on hierarchy.
- TreeSPICE's billion-scale goal is therefore **10⁹ device instances through hierarchy,
  isomorphism and exact reuse on the elimination tree**. Accuracy is checked against flat
  simulation on 10⁷-scale subsets.
- Firmware runs on RTL and gate-level models. SPICE handles the blocks that need it: SRAM,
  PLL, IO and power. The two are co-simulated.

Each item below is tagged **Established** or **Open**:

- **Established:** a published, reproducible technique. The risk is engineering only.
- **Open:** no published solution at this scale or in this setting. The risk is research.

Every exit gate is pass/fail. Accuracy is measured against ngspice or Xyce, and solver speed
against KLU.

### ✅ Phase 1: Foundations (done)
MNA with branch currents, the contraction-tree solver as the only linear solver, a
correct subcircuit hierarchy (hierarchical results bit-identical to flattened), and the
ngspice differential harness.

### ✅ Phase 2: Analysis parity (done)
Real AC and noise analysis, `.dc` sweeps, DC-seeded transient, `uic`/`.ic`, and breakpoints at
waveform corners, all running through one transient loop shared by the API and the GUI.

### 🔨 Phase 3: Device-model parity and a trustworthy CPU baseline (in progress)

| Item | Status |
|---|---|
| MOSFET level 1, DC and charges | ✅ done |
| Diode RS/CJO/TT/BV, Gummel–Poon BJT with junction capacitances | ✅ done |
| BSIM4, BSIM-CMG and PSP through an OSDI loader compiled with OpenVAF (the ngspice ≥ 39 route), not hand ports | ✅ done (BSIM4, BSIM-CMG verified against ngspice) |
| `.lib` sections and process corners (PDK ingestion) | ✅ done (`.param` expressions still open) |
| Dynamic gmin stepping (ngspice `dynamic_gmin`) | ✅ done |
| Source stepping | ✅ done |
| **Solver benchmark against KLU** on the SuiteSparse circuit matrices. KLU's authors found multifrontal methods ineffective on tiny-front transistor netlists, so TreeSPICE must win there too, not only on resistor grids | ✅ measured 2026-10-13 (`bench/results-2026-10-13.md`): 3–1,900× slower than KLU on transistor-level matrices, from ordering fill. Gap accepted on the condition that GPU acceleration (Phase 4) proves to close it |
| **Condition-number survey** of real MNA matrices, with and without gmin and scaling. This decides whether low-precision factors are viable | ✅ measured 2026-10-15 (`bench/cond-2026-10-15.md`): 9 of 17 SuiteSparse circuit matrices have equilibrated κ ≤ 2.5e6 (FP32 fronts + FP64 refinement viable); 8 reach 7.7e8–1.9e12 (need GMRES-based refinement or FP64 fronts); gmin does not matter |
| Per-front dirty flags with exact reuse of an unchanged subtree's Schur complement (CPU prototype) | ✅ done (`ContractionCache`): bit-identical to refactoring; one changed entry refactors 6 of 177 fronts. A grid + logic transient gains only 13 %: LTE changes h, and h enters every capacitor stamp (Phase 4: per-subtree multirate) |

**Gate:** standard-cell delay and power within 1 % of ngspice on a real PDK. DC converges on
every case that ngspice converges on. Factorization time within 2× of KLU on the SuiteSparse
circuit set, or the gap explained with front-size data.

### Phase 4: Single-GPU scale (target 10⁷–10⁸ elements)

| Item | Status |
|---|---|
| Batched GPU device evaluation with graph-coloured, lock-free stamping. Device evaluation is 66–75 % of SPICE runtime, so on its own this caps the speedup near 3–4× (Amdahl) | Established |
| **GPU multifrontal factorization:** level-scheduled tree, batched small dense LU at the leaves, dense tensor-core LU at the top separators, high-fanout supply nets placed in the top separator, delayed pivoting on the GPU. Baselines are cuDSS and GLU | Open |
| **FP32 front factorization with FP64 residual refinement**, two-sided equilibration and FP64 fallback. Apple GPUs have no FP64, so refinement runs on the CPU over unified memory. FP16/TF32 only if the Phase 3 condition-number survey supports it: classic refinement with FP16 is guaranteed only up to κ ≈ 10⁴, GMRES-IR up to about 10⁸ | Established in general, open for circuit matrices |
| **PDN as a linear subtree:** factor once, then fixed-step backward-Euler or trapezoidal back-solves each step, coupled to device subtrees through the Schur interface. This is not an explicit update, which is unstable on stiff grids | Established |
| **Isomorphism reuse:** identical cell instances share symbolic structure; instances in the same electrical state (terminal voltages and local supply within tolerance) share numeric factors | Established on CPU (HSIM), open on GPU |
| **Latency-aware refactorization:** quiescent subtrees reuse their cached factor exactly (no approximation), with a per-subtree multirate step, because the step size enters every capacitor stamp | Open (flagship) |
| TICER reduction of extracted RC parasitics before ordering | Established |
| Code generation from OpenVAF IR to GPU kernels for compact models | Open |

**Gate:** a 10⁷-element post-layout netlist fits on one 24–32 GB GPU. End-to-end transient
runs at least 3× faster than TreeSPICE's own multicore CPU path on the same machine. Waveforms
stay within RELTOL of ngspice. Fewer than 1 % of solves fall back to FP64. A 10⁸-element run
completes, and memory and time per element are reported.

### Phase 5: SoC flow (10⁹ instances through hierarchy)

| Item | Status |
|---|---|
| Event-driven digital / RTL / firmware kernel coupled to the analog tree through A2D/D2A port stamps | Established |
| Multirate integration with interface error control, partitioned by measured activity | Established theory, open at scale |
| Hierarchical netlist storage: about 10⁵ unique cells covering 10⁹ instances | Established |
| PDN: rational-Krylov exponential integrators for adaptive steps; sparsifier-preconditioned CG when the direct factor does not fit | Established |
| Block-low-rank or HSS compression of large PDN separators, inside iterative refinement | Established for elliptic problems, open for circuits |
| Out-of-core factor storage to host RAM | Established |
| Unified flow in the app: schematic/RTL → Yosys → OpenROAD → extraction → simulation → sign-off | Engineering |

**Gate:** "hello world" firmware runs on a reference SoC, with SRAM timing taken from SPICE
characterisation. A 10⁹-instance design runs a non-trivial transient on one workstation. On
10⁷-scale flat subsets, the hierarchical results match flat TreeSPICE and ngspice within stated
tolerances. The accuracy cost of isomorphic state-matching is measured, not asserted.

### Research directions

These are ranked by fit with TreeSPICE's architecture. Each is publishable only with measurements.

1. **Latency-aware multifrontal refactorization.** Exact subtree reuse, per-subtree multirate
   steps, and repartitioning as activity moves. Up to 80–90 % of a large circuit is latent
   at any instant (Günther & Rentrop 1993); this would turn that into exact solver savings.
2. **A GPU multifrontal for circuits** that is efficient on both tiny transistor fronts and
   large dense separators, with pivoting. The sparse LU is the recurring ceiling of every
   published GPU-SPICE effort.
3. **Mixed-precision factorization characterised on circuit matrices:** condition numbers,
   scaling and fallback statistics. No peer-reviewed circuit demonstration exists yet.
4. **GPU isomorphic batching with certified state-equivalence error.**
5. **Compressed (BLR/HSS) separators** inside a nonlinear transient.
6. **Batched corners and Monte Carlo** sharing one symbolic tree.
7. **Adjoint / differentiable transient analysis** that reuses the factor for transpose solves.
8. **Time-parallel (Parareal/MGRIT) PDN transient.**
9. **A neural compact-model fast tier** with a BSIM fallback.
10. **Tensor-train uncertainty quantification over process parameters.** This is where
    tensor-train methods legitimately apply in EDA.

### Ideas considered and rejected

| Idea | Why not | Instead |
|---|---|---|
| Treat the global solve as a scalar tensor contraction with O(2ʷ) intermediates | Linear elimination stays polynomial: w×w fronts | Multifrontal elimination scheduled as a contraction tree |
| Slice tensor indices to fit VRAM | Fronts have no discrete index to enumerate | Out-of-core fronts; tearing with low-rank correction |
| Compress quiescent blocks with truncated SVD | A quiet block is not a low-rank block | Exact reuse of the unchanged factor |
| Explicit PDN update V<sub>n+1</sub> = F·V<sub>n</sub> + I | Unstable on stiff grids at useful steps | Implicit or exponential step applied through a reused factorization |
| Spectral sparsification to lower treewidth | Removing edges from a mesh does not reduce its treewidth | TICER/PRIMA reduction; sparsifiers as preconditioners |
| One "unified" tree for analog, power and billions of logic cycles | Firmware-length runs need event-driven digital simulation | Event-driven kernel coupled at port stamps |
| Learned ranking of contraction orders | The order is computed once; exact symbolic cost picks among candidates | Evaluate 3–5 orderings symbolically |

---

## Building and testing

TreeSPICE is a C++17 CMake project with no required dependencies. Dawn/WebGPU is used when present.

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build                      # engine regression suites
python3 tests/ngspice_diff/run_diff.py      # waveform diff against ngspice (needs ngspice on PATH)
python3 tests/xyce_diff.py                  # Xyce regression gold (needs datasets/xyce)
python3 tests/ibmpg_diff.py -k ibmpg2       # IBM power-grid benchmarks (needs datasets/ibmpg)
python3 bench/klu_bench.py                 # solver vs SuiteSparse KLU (needs KLU; downloads matrices)
build/acutesim_engine_dump <netlist> cpu tran <step> <stop>   # machine-readable waveforms
```

### Command line

```bash
build/treespice circuit.cir                   # runs the netlist's .op/.dc/.ac/.tran/.noise/.pz/.sens cards, prints the .print tables
build/treespice -b -o run.log -r out.raw circuit.cir   # batch, log to a file, ngspice raw file (--ascii for text)
build/treespice --tran 1n 10u circuit.cir     # override the analysis; numbers take SI suffixes
build/treespice --help
cmake --install build --prefix ~/treespice    # bin/treespice, lib/libacutesim_engine.a, include/treespice/engine_api/
python3 tests/cli_check.py                    # .print tables and raw files against ngspice
```

`.print`, `.save` and `.probe` take v(a), v(a,b), i(v1) (a V, L, E, H or B-voltage source), expressions over them
(`v(a)*2`), and for `.ac` vdb(), vm(), vp(), vr(), vi(), db(), mag(), ph(), real(), imag(). Device currents
(`i(r1)`, `@m1[id]`) are an explicit error. Subcircuit nodes and sources are named as in ngspice (`x1.n1`, `v.x1.v1`).

`datasets/` is git-ignored. See `ENGINE_HANDOFF.md` for where to download the Xyce and IBM sets.

### Using TreeSPICE from another project

The public API is `engine_api/`: `ISimulationEngine`, sessions, request/response DTOs and callbacks.
The netlist data model (`netlist/circuit.h`, `netlist/spice_parser.h`) is also public for now.
Everything else is internal and may change. OpenVLSI consumes TreeSPICE as a git submodule
at `acutesim_engine/`, using `add_subdirectory(acutesim_engine)` and linking `acutesim_engine`.

## Repository layout

```
engine_api/    public API: ISimulationEngine, sessions, DTOs, callbacks
autodiff/      Dual-number forward-mode AD (device Jacobians)
math/          contraction solver, FPU control, integration methods
netlist/       SPICE parser, circuit model, canonical topology hash
physics/       stateless device models, MNA stamps, KCL audit
tensors/       structure-of-arrays storage of the linear elements; device-model resolvers
solvers/       DC / transient / AC / noise, LTE control, analysis registry
infrastructure/  scheduling, tracing, Monte Carlo
cli/           treespice (the command-line simulator), acutesim_engine_dump (machine-readable waveforms)
tests/         C++ suites, ngspice / Xyce / IBM power-grid differential harnesses
engine_impl.*  session API (.op/.dc/.tran/.ac/.noise)
```

## References

- R. J. Lipton, D. J. Rose, R. E. Tarjan, "Generalized nested dissection," *SIAM J. Numer. Anal.* 16(2), 1979.
- T. A. Davis, E. Palamadai Natarajan, "Algorithm 907: KLU, a direct sparse solver for circuit simulation problems," *ACM TOMS* 37(3), 2010.
- J. Gray, S. Kourtis, "Hyper-optimized tensor network contraction," *Quantum* 5, 410, 2021.
- I. L. Markov, Y. Shi, "Simulating quantum computation by contracting tensor networks," *SIAM J. Comput.* 38(3), 2008.
- P. Amestoy, A. Buttari, J.-Y. L'Excellent, T. Mary, "On the complexity of the block low-rank multifrontal factorization," *SIAM J. Sci. Comput.* 39(4), 2017.
- E. Carson, N. J. Higham, "Accelerating the solution of linear systems by iterative refinement in three precisions," *SIAM J. Sci. Comput.* 40(2), 2018.
- N. J. Higham, T. Mary, "Mixed precision algorithms in numerical linear algebra," *Acta Numerica* 31, 2022.
- H. Zhuang et al., "Simulation algorithms with exponential integration for time-domain analysis of large-scale power delivery networks," *IEEE TCAD*, 2016.
- S. R. Nassif, "Power grid analysis benchmarks," ASP-DAC 2008.
- C. W. Gear, D. R. Wells, "Multirate linear multistep methods," *BIT* 24, 1984.
- M. Günther, P. Rentrop, "Multirate ROW methods and latency of electric circuits," *Appl. Numer. Math.* 13, 1993.
- E. Lelarasmee, A. Ruehli, A. Sangiovanni-Vincentelli, "The waveform relaxation method for time-domain analysis of large scale integrated circuits," *IEEE TCAD* 1(3), 1982.
- B. N. Sheehan, "TICER: Realizable reduction of extracted RC circuits," ICCAD 1999.
- A. Odabasioglu, M. Celik, L. Pileggi, "PRIMA: Passive reduced-order interconnect macromodeling algorithm," *IEEE TCAD* 17(8), 1998.
- K. Gulati et al., "Fast circuit simulation on graphics processing units," ASP-DAC 2009.
- Z. Zhang et al., "Tensor computation: a new framework for high-dimensional problems in EDA," *IEEE TCAD*, 2017.
