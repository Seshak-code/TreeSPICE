# S3: simulating 1e9-cell arrays (design, for review)

Status: proposal, 2026-10-24. No code yet. S1 (memory per device, PR #21) is the base.

## Goal and gate

Simulate an SRAM-like array of 1e9 cells (6e9 transistors) on one machine (M4 Pro, 24 GB). Cover DC, transient and
AC, with answers equal to a flat solve of the same netlist within roundoff. Gate: a 1e9-cell, hierarchically
written 6T SRAM. Run a write and then a read of one word through a transient, within 24 GB. Also check equivalence
against the flat engine on arrays small enough to run flat (1e3 to 1e5 cells).

## What blocks it today (measured or read in the code)

Every per-instance record is too big at 1e9. Even 16 bytes per cell is 16 GB.

| Where | Per instance today | At 1e9 cells |
|---|---|---|
| Parser: nested `X` is expanded into flat leaf instances (`expandInstance`) | `BlockInstance` 96 B + node map + path string | > 100 GB |
| Parser: every internal node gets a global number, and `nodeIndex` holds a `"X1.X2.N"` string per internal node | ~2 strings per cell | > 100 GB |
| `BlockState` per instance (charge histories, OSDI state) | 288 B + 290 B/device | ~2 TB |
| Solution, residual and Newton vectors (one per unknown) | ~2 internal nodes × 8 B × several vectors | ~100 GB |
| Contraction plan and factors (one front per small group of variables) | ~0.6 KB per front | ~600 GB |
| Stamping: `branchSlots` builds a `std::function` per instance per call, and every device is stamped | O(instances) per Newton iteration | hours per iteration |

So the hierarchy has to survive parsing, and the solve has to touch each distinct cell, not each instance.

## Approach: solve the quotient circuit

Partition nets and instances into classes. Two instances share a class when they instantiate the same definition,
their elements have equal values, they see equal stimuli and equal initial state, and their port nets are in
equal classes. This is the **coarsest equitable partition** of the circuit graph. "Equitable" means every net in
class A meets the same number of element terminals of each class. It is computed by color refinement
(one-dimensional Weisfeiler–Leman):
- Initial colors: element kind, model, parameter values, source waveform, `.ic`/`.nodeset`.
- Refinement step: a net's color becomes its multiset of (incident element class, terminal); an instance's becomes
  (definition, the colors of its port nets).
- Repeat until stable.

On an equitable partition, the MNA system's solution is constant on each class whenever the solution is unique.
It is the solution of the **quotient system**, with one unknown per net class. In KCL for net class A, each
element class E contributes with its incidence count (how many E terminals one net of A meets). The quotient
matrix is unsymmetric in general; the contraction solver already handles that (LU with threshold pivoting).

Cells with identical inputs therefore share one unknown set, one device evaluation, one charge history and one
front. This holds by construction, not by comparing values. Value comparison would fail anyway: a flat solve's
roundoff already breaks bitwise equality between symmetric cells. It also fits the contraction bet: a class's
Schur complement enters its parent with its multiplicity. That is superposition, with k equal terms added once,
scaled by k.

### Why the partition is static

A circuit is deterministic. Instances that start in one class, with equal stimuli and equal state, stay equal for
the whole transient. So the partition is computed once, from topology, values, stimuli and initial conditions, and
needs no copy-on-write during a run. Classes split only where the inputs say they differ:
- the addressed row's word line;
- the written columns' bit lines;
- cells initialised to 0 versus 1.

During an idle-array read of one word, the class count is O(word width + a few). It does not grow with the array.

### Limits (deliberate)

- **Exact equivalence only.** Merging cells whose values differ slightly (post-layout parasitics, Monte Carlo
  mismatch) is FastSPICE's approximation. It is out of scope; such cells simply get classes of their own and cost
  what they cost today.
- **Symmetric metastable solutions.** Without `.ic` or `.nodeset`, a class of bistable cells is solved at its
  symmetric point. Each cell sits at Vdd/2, where a flat solve would let roundoff pick 0 or 1 cell by cell. That
  is reported, not silently returned: a warning names the class and asks for an initial condition.
- **Analyses:**
  - **Noise:** each instance's sources are independent, so a class's output noise power scales by its
    multiplicity (power, not amplitude).
  - **.sens / .tf / .pz:** these see the quotient. A sensitivity to one instance's parameter splits that
    instance out of its class first.

## Stages

Each stage lands with the full matrix (ctest, run_diff, Xyce, cli_check) and its own gate.

1. **S3.0: hierarchical netlist.**
   - **Change:** the parser keeps the instance tree (a definition, its child instances, their port bindings), not
     flat leaf instances. Internal nodes are numbered per definition. Hierarchical names (`x1.x2.n`) resolve on
     demand.
   - **Gate:** parse a 1e9-cell SRAM written as nested subcircuits (bank → row → cell) in O(definitions) memory.
     Flat solves stay identical, through a flattening view used by today's code path.
2. **S3.1: partition.**
   - **Change:** color refinement over the hierarchy, without flattening. An instance's color depends only on its
     definition and its port colors, so a definition refines once per distinct port-color context.
   - **Gate:** a 1e9-cell idle SRAM has O(10) classes; a one-word read has O(word width) more.
3. **S3.2: quotient solve.**
   - **Change:** assemble the quotient MNA (class unknowns; element classes stamped once, scaled by incidence
     counts); DC and transient through the contraction solver; outputs map an instance's node to its class.
   - **Gates:**
     - Flat equivalence on 1e3 to 1e5-cell arrays (max |dV| ≤ 1e-9 V, the same accepted steps).
     - 1e6- and 1e9-cell write-then-read transients: memory and time recorded.
4. **S3.3: the rest.**
   - **Change:** AC, noise (power multiplicity), `.meas`/`.save` on class-mapped names, `.ic`/`.nodeset` per
     instance (they split classes), Monte Carlo (per-instance variation splits too).
   - **Gate:** run_diff and Xyce unchanged with S3 on; equivalence on small arrays per analysis.

## Decisions for the owner

1. **Exact equivalence only (recommended).** No tolerance merging. Answers equal the flat solve's within
   roundoff; parasitic-extracted arrays gain little.
2. **Metastable warning versus error.** For a bistable class without an initial condition: warn (recommended),
   or refuse the run.
3. **S3.0 changes the parse result's internals.** The flat `instances` list becomes a view, and the GUI and
   driver code that walk `instances` move to that view. That's no change in behaviour, but it touches
   `compute/orchestration`.
