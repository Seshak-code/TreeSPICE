# ngspice differential tests

Runs every `netlists/*.cir` through TreeSPICE (`acutesim_engine_dump`, cpu and gpu modes) and through
ngspice, then compares all top-level node voltages and the branch currents `I(V)`, `I(L)`, `I(E)`, `I(H)`, `I(B)` of top-level sources
(linear E/H and B voltage sources only: ngspice has no branch vector for POLY/VALUE forms). Statuses: PASS, FAIL, SKIPPED (an analysis form the
harness cannot emulate), ENGINE-UNSUPPORTED (engine rejects the card, e.g. F/H/T/PWL/EXP/AREA). Exit code is 1 on any FAIL.

Prerequisites: `brew install ngspice`; `cmake --build build --target acutesim_engine_dump`; python3 (stdlib only).

    python3 tests/ngspice_diff/run_diff.py [-k name-substring] [--modes cpu,gpu] [-v]

The analysis comes from the netlist's own `.op` / `.dc` / `.tran` card. `.dc` (one V/I source) runs natively
in the engine (one row per sweep point). For ngspice the analysis/print cards are
dropped and a `.control` block runs the analysis and `wrdata`s the nodes. Both run at 27 C; on the ngspice
side, for `.tran`, `tmax = tstep/10` so the reference is
dense (engine samples are compared against it by linear interpolation; PULSE edges still follow `tstep`).

Branch currents: DC/AC |err| <= 10 nA + 0.1 %; transient as a node (10 % of swing or 2 % of peak, 1 uA floor), an instant
outside that passes when the mean over its `.tran` step agrees. `NO_BRANCH_CHECK` in `run_diff.py` lists the netlists whose
currents are skipped, with the reason. `tests/cli_check.py` checks the `treespice` `.print` tables and raw files.

Tolerances (see comment in `run_diff.py`; flags `--dc-atol --dc-rtol --tran-atol --tran-rtol`):
DC |err| <= 200 uV + 0.1 % of the reference (1 mOhm Norton sources/shorts give ~I*1 mOhm; 0.1 % is ngspice RELTOL).
Transient |err| <= max(5 mV, 2 % of the node's swing in the reference waveform), against the reference anywhere
within +-tstep/10 of the sample's time (timing tolerance: a steep edge otherwise demands sub-ps agreement).

Adding a circuit: drop a `.cir` with one analysis card in `netlists/`. Hierarchical cases use `.subckt`; only
top-level nodes are compared. Netlists whose floating nodes are set only by gmin (e.g. `aac_10_bridge`) are
ill-posed for comparison; `x_bridge_loaded` is the well-posed variant.

`aac_05..09` come from SPICE2-era examples whose sources read `ac <mag> sin`; the bare `sin` (no arguments)
is rejected by ngspice and meant nothing there, so it was removed. `.ac` rows compare complex node voltages
with the DC tolerances.
