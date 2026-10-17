// acutesim_engine_dump <netlist> <cpu|gpu> <dc|tran|ac|noise|pz|sens> [step stop]
// Machine-readable waveform dump for differential testing (tests/ngspice_diff).
// Prints: "CONVERGED <0|1>", "ERROR <text>", "BACKEND <what ran>", then one line per waveform point:
//   <time> <v1> <v2> ...   (engine-internal node order)
//   ac: <freq> <re1> <im1> <re2> <im2> ...
//   then the branch currents (V/L/E/H/B-voltage, into the + terminal as ngspice's i(name)), after all the
//   voltages: <i1> <i2> ... (ac: <re> <im> pairs). "NODES n1 n2 ..." ("-" if unnamed) and
//   "BRANCHES b1 b2 ..." (lower case) precede the points and name the columns in order.
//   (env TREESPICE_BASEDIR overrides the directory .include/.lib paths resolve against)
//   pz: no waveform; "POLE <re> <im>" and "ZERO <re> <im>" lines
//   sens: "SENS <name> <d(output)/d(parameter)>" lines, "UNSUPPORTED <element>" for devices left out
//   noise: <freq> <onoise> <inoise> ... then "TOTAL <onoise_total> <inoise_total>"
// A nested .dc / .temp list adds "SWEEP <name>=<value> ..." before the first row of each outer value
//   (the row's first column is the innermost sweep value).
#include "engine_api/isimulation_engine.h"
#include "engine_api/simulation_session.h"
#include "engine_api/simulation_dto.h"
#include "engine_api/engine_callbacks.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

using namespace acutesim;
using namespace acutesim::compute::orchestration;

int main(int argc, char** argv) {
    if (argc < 4) return 2;
    std::ifstream f(argv[1]);
    std::stringstream ss; ss << f.rdbuf();
    auto engine = ISimulationEngine::create();
    engine->setPrecisionMode(std::string(argv[2]) == "cpu" ? PrecisionMode::Native_F64
                                                         : PrecisionMode::Mixed);
    auto session = engine->createSession();
    SimulationRequestDTO req;
    req.source.payload = ss.str();
    const std::string path = argv[1];
    req.source.baseDir = path.find('/') == std::string::npos ? "" : path.substr(0, path.rfind('/'));
    // The harness runs a copy of the netlist; TREESPICE_BASEDIR points .include/.lib at the original directory.
    if (const char* d = std::getenv("TREESPICE_BASEDIR")) req.source.baseDir = d;
    EngineCallbacks cb;
    std::string mode = argv[3];
    SimulationResponseDTO r;
    if (mode == "tran") {
        req.transient.stepSizeS = std::atof(argv[4]);
        req.transient.stopTimeS = std::atof(argv[5]);
        r = session->runTransient(req, cb, nullptr);
    } else if (mode == "ac") {
        r = session->runAC(req, cb);
    } else if (mode == "sens") {
        r = session->runSens(req, cb);
    } else if (mode == "pz") {
        r = session->runPoleZero(req, cb);
    } else if (mode == "noise") {
        r = session->runNoise(req, cb);
    } else {
        r = session->runDC(req, cb);
    }
    std::printf("CONVERGED %d\nERROR %s\nBACKEND %s\n", r.converged ? 1 : 0, r.errorDetail.c_str(), r.backend.c_str());
    const auto& nz = r.extended.noise;
    for (size_t k = 0; k < nz.frequencies.size(); ++k)
        std::printf("%.9g %.9g %.9g\n", nz.frequencies[k], nz.onoise[k], nz.inoise[k]);
    if (r.extended.hasNoise) std::printf("TOTAL %.9g %.9g\n", nz.onoiseTotal, nz.inoiseTotal);
    const auto& pz = r.extended.pz;
    for (size_t k = 0; k < pz.poleRe.size(); ++k) std::printf("POLE %.9g %.9g\n", pz.poleRe[k], pz.poleIm[k]);
    for (size_t k = 0; k < pz.zeroRe.size(); ++k) std::printf("ZERO %.9g %.9g\n", pz.zeroRe[k], pz.zeroIm[k]);
    const auto& sn = r.extended.sens;
    for (size_t k = 0; k < sn.names.size(); ++k) std::printf("SENS %s %.9g\n", sn.names[k].c_str(), sn.values[k]);
    for (const auto& u : sn.unsupported) std::printf("UNSUPPORTED %s\n", u.c_str());
    std::printf("NODES");
    for (const auto& n : r.nodeNames) std::printf(" %s", n.empty() ? "-" : n.c_str());
    std::printf("\nBRANCHES");
    for (const auto& n : r.branchNames) std::printf(" %s", n.c_str());
    std::printf("\n");
    const std::vector<double>* shown = nullptr;
    for (const auto& p : r.waveformPoints) {
        if (!p.outer.empty() && (!shown || *shown != p.outer)) {
            std::printf("SWEEP");
            for (size_t i = 0; i < p.outer.size() && i < r.outerNames.size(); ++i)
                std::printf(" %s=%.9g", r.outerNames[i].c_str(), p.outer[i]);
            std::printf("\n");
            shown = &p.outer;
        }
        std::printf("%.9g", p.timeS);
        for (size_t i = 0; i < p.nodeVoltages.size(); ++i) {
            std::printf(" %.9g", p.nodeVoltages[i]);
            if (i < p.nodeVoltagesImag.size()) std::printf(" %.9g", p.nodeVoltagesImag[i]);
        }
        for (size_t i = 0; i < p.branchCurrents.size(); ++i) {
            std::printf(" %.9g", p.branchCurrents[i]);
            if (i < p.branchCurrentsImag.size()) std::printf(" %.9g", p.branchCurrentsImag[i]);
        }
        std::printf("\n");
    }
    return 0;
}
