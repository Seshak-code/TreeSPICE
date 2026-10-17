// treespice: ngspice-style batch simulator.
//   treespice [-b] [-o log] [-r out.raw] [--ascii] [--cpu] [analysis override] netlist.cir
//   treespice --help
// Runs the netlist's .op/.dc/.ac/.tran/.noise cards in file order, prints the .print tables
// (or the operating point), and writes an ngspice raw file with -r. Exit 0 when every analysis
// converged, 1 on a failure, 2 on a usage error.
#include "engine_api/isimulation_engine.h"
#include "engine_api/simulation_session.h"
#include "engine_api/simulation_dto.h"
#include "engine_api/engine_callbacks.h"
#include "engine_api/measure.h"
#include "engine_api/output.h"
#include "netlist/numparam.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace acutesim;
using namespace acutesim::compute::orchestration;

namespace {

const char* kUsage =
    "usage: treespice [options] [netlist]   (no netlist: read it from stdin)\n"
    "  -b               batch mode (the default; accepted for ngspice scripts)\n"
    "  -o FILE          write the log (everything printed to stdout) to FILE\n"
    "  -r FILE          write an ngspice raw file with every vector (or the .save/.probe ones)\n"
    "  --ascii          ascii raw file (default: binary)\n"
    "  --cpu            double-precision CPU solve (default: mixed precision, GPU when available)\n"
    "  --op | --tran STEP STOP | --ac dec|oct|lin N FSTART FSTOP | --dc SRC START STOP STEP\n"
    "                   run this analysis instead of the netlist's cards (numbers take SI suffixes: 10u, 1meg)\n"
    "  -h, --help       this text\n"
    "  -v, --version    print the version\n"
    "Output: .print tran|ac|dc|op expr... prints a table; expr is v(a), v(a,b)*2, i(v1), vdb(out),\n"
    "db(), mag(), ph(), real(), imag(). Without .print the operating point lists every node.\n";

struct Analysis {
    std::string kind;  // op, dc, ac, tran, noise, pz, sens
    SimulationRequestDTO req;
};

[[noreturn]] void usageError(const std::string& why) {
    std::cerr << "treespice: " << why << "\n" << kUsage;
    std::exit(2);
}

double number(const std::string& s) {
    try { return numparam::eval(s, [](const std::string&, double&) { return false; }); }
    catch (const std::exception&) { usageError("'" + s + "' is not a number"); }
}

// The analysis cards of the netlist (title line excluded), in file order.
std::vector<std::string> analysisCards(const std::string& text) {
    std::vector<std::string> kinds;
    std::istringstream in(text);
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string kw;
        ls >> kw;
        for (auto& c : kw) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (kw == ".op" || kw == ".dc" || kw == ".ac" || kw == ".tran" || kw == ".noise" || kw == ".pz" || kw == ".sens") kinds.push_back(kw.substr(1));
    }
    return kinds;
}

std::string title(const std::string& text) { return text.substr(0, text.find('\n')); }

void printOp(std::ostream& os, const SimulationResponseDTO& r) {
    char b[160];
    const auto& p = r.waveformPoints.front();
    os << "\tNode                                  Voltage\n\t----                                  -------\n";
    for (size_t i = 0; i < p.nodeVoltages.size(); ++i)
        if (!r.nodeNames[i].empty()) { std::snprintf(b, sizeof b, "\t%-30s%16.6e\n", r.nodeNames[i].c_str(), p.nodeVoltages[i]); os << b; }
    if (r.branchNames.empty()) return;
    os << "\n\tSource\tCurrent\n\t------\t-------\n\n";
    for (size_t i = 0; i < r.branchNames.size() && i < p.branchCurrents.size(); ++i) {
        std::snprintf(b, sizeof b, "\ti(%s)%*s%16.6e\n", r.branchNames[i].c_str(),
                      std::max(1, 26 - static_cast<int>(r.branchNames[i].size())), "", p.branchCurrents[i]);
        os << b;
    }
}

void printNoise(std::ostream& os, const SimulationResponseDTO& r) {
    char b[160];
    const auto& nz = r.extended.noise;
    os << "Index   frequency       onoise_spectrum inoise_spectrum\n";
    for (size_t k = 0; k < nz.frequencies.size(); ++k) {
        std::snprintf(b, sizeof b, "%zu\t%.6e\t%.6e\t%.6e\n", k, nz.frequencies[k], nz.onoise[k], nz.inoise[k]);
        os << b;
    }
    std::snprintf(b, sizeof b, "onoise_total = %.6e\ninoise_total = %.6e\n", nz.onoiseTotal, nz.inoiseTotal);
    os << b;
}

// ngspice's pz table: pole(k) = re,im then zero(k) = re,im
void printPZ(std::ostream& os, const SimulationResponseDTO& r) {
    char b[160];
    const auto& pz = r.extended.pz;
    for (size_t k = 0; k < pz.poleRe.size(); ++k) { std::snprintf(b, sizeof b, "pole(%zu) = %.5e,%.6e\n", k + 1, pz.poleRe[k], pz.poleIm[k]); os << b; }
    for (size_t k = 0; k < pz.zeroRe.size(); ++k) { std::snprintf(b, sizeof b, "zero(%zu) = %.5e,%.6e\n", k + 1, pz.zeroRe[k], pz.zeroIm[k]); os << b; }
}

// ngspice lists "name = value" per parameter; elements left out say so (a missing line is never a zero)
void printSens(std::ostream& os, const SimulationResponseDTO& r) {
    char b[160];
    const auto& sn = r.extended.sens;
    for (size_t k = 0; k < sn.names.size(); ++k) { std::snprintf(b, sizeof b, "%s = %.6e\n", sn.names[k].c_str(), sn.values[k]); os << b; }
    for (const auto& u : sn.unsupported) os << "sensitivity of " << u << " parameters is not supported yet\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string netlistPath, logPath, rawPath;
    bool ascii = false, cpu = false;
    std::vector<Analysis> overrides;
    auto engine = ISimulationEngine::create();
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](int n) {  // the n arguments after option `a`
            if (i + n >= argc) usageError(a + " needs " + std::to_string(n) + " argument(s)");
            std::vector<std::string> v(argv + i + 1, argv + i + 1 + n);
            i += n;
            return v;
        };
        if (a == "-h" || a == "--help") { std::cout << kUsage; return 0; }
        else if (a == "-v" || a == "--version") { std::cout << engine->capabilities().engineVersionString << "\n"; return 0; }
        else if (a == "-b") {}
        else if (a == "-o") logPath = next(1)[0];
        else if (a == "-r") rawPath = next(1)[0];
        else if (a == "--ascii") ascii = true;
        else if (a == "--cpu") cpu = true;
        else if (a == "--op") overrides.push_back({"op", {}});
        else if (a == "--tran") {
            const auto v = next(2);
            Analysis an{"tran", {}};
            an.req.transient.stepSizeS = number(v[0]);
            an.req.transient.stopTimeS = number(v[1]);
            if (!(an.req.transient.stepSizeS > 0 && an.req.transient.stopTimeS > 0)) usageError("--tran needs STEP > 0 and STOP > 0");
            overrides.push_back(std::move(an));
        } else if (a == "--ac") {
            const auto v = next(4);
            Analysis an{"ac", {}};
            an.req.ac = {v[0], static_cast<int>(number(v[1])), number(v[2]), number(v[3])};
            if (an.req.ac.points <= 0) usageError("--ac needs N >= 1");
            overrides.push_back(std::move(an));
        } else if (a == "--dc") {
            const auto v = next(4);
            Analysis an{"dc", {}};
            an.req.dc = {v[0], number(v[1]), number(v[2]), number(v[3])};
            overrides.push_back(std::move(an));
        } else if (a.size() > 1 && a[0] == '-') usageError("unknown option " + a);
        else if (netlistPath.empty()) netlistPath = a;
        else usageError("more than one netlist: " + netlistPath + ", " + a);
    }

    std::ostringstream src;
    if (netlistPath.empty()) src << std::cin.rdbuf();
    else {
        std::ifstream f(netlistPath);
        if (!f) { std::cerr << "treespice: cannot open " << netlistPath << "\n"; return 1; }
        src << f.rdbuf();
    }
    const std::string text = src.str();
    std::ofstream logFile;
    if (!logPath.empty()) {
        logFile.open(logPath);
        if (!logFile) { std::cerr << "treespice: cannot write " << logPath << "\n"; return 1; }
    }
    std::ostream& out = logPath.empty() ? std::cout : logFile;
    std::ofstream rawFile;
    if (!rawPath.empty()) {
        rawFile.open(rawPath, std::ios::binary);
        if (!rawFile) { std::cerr << "treespice: cannot write " << rawPath << "\n"; return 1; }
    }

    std::vector<Analysis> todo = overrides;
    if (todo.empty()) {
        // ponytail: an .op beside a .dc is dropped (the session sweeps whenever the netlist has a .dc card)
        const auto cards = analysisCards(text);
        for (const auto& k : cards)
            if (k != "op" || std::find(cards.begin(), cards.end(), "dc") == cards.end()) todo.push_back({k, {}});
        if (todo.empty()) todo.push_back({"op", {}});  // ngspice would do nothing; an .op is the useful default
    }
    engine->setPrecisionMode(cpu ? PrecisionMode::Native_F64 : PrecisionMode::Mixed);

    out << engine->capabilities().engineVersionString << "\nCircuit: " << title(text) << "\n\n";
    int status = 0;
    for (auto& an : todo) {
        an.req.source.payload = text;
        an.req.source.baseDir = netlistPath.find('/') == std::string::npos ? "" : netlistPath.substr(0, netlistPath.rfind('/'));
        if (an.kind == "tran" && overrides.empty()) an.req.transient.stepSizeS = an.req.transient.stopTimeS = 0;  // the .tran card
        auto session = engine->createSession();
        EngineCallbacks cb;
        const auto t0 = std::chrono::steady_clock::now();
        SimulationResponseDTO r = an.kind == "tran" ? session->runTransient(an.req, cb, nullptr)
                                : an.kind == "ac"   ? session->runAC(an.req, cb)
                                : an.kind == "noise" ? session->runNoise(an.req, cb)
                                : an.kind == "sens" ? session->runSens(an.req, cb)
                                : an.kind == "pz"   ? session->runPoleZero(an.req, cb)
                                                    : session->runDC(an.req, cb);
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (!r.converged) {
            std::cerr << "treespice: ." << an.kind << " failed: " << r.errorDetail << "\n";
            status = 1;
            continue;
        }
        try {
            if (an.kind == "sens") {
                if (!rawPath.empty()) throw std::runtime_error("raw output of .sens is not supported yet");
                printSens(out, r);
            } else if (an.kind == "pz") {
                if (!rawPath.empty()) throw std::runtime_error("raw output of .pz is not supported yet");
                printPZ(out, r);
            } else if (an.kind == "noise") {
                if (!rawPath.empty()) throw std::runtime_error("raw output of .noise is not supported yet");
                printNoise(out, r);
            } else {
                const auto prints = requestedSpecs(r, 'p');
                const auto saves = requestedSpecs(r, 's');
                if (!rawPath.empty() && !r.stepped) writeRaw(rawFile, selectOutputs(r, saves), title(text), ascii);  // ngspice: one plot, the inner sweep repeating
                int step = 0;
                for (const auto& part : splitByOuter(r)) {  // a nested .dc / .temp / .step list: one table per outer value
                    if (!rawPath.empty() && r.stepped) writeRaw(rawFile, selectOutputs(part, saves), title(text), ascii);  // one plot per step
                    if (!part.waveformPoints.empty() && !part.waveformPoints[0].outer.empty()) {
                        out << "Sweep:";
                        for (size_t i = 0; i < r.outerNames.size(); ++i) out << " " << r.outerNames[i] << "=" << part.waveformPoints[0].outer[i];
                        out << "\n";
                    }
                    if (!prints.empty()) writeTable(out, selectOutputs(part, prints), title(text));
                    else if (part.scaleName.empty()) printOp(out, part);
                    else if (rawPath.empty() && r.measures.empty()) out << "." << an.kind << ": " << part.waveformPoints.size() << " points; add a .print card or -r FILE to see them\n";
                    writeMeasures(out, r, step++);
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "treespice: ." << an.kind << " output: " << e.what() << "\n";
            status = 1;
        }
        out << "." << an.kind << " analysis time (seconds) = " << secs << (r.backend.empty() ? "" : ", backend: " + r.backend) << "\n\n";
    }
    return status;
}
