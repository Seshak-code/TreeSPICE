#pragma once
// ============================================================================
// simulation_dto.h — Pure Data Transfer Objects for the Orchestration Layer
// ============================================================================
// ARCHITECTURAL RULE: This header must NEVER include Qt, Eigen, Physics, or
// any Solver types. Only standard library types are allowed.
// If this header compiles without QApplication, you are compliant.
// ============================================================================

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <functional>
#include <cmath>

struct TensorNetlist;
#include "job_status.h"

namespace acutesim {

// ── Precision control (Moved to DTO layer for telemetry) ────────────────────
enum class PrecisionMode : uint8_t {
    Native_F64,     // IEEE 754 double — CPU scalar paths
    Emulated_F64,   // Hi-lo pair splitting — GPU determinism paths
    Mixed           // F64 for NR outer loop, EmF64 for GPU kernels
};

namespace compute {
namespace orchestration {

// ── Analysis Types (mirrored as strings for serialization safety) ────────────
enum class AnalysisTypeDTO : int {
    DC        = 0,
    TRANSIENT = 1,
    AC        = 2,
    NOISE     = 3,
    MONTE_CARLO = 4,
    SENSITIVITY = 5,
    TRANSFER_FUNCTION = 6,
    POLE_ZERO         = 7,
    FOURIER           = 8
};

inline std::string analysisTypeToString(AnalysisTypeDTO t) {
    switch (t) {
        case AnalysisTypeDTO::DC:                return "DC";
        case AnalysisTypeDTO::TRANSIENT:         return "TRANSIENT";
        case AnalysisTypeDTO::AC:                return "AC";
        case AnalysisTypeDTO::NOISE:             return "NOISE";
        case AnalysisTypeDTO::MONTE_CARLO:       return "MONTE_CARLO";
        case AnalysisTypeDTO::SENSITIVITY:       return "SENSITIVITY";
        case AnalysisTypeDTO::TRANSFER_FUNCTION: return "TRANSFER_FUNCTION";
        case AnalysisTypeDTO::POLE_ZERO:         return "POLE_ZERO";
        case AnalysisTypeDTO::FOURIER:           return "FOURIER";
    }
    return "UNKNOWN";
}

// ── Run Modes ──────────────────────────────────────────────────────────────
enum class RunModeDTO : int {
    FULL = 0,
    QUICK = 1,
    RESUME = 2,
    INCREMENTAL = 3
};

// ── Corner Parameters (POD) ─────────────────────────────────────────────────
struct CornerParametersDTO {
    std::string name        = "TT";
    double temperature      = 27.0;
    double supplyVoltage    = 1.8;
    double voltageScale     = 1.0;
    std::string processCorner = "TT";

    static std::vector<CornerParametersDTO> standard5() {
        return {
            {"TT", 27.0, 1.8, 1.0, "TT"},
            {"FF", -40.0, 1.8, 1.1, "FF"},
            {"SS", 125.0, 1.8, 0.9, "SS"},
            {"SF", 27.0, 1.8, 1.0, "SF"},
            {"FS", 27.0, 1.8, 1.0, "FS"}
        };
    }
};

struct ParameterVariationDTO {
    std::string componentName;
    std::string propertyName;
    double      overrideValue;
};

enum class NetlistFormatDTO : int {
    JSON     = 0,
    VERILOG_A = 1
};

struct SimulationSourceDTO {
    NetlistFormatDTO format  = NetlistFormatDTO::JSON;
    std::string      payload;
    std::string      baseDir;   // .include / .lib paths resolve against it (empty: working directory)
};

struct TransientConfigDTO {
    double stopTimeS = 1e-6;   // 0 (either): the netlist's .tran
    double stepSizeS = 1e-9;
    double maxStepS  = 1e-8;
    bool   uic       = false;   // start from .ic / IC= values, skip the DC bias
};

struct ACConfigDTO {
    std::string sweep = "dec";   // dec | oct | lin
    int    points = 0;           // per decade/octave, or total (lin); 0 = use the netlist's .ac
    double fStart = 0.0;
    double fStop  = 0.0;
};

// .dc sweepVariable: a V/I source, a resistor, or TEMP (Celsius)
struct DCConfigDTO {
    std::string sweepVariable;
    double start = 0.0;
    double stop  = 1.8;
    double step  = 0.1;
    std::vector<DCConfigDTO> outer;  // slower sweeps, next-slowest first (.dc V1 .. V2 ..: V2 is outer)
};

struct MonteCarloVariationDTO {
    std::string componentName;
    std::string propertyName;
    double      sigma            = 0.05;
    std::string distributionType = "Gaussian";
    double      nominalValue     = 0.0;
    double      tolerance        = 0.05;
};

struct MonteCarloConfigDTO {
    uint32_t sampleCount = 10;   // number of Monte Carlo runs
    uint32_t seed        = 0;
    std::vector<MonteCarloVariationDTO> variations;
};

// One value of a .step card, applied on top of the netlist: kind 'P' a .param, 'T' the temperature (Celsius),
// 'E' the value of a V/I source or resistor.
struct StepValueDTO {
    char        kind;
    std::string name;   // upper case
    double      value;
};

struct SimulationRequestDTO {
    JobID    jobId;
    SimulationSourceDTO source;
    AnalysisTypeDTO analysis = AnalysisTypeDTO::DC;
    std::unordered_map<std::string, std::string> parameters;
    CornerParametersDTO corner;
    std::string solverMode; 

    TransientConfigDTO transient;
    DCConfigDTO        dc;
    ACConfigDTO        ac;
    MonteCarloConfigDTO mc;
    
    uint32_t mcSeed = 0; 
    std::vector<StepValueDTO> stepValues;  // set by the session for one run of a .step list; empty: the netlist's own
    uint64_t topologyHash = 0;
    std::string sessionToken;
};

// ── Extended Results ────────────────────────────────────────────────────────

struct TFResultDTO {
    std::vector<double> frequencies;
    std::vector<double> gainDb;
    std::vector<double> phaseDegrees;
    double gainMarginDb      = 0.0;
    double phaseMarginDeg    = 0.0;
    double gainCrossoverHz   = 0.0;
    double phaseCrossoverHz  = 0.0;
};

struct FourierResultDTO {
    double fundamentalFrequencyHz = 0.0;
    std::vector<double> harmonicAmplitudes;
    std::vector<double> harmonicMagDb;
    std::vector<double> harmonicPhaseDeg;
    double thdPercent             = 0.0;
};

struct NoiseResultDTO {
    std::vector<double> frequencies;
    std::vector<double> onoise;        // output noise density [V/sqrt(Hz)]
    std::vector<double> inoise;        // input-referred density [V/sqrt(Hz)] (or A/sqrt(Hz))
    double onoiseTotal = 0.0;          // integrated over the sweep [V rms]
    double inoiseTotal = 0.0;
};

struct PoleZeroResultDTO {
    std::vector<double> poleRe, poleIm;  // every pole (conjugate pairs listed both), ngspice's pole(k) = re,im
    std::vector<double> zeroRe, zeroIm;
};

struct SensResultDTO {
    std::vector<std::string> names;     // ngspice's vector names, lower case: "r1", "r1_m", "r1_scale", "v1", "i1", "i1_m"
    std::vector<double> values;         // d(output)/d(parameter), same order
    std::vector<std::string> unsupported;  // device groups whose parameters were not computed, e.g. "2 BJT(s)" (never reported as 0)
};

struct ExtendedAnalysisResultsDTO {
    bool hasTF = false;
    TFResultDTO tf;
    bool hasNoise = false;
    NoiseResultDTO noise;
    bool hasSens = false;
    SensResultDTO sens;
    bool hasPZ = false;
    PoleZeroResultDTO pz;
    bool hasFourier = false;
    FourierResultDTO fourier;
};

struct WaveformDataPoint {
    double              timeS;            // time (tran), swept value (.dc, the innermost sweep), frequency (.ac)
    std::vector<double> nodeVoltages;     // real part for .ac
    std::vector<double> nodeVoltagesImag; // .ac only
    std::vector<double> branchCurrents;     // SimulationResponseDTO::branchNames order; real part for .ac
    std::vector<double> branchCurrentsImag; // .ac only
    std::vector<double> outer;            // values of the slower sweeps (SimulationResponseDTO::outerNames)
};

struct ConvergenceTelemetryDTO {
    int iterations = 0;
    double maxResidual = 0.0;
    double rmsResidual = 0.0;
    std::string method;
    bool gminSteppingUsed = false;
    bool sourceSteppingUsed = false;
    double solveTimeMs = 0.0;
    bool rankDeficient = false;
    int rankDeficientNode = -1;
    double smallestPivot = 0.0;
    bool stagnationDetected = false;
    bool divergenceDetected = false;
    int worstNodeIndex = -1;
    double worstNodeVoltage = 0.0;
    double gminApplied = 0.0;
    int gminInjections = 0;
    bool converged = false;
};

// A netlist .print (kind 'p') or .save/.probe (kind 's') card; analysis "" applies to every analysis.
struct OutputRequestDTO {
    char kind = 'p';
    std::string analysis;
    std::vector<std::string> specs;  // as written: "v(a,b)*2", "vdb(out)", "i(v1)", "all"
};

// One .meas result. step indexes the parts of a nested .dc / .temp / .step run (0 when there is one).
struct MeasureResultDTO {
    std::string name;      // lower case
    double      value = 0.0;
    bool        ok = false;
    std::string extra;     // ngspice's trailing " targ=... trig=..." / " from=... to=..." / " at=..."
    std::string error;     // ngspice's failure text when !ok
    int         step = 0;
};

struct SimulationResponseDTO {
    JobID       jobId;
    int32_t     runIndex = -1;  
    bool        success = false;   // fail-closed: a response must be marked good explicitly
    bool        converged = false;
    int         iterations = 0;
    double      residual = 0.0;
    uint64_t    topologyHash = 0;
    std::string errorMessage; // Proto compatibility
    std::string errorDetail;  // Backend consistency
    std::string solverProvenance;
    std::string backend;  // what actually ran: "cpu", or "gpu (...)"; says why when gpu was asked for and not used

    double      solveTimeMs = 0.0;
    AnalysisTypeDTO analysisType = AnalysisTypeDTO::DC;
    PrecisionMode   precisionMode = PrecisionMode::Mixed;

    std::vector<WaveformDataPoint> waveformPoints;
    // Column names, lower case as ngspice prints them: nodeNames[i] is nodeVoltages[i] ("" if unnamed);
    // branchNames[i] is branchCurrents[i] (the current into the + terminal of v1, l1, e1, h1, b1).
    std::vector<std::string> nodeNames, branchNames;
    std::string scaleName;  // what WaveformDataPoint::timeS is: "time", "frequency", "v-sweep", "i-sweep", "" (.op)
    std::vector<OutputRequestDTO> outputRequests;
    std::vector<std::string> outerNames;  // what WaveformDataPoint::outer holds: outer .dc sources, TEMP
    bool stepped = false;  // the last outerNames are .step values (one plot per step in a raw file)
    std::vector<MeasureResultDTO> measures;
    std::vector<double> timeVector;
    std::unordered_map<int, std::vector<double>> nodeVoltages;
    ExtendedAnalysisResultsDTO extended;

    // Diagnostics and operating point
    std::vector<double> dcOperatingPoint;
    std::unordered_map<std::string, std::vector<double>> probeValues;
    ConvergenceTelemetryDTO convergence;
};

struct BatchRunDTO {
    uint32_t runIndex = 0;
    std::vector<ParameterVariationDTO> overrides;
};

struct StatisticalSummaryDTO {
    std::unordered_map<std::string, double> mean;
    std::unordered_map<std::string, double> stddev;
    std::unordered_map<std::string, double> minVal;
    std::unordered_map<std::string, double> maxVal;
};

struct SimulationBatchRequestDTO {
    SimulationSourceDTO source;
    AnalysisTypeDTO analysis;
    std::unordered_map<std::string, std::string> parameters;
    CornerParametersDTO corner;
    std::vector<BatchRunDTO> runs;
    std::string sessionToken;
};

struct SimulationBatchResponseDTO {
    std::vector<SimulationResponseDTO> runResults;
    bool success = true;
    bool hadFailures = false;
    std::string sessionToken;
    std::string errorMessage; // Proto compatibility
    std::string errorDetail;  // Backend consistency
    StatisticalSummaryDTO statistics;
};

} // namespace orchestration
} // namespace compute
} // namespace acutesim
