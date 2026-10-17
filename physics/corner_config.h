#pragma once

#include <vector>
#include <string>
#include <functional> // for std::hash

#if defined(QT_CORE_LIB) || defined(USE_QT)
#include <QString>
#include <QMetaType>
#include <QList>
#include <QVector>
#endif

namespace acutesim::physics {

enum class ProcessCorner {
    TT, // Typical-Typical
    FF, // Fast-Fast
    SS, // Slow-Slow
    FS, // Fast-Slow
    SF  // Slow-Fast
};

enum class VariationMode {
    NONE,
    WORST_CASE, // Linear worst-case combination
    MONTE_CARLO // Statistical sampling
};

enum class RunMode {
    FULL,       // Run full simulation
    QUICK,      // 1-iteration preview
    RESUME,     // Resume from last point
    INCREMENTAL, // Only solve affected blocks
    DRY_RUN,    // Just validate, don't solve
    BATCH       // Part of a larger batch
};

/**
 * @brief Configuration for a single Simulation Corner (PVT + Statistical)
 */
struct CornerConfig {
    std::string name; // e.g., "Worst-Case Hot"
    
    // Environmental
    double temperatureC = 27.0;
    double voltageScale = 1.0; // 1.0 = Nominal VDD
    
    // Process
    ProcessCorner process = ProcessCorner::TT;
    
    // Statistical
    VariationMode mode = VariationMode::NONE;
    int iterationIndex = 0; // 0 to N-1 for MC runs
    
    // Helper to generate a unique hash for caching
    size_t hash() const {
        return std::hash<double>{}(temperatureC) ^ 
               std::hash<double>{}(voltageScale) ^
               static_cast<size_t>(process);
    }
    
    // Factory methods
    static CornerConfig Nominal() { 
        return { "Nominal", 27.0, 1.0, ProcessCorner::TT }; 
    }
    
    static CornerConfig WorstCaseHot() {
        return { "Slow-Hot", 125.0, 0.9, ProcessCorner::SS, VariationMode::WORST_CASE };
    }
    
    static CornerConfig WorstCaseCold() {
        return { "Fast-Cold", -40.0, 1.1, ProcessCorner::FF, VariationMode::WORST_CASE };
    }
    
    // Helper for UI population
    static std::vector<CornerConfig> standard5() {
        return {
            { "TT (Typical)", 27.0, 1.0, ProcessCorner::TT },
            { "FF (Fast-Fast)", 27.0, 1.0, ProcessCorner::FF },
            { "SS (Slow-Slow)", 27.0, 1.0, ProcessCorner::SS },
            { "FS (Fast-Slow)", 27.0, 1.0, ProcessCorner::FS },
            { "SF (Slow-Fast)", 27.0, 1.0, ProcessCorner::SF }
        };
    }

#if defined(QT_CORE_LIB) || defined(USE_QT)
    // Conversion helpers if needed
    QString qName() const { return QString::fromStdString(name); }
#endif
};

} // namespace acutesim::physics

#if defined(QT_CORE_LIB) || defined(USE_QT)
Q_DECLARE_METATYPE(acutesim::physics::CornerConfig)
#endif

