cmake_minimum_required(VERSION 3.20)

# ===========================================================================
# TreeSPICE (acutesim_engine) — SPICE-equivalent circuit simulation engine
# ===========================================================================
# Builds standalone (this directory is the top level: library, tools, tests)
# or as a subdirectory of OpenVLSI (git submodule at acutesim_engine/),
# which consumes it through engine_api/ and the netlist data model.
#
# Build pattern: OBJECT → STATIC (future: SHARED/PLUGIN without recompilation)
# ===========================================================================

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(TreeSPICE LANGUAGES C CXX)
    set(TREESPICE_STANDALONE ON)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release)
    endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The one version string: treespice --version and EngineCapabilities::engineVersionString
set(TREESPICE_VERSION "1.0-alpha")

# ---------------------------------------------------------------------------
# GPU (Phase 4): Dawn / WebGPU. OpenVLSI provides dawn_native when it builds the
# engine as a subdirectory; standalone, -DTREESPICE_GPU=ON fetches the same Dawn
# revision (point FETCHCONTENT_SOURCE_DIR_DAWN at an existing checkout to reuse it).
# ---------------------------------------------------------------------------
option(TREESPICE_GPU "Standalone build: fetch and build Dawn for the GPU path" OFF)
if(TREESPICE_STANDALONE AND TREESPICE_GPU AND NOT TARGET dawn_native)
    include(FetchContent)
    FetchContent_Declare(dawn GIT_REPOSITORY https://dawn.googlesource.com/dawn
                              GIT_TAG chromium/6943 GIT_SHALLOW ON GIT_SUBMODULES "")
    foreach(opt DAWN_WERROR TINT_WERROR DAWN_BUILD_SAMPLES DAWN_ENABLE_VULKAN DAWN_ENABLE_D3D12
                DAWN_ENABLE_D3D11 DAWN_ENABLE_OPENGLES DAWN_ENABLE_DESKTOP_GL TINT_BUILD_TESTS DAWN_BUILD_TESTS)
        set(${opt} OFF CACHE BOOL "" FORCE)
    endforeach()
    set(DAWN_ENABLE_METAL ${APPLE} CACHE BOOL "" FORCE)
    set(DAWN_FETCH_DEPENDENCIES ON CACHE BOOL "" FORCE)
    set(CMAKE_CXX_FLAGS_SAVED "${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")  # Dawn / Tint / Abseil warnings
    FetchContent_MakeAvailable(dawn)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_SAVED}")
    # Dawn's pinned Abseil lists "-Xarch_x86_64;-maes;-Xarch_x86_64;-msse4.1" on Apple without SHELL:,
    # and CMake de-duplicates the repeated "-Xarch_x86_64", leaving a bare -msse4.1 that arm64 rejects.
    foreach(t absl_random_internal_randen_hwaes absl_random_internal_randen_hwaes_impl)
        if(APPLE AND TARGET ${t})
            get_target_property(o ${t} COMPILE_OPTIONS)
            string(REGEX REPLACE "(-Xarch_[a-z0-9_]+);([^;]+)" "SHELL:\\1 \\2" o "${o}")
            set_property(TARGET ${t} PROPERTY COMPILE_OPTIONS "${o}")
        endif()
    endforeach()
    set(DAWN_INTERNAL_INCLUDES ${dawn_SOURCE_DIR} ${dawn_SOURCE_DIR}/src ${dawn_BINARY_DIR}/gen/src
        ${dawn_SOURCE_DIR}/third_party/abseil-cpp ${dawn_SOURCE_DIR}/third_party/partition_alloc)
endif()

# ---------------------------------------------------------------------------
# OBJECT target — compiled once, wrapped in STATIC (or SHARED later)
# ---------------------------------------------------------------------------
add_library(acutesim_engine_objects OBJECT)
target_compile_features(acutesim_engine_objects PUBLIC cxx_std_17)
target_compile_definitions(acutesim_engine_objects PRIVATE TREESPICE_VERSION="${TREESPICE_VERSION}")

# === HARDENING: Symbol Visibility Discipline ===
target_compile_options(acutesim_engine_objects PRIVATE
    -fvisibility=hidden
    -fvisibility-inlines-hidden
)

# No -ffast-math: it implies -ffinite-math-only, which deletes every
# isnan/isfinite guard in the solver (NaN results were reported as converged),
# and it breaks the project's cross-platform determinism rule (see CLAUDE.md).
# No FMA contraction either: a fused a*b+c rounds once instead of twice, so arm64 (where
# clang contracts within expressions and GCC across them) and x86 gave different bits.
# No -march=native: the binary must run, and give the same results, on any machine.
# The engine and every later target (tests and tools compile header-only engine parts)
if(MSVC)
    set(TREESPICE_FP_FLAGS /fp:strict)
else()
    set(TREESPICE_FP_FLAGS -ffp-contract=off)
    target_compile_options(acutesim_engine_objects PRIVATE -O3)
endif()
target_compile_options(acutesim_engine_objects PRIVATE ${TREESPICE_FP_FLAGS})
add_compile_options(${TREESPICE_FP_FLAGS})

target_compile_definitions(acutesim_engine_objects PRIVATE
    ACUTESIM_ENGINE_INTERNAL=1
)

# Tree levels factor on std::thread workers (math/contraction_solver.cpp). The
# app defined this globally, so the standalone build silently ran serial.
if(NOT EMSCRIPTEN)
    find_package(Threads REQUIRED)
    target_compile_definitions(acutesim_engine_objects PRIVATE ACUTESIM_THREADS_ENABLED)
    target_link_libraries(acutesim_engine_objects PUBLIC Threads::Threads)
endif()

# ---------------------------------------------------------------------------
# Include directories
# ---------------------------------------------------------------------------
target_include_directories(acutesim_engine_objects
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>              # "engine_api/x.h", "netlist/circuit.h"
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/engine_api>   # bare "simulation_dto.h"
        $<INSTALL_INTERFACE:include/treespice>
        $<INSTALL_INTERFACE:include/treespice/engine_api>
)

# ---------------------------------------------------------------------------
# GPU: Dawn/WebGPU
# ---------------------------------------------------------------------------
if(NOT EMSCRIPTEN AND TARGET dawn_native)
    target_include_directories(acutesim_engine_objects PRIVATE
        ${DAWN_INTERNAL_INCLUDES}
    )
    target_link_libraries(acutesim_engine_objects PRIVATE
        dawn_native
        dawn_proc
        webgpu_cpp
    )
    target_compile_definitions(acutesim_engine_objects PRIVATE
        ACUTESIM_GPU_ENABLED=1
        ACUTESIM_HAS_DAWN_NATIVE=1
        ACUTESIM_GPU_PROBE_AVAILABLE=1
    )
endif()

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------

target_sources(acutesim_engine_objects PRIVATE
    # === Façade (engine-api boundary) ===
    ${CMAKE_CURRENT_SOURCE_DIR}/engine_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_context_manager.cpp

    # === Infrastructure ===
    ${CMAKE_CURRENT_SOURCE_DIR}/infrastructure/postproc_engine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/infrastructure/hardware_capabilities.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/infrastructure/output.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/infrastructure/measure.cpp

    # === Math ===
    ${CMAKE_CURRENT_SOURCE_DIR}/math/contraction_solver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/math/gpu_front_lu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/math/pole_zero.cpp

    # === Solvers ===
    ${CMAKE_CURRENT_SOURCE_DIR}/solvers/circuitsim_solver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/solvers/fourier_analysis.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/solvers/tf_analysis.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/solvers/register_analyses.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/solvers/analysis_registry.cpp

    # === Tensors (SoA linear elements) ===
    ${CMAKE_CURRENT_SOURCE_DIR}/tensors/physics_tensors.cpp

    # === Netlist ===
    ${CMAKE_CURRENT_SOURCE_DIR}/netlist/spice_parser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/netlist/numparam.cpp

    # === OSDI devices (compiled Verilog-A: BSIM4, BSIM-CMG, ... from OpenVAF) ===
    ${CMAKE_CURRENT_SOURCE_DIR}/osdi/osdi_device.cpp
)
target_link_libraries(acutesim_engine_objects PUBLIC ${CMAKE_DL_LIBS})  # dlopen for .osdi libraries

# ---------------------------------------------------------------------------
# STATIC library wrapper
# ---------------------------------------------------------------------------
add_library(acutesim_engine STATIC
    $<TARGET_OBJECTS:acutesim_engine_objects>
)

target_link_libraries(acutesim_engine PUBLIC acutesim_engine_objects)

if(NOT EMSCRIPTEN AND TARGET dawn_native)
    target_link_libraries(acutesim_engine PUBLIC
        dawn_native
        dawn_proc
        webgpu_cpp
    )
endif()

# ---------------------------------------------------------------------------
# CLI — Engine Consumer
# ---------------------------------------------------------------------------
add_executable(treespice cli/engine_cli_main.cpp)
target_link_libraries(treespice PRIVATE acutesim_engine)

# cmake --install build --prefix X: bin/treespice, lib/libacutesim_engine.a, include/treespice/engine_api/*.h
# (the static library carries every object, so the OBJECT target needs no install)
if(TREESPICE_STANDALONE)
    include(GNUInstallDirs)
    install(TARGETS treespice RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(TARGETS acutesim_engine ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(DIRECTORY engine_api DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/treespice FILES_MATCHING PATTERN "*.h")
endif()

# ---------------------------------------------------------------------------
# Waveform dump tool — machine-readable output for tests/ngspice_diff
# Output: "CONVERGED <0|1>", "ERROR <text>", then "<time> <v1> <v2> ..." rows
# (node order = first appearance in the netlist, subcircuit bodies excluded)
# ---------------------------------------------------------------------------
add_executable(acutesim_engine_dump cli/engine_dump_main.cpp)
target_link_libraries(acutesim_engine_dump PRIVATE acutesim_engine)

# ---------------------------------------------------------------------------
# Engine tests (standalone build): C++ checks + differential harnesses
#   ctest                                  — the C++ suites below
#   python3 tests/ngspice_diff/run_diff.py — vs ngspice (needs ngspice on PATH)
#   python3 tests/xyce_diff.py, tests/ibmpg_diff.py — need datasets/ (git-ignored)
# ---------------------------------------------------------------------------
if(TREESPICE_STANDALONE)
    enable_testing()
    foreach(t test_engine_fixes test_transient_session test_contraction_solver test_convergence_engine)
        add_executable(${t} tests/${t}.cpp)
        target_link_libraries(${t} PRIVATE acutesim_engine)
        add_test(NAME ${t} COMMAND ${t})
    endforeach()

    # Solver benchmark vs SuiteSparse KLU (bench/klu_bench.py); built only if KLU is installed
    # (plain find_*: Homebrew's SuiteSparse CMake package demands OpenMP, which Apple clang lacks)
    find_path(KLU_INCLUDE_DIR klu.h PATH_SUFFIXES suitesparse)
    find_library(KLU_LIBRARY klu)
    if(KLU_INCLUDE_DIR AND KLU_LIBRARY)
        add_executable(klu_bench bench/klu_bench.cpp)
        target_include_directories(klu_bench PRIVATE ${KLU_INCLUDE_DIR})
        target_link_libraries(klu_bench PRIVATE acutesim_engine ${KLU_LIBRARY})
    endif()
endif()
