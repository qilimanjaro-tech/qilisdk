# Standalone build for the QiliSim GPU microbenchmarks.
#
# This is intentionally decoupled from the main scikit-build/pybind build: the
# benchmark compiles src/qilisdk_cpp/libs/gpu.cpp directly (it has no pybind
# dependency) and only needs Eigen + libdl. Build & run via ./run_bench.sh, or
# manually:
#   cmake -S benchmarks -B benchmarks/build -DCMAKE_BUILD_TYPE=RelWithDebInfo
#   cmake --build benchmarks/build -j
#   ./benchmarks/build/bench_sr_solve

cmake_minimum_required(VERSION 3.18)
project(qilisdk_gpu_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

# Eigen: reuse a system/prebuilt copy if present, otherwise fetch the same tag
# the root project pins (see ../CMakeLists.txt).
find_package(Eigen3 5 QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "[bench] Eigen3 not found, fetching 5.0.1...")
    set(BUILD_TESTING OFF)
    set(EIGEN_BUILD_TESTING OFF)
    set(EIGEN_BUILD_PKGCONFIG OFF)
    set(EIGEN_BUILD_DOC OFF)
    include(FetchContent)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 5.0.1
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(eigen)
endif()

set(GPU_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../src/qilisdk_cpp/libs/gpu.cpp)

add_executable(bench_sr_solve bench_sr_solve.cpp ${GPU_SRC})
target_include_directories(bench_sr_solve PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/qilisdk_cpp/libs)
target_link_libraries(bench_sr_solve PRIVATE Eigen3::Eigen ${CMAKE_DL_LIBS})

# Optimize both the CPU reference and the harness, keep symbols for nsys, and
# match the SDK's default arch target so the CPU baseline is representative.
target_compile_options(bench_sr_solve PRIVATE -O3 -g -fcx-limited-range)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=x86-64-v3" BENCH_HAS_X86_64_V3)
if(BENCH_HAS_X86_64_V3)
    target_compile_options(bench_sr_solve PRIVATE -march=x86-64-v3)
endif()
