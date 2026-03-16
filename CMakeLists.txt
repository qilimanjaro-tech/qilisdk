# To enable debug build:
#   uv pip install -v -e ./ -Ccmake.build-type=Debug
# To run with clang-tidy:
#   uv pip install -v -e ./ -Ccmake.define.tidy=ON
# To enable (very) verbose simulator output:
#   uv pip install -v -e ./ -Ccmake.define.verbose=ON

cmake_minimum_required(VERSION 3.18)
project(qilisdk LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

# Check to make sure we have a C++ compiler
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("" COMPILER_EXISTS)
if(NOT COMPILER_EXISTS OR CMAKE_CXX_COMPILER STREQUAL "")
    message(FATAL_ERROR "[QiliSDK C++] No C++ compiler found. Please either install one (e.g. \"sudo apt install build-essential\" on Ubuntu) or recompile with -Ccmake.define.qilisdk=OFF to disable QiliSDK backend.")
endif()

# pybind should already be installed via the dependencies
find_package(pybind11 REQUIRED)

# This is optional, but should be available as part of most compilers
find_package(OpenMP)

# Fetch Eigen if not found (it's a header only library)
find_package(Eigen3 5 QUIET)
if (NOT Eigen3_FOUND)
    message(STATUS "[QiliSDK C++] Eigen3 not found, fetching...")
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

# Output the version of Eigen being used
message(STATUS "[QiliSDK C++] Using Eigen version: ${EIGEN3_VERSION}")

# Assuming we have the core dependencies, prep the C++ backend
message(STATUS "[QiliSDK C++] All dependencies found, preparing build files")

# QTensor
pybind11_add_module(qtensor_module src/qilisdk_cpp/core/qtensor_bindings.cpp)
file(GLOB QTENSOR_SOURCES src/qilisdk_cpp/core/*.cpp)
file(GLOB QTENSOR_SUBDIR_SOURCES src/qilisdk_cpp/libs/numpy.cpp)
list(APPEND QTENSOR_SOURCES ${QTENSOR_SUBDIR_SOURCES})
target_sources(qtensor_module PRIVATE ${QTENSOR_SOURCES})

# QiliSim
pybind11_add_module(qilisim_module src/qilisdk_cpp/backends/qilisim/qilisim_bindings.cpp)
file(GLOB QILISIM_SOURCES src/qilisdk_cpp/backends/qilisim/*.cpp)
file(GLOB QILISIM_SUBDIR_SOURCES src/qilisdk_cpp/backends/qilisim/*/*.cpp)
list(APPEND QILISIM_SOURCES ${QILISIM_SUBDIR_SOURCES})
file(GLOB QILISIM_LIBS_SOURCES src/qilisdk_cpp/libs/numpy.cpp)
list(APPEND QILISIM_SOURCES ${QILISIM_LIBS_SOURCES})
target_sources(qilisim_module PRIVATE ${QILISIM_SOURCES})

# List of all modules to build
set(MODULES_TO_BUILD qtensor_module qilisim_module)

# For each module
foreach(module ${MODULES_TO_BUILD})
    message(STATUS "[QiliSDK C++] Configuring module: ${module}")

    # Set compile options
    set (CMAKE_CXX_STANDARD 17)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(STATUS "[QiliSDK C++] Using GCC/Clang specific compile options")
        target_compile_options(${module} PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    # Link libraries and include directories
    target_link_libraries(${module} PUBLIC Eigen3::Eigen)
    target_include_directories(${module} PRIVATE ${pybind11_INCLUDE_DIRS})
    if (OpenMP_CXX_FOUND)
        message(STATUS "[QiliSDK C++] Using OpenMP")
        target_link_libraries(${module} PRIVATE OpenMP::OpenMP_CXX)
    endif()

    # Enable if you want to run clang-tidy on the code
    if (CMAKE_DEFINE_TIDY)
        message(STATUS "[QiliSDK C++] Enabling clang-tidy")
        set_target_properties(${module} PROPERTIES
            CXX_CLANG_TIDY
            "clang-tidy;-header-filter=.;--extra-arg=-Wno-ignored-optimization-argument"
        )
    endif()

    # If we should also build the test suite
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(STATUS "[QiliSim] Building test suite")
        include(FetchContent)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
        )
        FetchContent_MakeAvailable(googletest)
        enable_testing()
        add_executable(
            test_cpp
            tests/unit_cpp/test_utils.cpp
        )
        target_compile_options(test_cpp PRIVATE --coverage -fprofile-arcs -ftest-coverage)
        target_link_libraries(test_cpp PRIVATE --coverage GTest::gtest_main)
        include(GoogleTest)
        gtest_discover_tests(test_cpp)
        set_target_properties(test_cpp PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit_cpp)
    endif()

    # Compile
    install(TARGETS ${module} DESTINATION .)

endforeach()