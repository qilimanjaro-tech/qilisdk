cmake_minimum_required(VERSION 3.18)
project(qilisdk LANGUAGES CXX)

# pybind should already be installed via the dependencies
find_package(pybind11 REQUIRED)

# This is optional, but should be available as part of most compilers
find_package(OpenMP)

# Fetch Eigen if not found (it's a header only library)
find_package(Eigen3 REQUIRED QUIET)
if (NOT EIGEN3_FOUND)
    message(STATUS "Eigen3 not found, fetching...")
    include(FetchContent)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(eigen)
    find_package(Eigen3 REQUIRED)
endif()

# If we have the core dependencies, build the C++ backend
if (EIGEN3_FOUND AND pybind11_FOUND)
    message(STATUS "Building QiliSim C++ backend...")
    pybind11_add_module(qilisim_module src/qilisdk/backends/qilisim.cpp)

    # Set compile options
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(STATUS "Using GCC/Clang specific compile options")
        # target_compile_options(qilisim_module PRIVATE -shared -fPIC)
        # target_compile_options(qilisim_module PRIVATE -O3)
        target_compile_options(qilisim_module PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_features(qilisim_module PRIVATE cxx_std_17)
    endif()

    # Link libraries and include directories
    # target_link_libraries(qilisim_module PUBLIC Eigen3::Eigen)
    target_include_directories(qilisim_module PRIVATE ${EIGEN3_INCLUDE_DIRS})
    target_include_directories(qilisim_module PRIVATE ${pybind11_INCLUDE_DIRS})
    if (OpenMP_CXX_FOUND)
        message(STATUS "Found OpenMP, enabling")
        target_link_libraries(qilisim_module PRIVATE OpenMP::OpenMP_CXX)
    endif()

    # Compile
    install(TARGETS qilisim_module DESTINATION .)

endif()
