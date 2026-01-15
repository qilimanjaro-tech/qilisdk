cmake_minimum_required(VERSION 3.18)
project(qilisdk LANGUAGES CXX)

# The CMake option to enable QiliSim backend
# To enable/disable QiliSim backend or to choose build type, use:
#   uv pip install -v -e ./ -Ccmake.define.qilisim=ON -Ccmake.build-type=Release
#   uv pip install -v -e ./ -Ccmake.define.qilisim=ON -Ccmake.build-type=Debug
if (DEFINED qilisim AND qilisim STREQUAL "ON")
    message(STATUS "[QiliSim] QiliSim backend enabled, preparing to build")
    message(STATUS "[QiliSim] Build type: ${CMAKE_BUILD_TYPE}")
    message(STATUS "[QiliSim] Verbose: ${CMAKE_VERBOSE_MAKEFILE}")

    # pybind should already be installed via the dependencies
    find_package(pybind11 REQUIRED)

    # This is optional, but should be available as part of most compilers
    find_package(OpenMP)

    # Fetch Eigen if not found (it's a header only library)
    find_package(Eigen3 QUIET)
    if (NOT Eigen3_FOUND)
        message(STATUS "[QiliSim] Eigen3 not found, fetching...")
        set(BUILD_TESTING OFF)
        set(EIGEN_BUILD_TESTING OFF)
        set(EIGEN_BUILD_PKGCONFIG OFF)
        set(EIGEN_BUILD_DOC OFF)
        include(FetchContent)
        FetchContent_Declare(
            eigen
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG 3.4.0
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(eigen)
    endif()

    # Assuming we have the core dependencies, prep the C++ backend
    message(STATUS "[QiliSim] All dependencies found, preparing QiliSim build files")
    pybind11_add_module(qilisim_module src/qilisdk_cpp/backends/qilisim/qilisim.cpp)

    # Set compile options
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(STATUS "[QiliSim] Using GCC/Clang specific compile options")
        target_compile_options(qilisim_module PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_features(qilisim_module PRIVATE cxx_std_17)
    endif()

    # Link libraries and include directories
    target_link_libraries(qilisim_module PUBLIC Eigen3::Eigen)
    target_include_directories(qilisim_module PRIVATE ${pybind11_INCLUDE_DIRS})
    if (OpenMP_CXX_FOUND)
        message(STATUS "[QiliSim] Using OpenMP")
        target_link_libraries(qilisim_module PRIVATE OpenMP::OpenMP_CXX)
    endif()

    # Enable if you want to run clang-tidy on the code
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(STATUS "[QiliSim] Enabling clang-tidy")
        set_target_properties(qilisim_module PROPERTIES
            CXX_CLANG_TIDY
            "clang-tidy;-header-filter=.;--extra-arg=-Wno-ignored-optimization-argument"
        )
    endif()

    # Compile
    install(TARGETS qilisim_module DESTINATION .)
        
else()
    message(STATUS "[QiliSim] QiliSim backend not enabled, skipping build")

endif()
