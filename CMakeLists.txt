# To enable debug build:
#   uv pip install -v -e ./ -Ccmake.build-type=Debug
# To run with clang-tidy:
#   uv pip install -v -e ./ -Ccmake.define.tidy=ON
# To enable (very) verbose simulator output:
#   uv pip install -v -e ./ -Ccmake.define.verbose=ON
# To enable the test suite:
#   uv pip install -v -e ./ -Ccmake.define.tests=ON
# To enable coverage flags:
#   uv pip install -v -e ./ -Ccmake.define.coverage=ON
# For a fully editable install:
#   uv pip install --config-settings=editable.rebuild=true -Ccmake.build_type=Debug -Cbuild-dir=build -ve .

cmake_minimum_required(VERSION 3.18)
project(qilisdk LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

# Check to make sure we have a C++ compiler
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("" COMPILER_EXISTS)
if(NOT COMPILER_EXISTS OR CMAKE_CXX_COMPILER STREQUAL "")
    message(FATAL_ERROR "[QiliSDK C++] No C++ compiler found. Please install one (e.g. \"sudo apt install build-essential\" on Ubuntu or \"xcode-select --install\" on macOS) and try again.")
endif()

# pybind should already be installed via the dependencies
set(pybind11_DIR "${CMAKE_CURRENT_LIST_DIR}/.venv/share/cmake/pybind11")
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

# The common libs
pybind11_add_module(qilisdk_libs SHARED)
file(GLOB LIBS_SOURCES src/qilisdk_cpp/libs/*.cpp)
target_sources(qilisdk_libs PRIVATE ${LIBS_SOURCES})

# QTensor
pybind11_add_module(qtensor_module SHARED src/qilisdk_cpp/core/qtensor_bindings.cpp)
file(GLOB QTENSOR_SOURCES src/qilisdk_cpp/core/*.cpp)
target_sources(qtensor_module PRIVATE ${QTENSOR_SOURCES})
target_link_libraries(qtensor_module PRIVATE qilisdk_libs)
set_target_properties(qtensor_module PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)

# QiliSim
pybind11_add_module(qilisim_module SHARED src/qilisdk_cpp/backends/qilisim/qilisim_bindings.cpp)
file(GLOB QILISIM_SOURCES src/qilisdk_cpp/backends/qilisim/*.cpp)
file(GLOB QILISIM_SUBDIR_SOURCES src/qilisdk_cpp/backends/qilisim/*/*.cpp)
list(APPEND QILISIM_SOURCES ${QILISIM_SUBDIR_SOURCES})
target_sources(qilisim_module PRIVATE ${QILISIM_SOURCES})
target_link_libraries(qilisim_module PRIVATE qilisdk_libs)
set_target_properties(qilisim_module PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)

# List of all modules to build
set(MODULES_TO_BUILD qilisdk_libs qtensor_module qilisim_module)

# For each module
foreach(module ${MODULES_TO_BUILD})
    message(STATUS "[QiliSDK C++] Configuring module: ${module}")

    # Set compile options
    set (CMAKE_CXX_STANDARD 17)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(STATUS "[QiliSDK C++] Using GCC/Clang specific compile options")
        target_compile_options(${module} PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    # Make everything visible, so that the test suite can access internal functions
    target_compile_options(${module} PRIVATE -fvisibility=default)

    # Link libraries and include directories
    target_link_libraries(${module} PUBLIC Eigen3::Eigen)
    target_include_directories(${module} PRIVATE ${pybind11_INCLUDE_DIRS})
    if (OpenMP_CXX_FOUND)
        message(STATUS "[QiliSDK C++] Using OpenMP")
        target_link_libraries(${module} PRIVATE OpenMP::OpenMP_CXX)
    endif()

    # Enable if you want to run clang-tidy on the code
    if (DEFINED tidy AND tidy STREQUAL "ON")
        message(STATUS "[QiliSDK C++] Enabling clang-tidy")
        set_target_properties(${module} PROPERTIES
            CXX_CLANG_TIDY
            "clang-tidy;--extra-arg=-Wno-ignored-optimization-argument"
        )
    endif()

    # If told to build with coverage, add the appropriate flags to compile and link
    if (DEFINED coverage AND coverage STREQUAL "ON")
        message(STATUS "[QiliSDK C++] Enabling coverage flags for module: ${module}")
        target_compile_options(${module} PRIVATE --coverage -fno-elide-constructors -fno-default-inline)
        target_link_options(${module} PRIVATE --coverage)
    endif()

    # Compile
    install(TARGETS ${module} DESTINATION .)

    # Also copy the .so file to the tests/unit_cpp directory so that the test suite can find it when running
    add_custom_command(TARGET ${module} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:${module}>
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit_cpp/$<TARGET_FILE_NAME:${module}>
        COMMENT "Copying ${module} to test directory"
    )

    # Need to copy coverage files to a separate directory to avoid them being deleted by CMake's cleanup
    if (DEFINED coverage AND coverage STREQUAL "ON")
        set(COVERAGE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit_cpp/coverage)
        file(MAKE_DIRECTORY ${COVERAGE_DIR})
        add_custom_command(TARGET ${module} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_BINARY_DIR}/CMakeFiles/${module}.dir
                ${COVERAGE_DIR}
            COMMENT "Copying .gcno files to coverage directory for module: ${module}"
        )
    endif()

endforeach()

# If we should also build the test suite
if (DEFINED tests AND tests STREQUAL "ON")
    message(STATUS "[QiliSim] Building test suite")

    find_package(Python3 COMPONENTS Interpreter Development.Embed REQUIRED)
    message(STATUS "Python libraries: ${Python3_LIBRARIES}")
    message(STATUS "Python include dirs: ${Python3_INCLUDE_DIRS}")
    message(STATUS "Python library dirs: ${Python3_LIBRARY_DIRS}")
    
    # Get googletest for C++ testing
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    )
    FetchContent_MakeAvailable(googletest)
    enable_testing()

    # Set up the test executable, compile options, and link libraries
    add_executable(
        test_cpp
        tests/unit_cpp/test_cpp.cpp
    )
    file(GLOB TEST_SOURCES tests/unit_cpp/*.cpp)
    file(GLOB TEST_SUBDIR_SOURCES tests/unit_cpp/*/*.cpp)
    list(APPEND TEST_SOURCES ${TEST_SUBDIR_SOURCES})
    target_sources(test_cpp PRIVATE ${TEST_SOURCES})
    target_include_directories(test_cpp PRIVATE src/qilisdk_cpp/backends/qilisim)
    target_include_directories(test_cpp PRIVATE ${pybind11_INCLUDE_DIRS})
    target_include_directories(test_cpp PRIVATE ${Python_INCLUDE_DIRS})
    if (DEFINED coverage AND coverage STREQUAL "ON")
        message(STATUS "[QiliSDK C++] Enabling coverage flags for test suite")
        target_compile_options(test_cpp PRIVATE --coverage -fno-elide-constructors -fno-default-inline)
        target_link_options(test_cpp PRIVATE --coverage)
    endif()
    target_link_libraries(test_cpp PRIVATE gcov GTest::gtest Eigen3::Eigen)
    target_link_directories(test_cpp PRIVATE ${Python3_LIBRARY_DIRS})
    target_link_libraries(test_cpp PRIVATE pybind11::embed)
    foreach(module ${MODULES_TO_BUILD})
        target_link_libraries(test_cpp PRIVATE ${module})
    endforeach()

    # Google test discovery
    include(GoogleTest)
    gtest_discover_tests(test_cpp)
    set_target_properties(test_cpp PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit_cpp)

    # Make it check next to itself for the modules
    set_target_properties(test_cpp PROPERTIES
        BUILD_RPATH ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit_cpp
        INSTALL_RPATH ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit_cpp
    )
    
    # Need to copy coverage files to a separate directory to avoid them being deleted by CMake's cleanup
    set(COVERAGE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit_cpp/coverage)
    file(MAKE_DIRECTORY ${COVERAGE_DIR})
    add_custom_command(TARGET test_cpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_BINARY_DIR}/CMakeFiles/test_cpp.dir
            ${COVERAGE_DIR}
        COMMENT "Copying .gcno files to coverage directory"
    )

endif()
