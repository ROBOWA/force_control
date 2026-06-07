cmake_minimum_required(VERSION 3.14)
project(franka_python_bindings VERSION ${libfranka_VERSION})

# Set C++ standard - using C++17 to match parent project
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    # Try to find pybind11 via pip
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE pybind11_RESULT
    )
    if(pybind11_RESULT EQUAL 0)
        list(APPEND CMAKE_PREFIX_PATH "${pybind11_DIR}")
        find_package(pybind11 CONFIG REQUIRED)
    else()
        message(FATAL_ERROR "pybind11 not found. Install with: pip install pybind11")
    endif()
endif()

find_package(Eigen3 REQUIRED)

# Create Python module
pybind11_add_module(_pylibfranka
    src/pylibfranka.cpp
)

# Include directories
target_include_directories(_pylibfranka PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# Link against the shared libfranka library
target_link_libraries(_pylibfranka PRIVATE
    franka
    Eigen3::Eigen3
)

# Install the Python module
install(TARGETS _pylibfranka DESTINATION pylibfranka)

# Copy Python package files
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    DESTINATION pylibfranka
)
