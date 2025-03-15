cmake_minimum_required(VERSION 3.18)
project(CUDA_Mini_Projects LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA settings
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Detect CUDA architecture for the current GPU automatically if possible
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
set(CMAKE_CUDA_ARCHITECTURES ${INSTALLED_GPU_CCS_3})

# If auto-detection fails, fallback to common architectures
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
    message(STATUS "Could not auto-detect CUDA architecture, using default set: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "Using detected CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build" FORCE)
endif()

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/common/include)

# Add common library
add_subdirectory(common)

# Find all project directories and add them
file(GLOB PROJECT_DIRS ${CMAKE_SOURCE_DIR}/projects/*)
foreach(PROJECT_DIR ${PROJECT_DIRS})
    if(IS_DIRECTORY ${PROJECT_DIR} AND EXISTS ${PROJECT_DIR}/CMakeLists.txt)
        add_subdirectory(${PROJECT_DIR})
    endif()
endforeach()

# Add test suite
option(BUILD_TESTS "Build the test suite" ON)
if(BUILD_TESTS AND EXISTS ${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt)
    add_subdirectory(tests)
endif()

# Print build configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")