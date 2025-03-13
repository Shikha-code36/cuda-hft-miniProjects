cmake_minimum_required(VERSION 3.18)

# Project name
set(PROJECT_NAME market_data_parser)

# Get source files
file(GLOB SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/common/include
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link with common library
target_link_libraries(${PROJECT_NAME} cuda_common_lib)

# Set CUDA-specific compilation options
set_target_properties(${PROJECT_NAME} PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)

# Optimization flags for CUDA
target_compile_options(${PROJECT_NAME} PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Set runtime output directory
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/${PROJECT_NAME}"
)

# Create a sample data directory
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "${CMAKE_BINARY_DIR}/bin/${PROJECT_NAME}/data"
)

# If sample data files exist, copy them to the build directory
if(EXISTS "${CMAKE_SOURCE_DIR}/data/market_data")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/data/market_data"
            "${CMAKE_BINARY_DIR}/bin/${PROJECT_NAME}/data"
    )
endif()

# Print configuration summary
message(STATUS "Configuring ${PROJECT_NAME}")
message(STATUS "  Source files: ${SOURCES}")
message(STATUS "  Include directories: ${CMAKE_CURRENT_SOURCE_DIR}/include, ${CMAKE_SOURCE_DIR}/common/include")