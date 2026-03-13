cmake_minimum_required(VERSION 3.18)

project(VectorAddCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(Include)

set(SOURCES
    src/main.cu
    src/Kernels.cu
    src/utils.cpp
)

add_executable(VectorAdd ${SOURCES})

set_target_properties(VectorAdd PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 14
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CUDA_FLAGS_DEBUG "-G -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")