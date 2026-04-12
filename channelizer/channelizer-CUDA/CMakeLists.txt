cmake_minimum_required(VERSION 3.18)

project(PolyphaseChannelizer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(Include)

set(SOURCES
    src/main.cu
    src/ChannelizerDirect.cu
    src/ChannelizerCublas.cu
    src/utils.cu
)

add_executable(channelizer ${SOURCES})

set_target_properties(channelizer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 14
)

set(CMAKE_CUDA_ARCHITECTURES 87)

find_package(CUDA REQUIRED)
target_link_libraries(channelizer PRIVATE cublas cudart)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CUDA_FLAGS_DEBUG "-G -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")