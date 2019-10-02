cmake_minimum_required(VERSION 3.10)
project(Clock CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA QUIET REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
set(CUDA_SRC
        Reduction.cu
        )

CUDA_COMPILE(CUDA_O ${CUDA_SRC})
add_executable(Clock
        ${CUDA_O}
        main.cpp)