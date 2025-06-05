cmake_minimum_required(VERSION 3.28)
project(task)

set(CMAKE_CXX_STANDARD 17)

FIND_PACKAGE(OpenMP REQUIRED)
if (OPENMP_FOUND)
    MESSAGE("openmp found")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

link_directories(${CMAKE_SOURCE_DIR}/omp-logs)
add_executable(task stencil.cpp)

target_compile_options(task PRIVATE
        -Wall
        -pedantic
        -O3
        -ffast-math
        -fopenmp
        -DOPENMP
        -Wfatal-errors
)

target_link_libraries(task omp_logs)
