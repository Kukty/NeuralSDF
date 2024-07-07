cmake_minimum_required(VERSION 3.0.0)
project(neural_sdf VERSION 0.1.0 LANGUAGES CXX)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
set(CMAKE_CUDA_STANDARD 11)


enable_language(CUDA)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

file(GLOB_RECURSE SOURCE_FILES
    "src/*.cpp"
    "main.cpp"
)

set(CUDA_SOURCES src/module_kernel.cu src/cuda_kernel.cu)
find_package(OpenMP)
# Add the executable target
add_executable(neural_sdf ${SOURCE_FILES} ${CUDA_SOURCES})

target_link_libraries(neural_sdf PUBLIC OpenMP::OpenMP_CXX ${CUDA_LIBRARIES})



set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)