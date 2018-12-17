cmake_minimum_required (VERSION 3.1)

include(cmake/safeguards.cmake)

project(nbody
  VERSION 0.1.0
  LANGUAGES C CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
    message(FATAL_ERROR "gcc5.0 or newer required.")
  endif()
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# --- CUDA ---
find_package(CUDA REQUIRED)

list(APPEND CUDA_NVCC_FLAGS -std=c++11 -lineinfo)
list(APPEND CUDA_NVCC_FLAGS -arch=sm_30)
# list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60)
if(NOT CUDA_VERSION_STRING VERSION_LESS 9.0)
  list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_70,code=sm_70)
endif()

# workaround for FindCUDA that uses plain link_libraries
set(CUDA_LIBRARIES PRIVATE ${CUDA_LIBRARIES})

add_subdirectory(src)
