
cuda_add_executable(nbody nbody.cu)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/nbody-solution.cu")
  message(STATUS "Build nbody-solution target.")
  cuda_add_executable(nbody-solution nbody-solution.cu)
endif()
