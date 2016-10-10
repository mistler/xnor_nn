project(xnor_nn)
cmake_minimum_required(VERSION 2.8)

set(LIB_NAME xnor_nn)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE is unset, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

set(OPENMP_FLAGS "-fopenmp")
set(CCXX_WARN_FLAGS "-Wall -Werror -Wno-unknown-pragmas")
set(CMAKE_CCXX_FLAGS "${CMAKE_CCXX_FLAGS} ${OPENMP_FLAGS} ${CCXX_WARN_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_CCXX_FLAGS} -std=c99")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CCXX_FLAGS} -std=c++11")

include_directories(include)

enable_testing()

add_subdirectory(external)
add_subdirectory(src)
add_subdirectory(example)
add_subdirectory(test)
