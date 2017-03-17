project(xnor_nn)
cmake_minimum_required(VERSION 2.8)

set(LIB_NAME xnor_nn)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE is unset, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

if(CMAKE_BUILD_TYPE MATCHES "^(debug|DEBUG|Debug)$")
    message("Debug mode")
    set(XNOR_NN_DEBUG 1)
else()
    set(XNOR_NN_DEBUG 0)
endif()

#set(OPTREPORT_FLAGS "-qopt-report=5")
set(OPENMP_FLAGS "-fopenmp")
set(CCXX_WARN_FLAGS "-Wall -Werror -Wno-unknown-pragmas")
set(CMAKE_CCXX_FLAGS "${CMAKE_CCXX_FLAGS} ${OPENMP_FLAGS} ${CCXX_WARN_FLAGS} ${OPTREPORT_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_CCXX_FLAGS} -std=c11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CCXX_FLAGS} -std=c++11")

#string(REPLACE "-O3" "-O2" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
#string(REPLACE "-O3" "-O2" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")

include_directories(include)

enable_testing()

add_subdirectory(src)
add_subdirectory(example)
add_subdirectory(external)
add_subdirectory(test)
