project(xnor_nn)
cmake_minimum_required(VERSION 2.8)

set(LIB_NAME xnor_nn)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE is unset, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

# uncomment it for cross-compilation, also change arch flags
#set(CMAKE_SYSROOT "/usr/arm-linux-gnueabi")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    set(CPU_ARCHITECTURE x86)
    set(CCXX_ARCH_FLAGS "-march=native")
else(CMAKE_SYSTEM_PROCESSOR MATCHES "(armv7)|(armv8)")
    set(CPU_ARCHITECTURE ARM)
    set(CCXX_ARCH_FLAGS "-mfpu=neon -march=native")
endif()

set(OPENMP_FLAGS "-fopenmp")
set(CCXX_WARN_FLAGS "-Wall -Werror -Wno-unknown-pragmas")
set(CMAKE_CCXX_FLAGS "${CMAKE_CCXX_FLAGS} ${CCXX_ARCH_FLAGS} ${OPENMP_FLAGS} ${CCXX_WARN_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_CCXX_FLAGS} -std=c11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CCXX_FLAGS} -std=c++11")

include_directories(include)

enable_testing()

# TEMPRORARY
#string(REPLACE "-O3" "-O2" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
#string(REPLACE "-O3" "-O2" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")

add_subdirectory(src)
add_subdirectory(example)
add_subdirectory(external)
add_subdirectory(test)
