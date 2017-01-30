include(CMakeForceCompiler)

set(CMAKE_SYSTEM_NAME Linux)

set(CMAKE_SYSROOT "/usr/arm-linux-gnueabi")
set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})

cmake_force_c_compiler(arm-linux-gnueabihf-gcc GNU)
cmake_force_cxx_compiler(arm-linux-gnueabihf-g++ GNU)

set(CCXX_ARCH_FLAGS "-mfpu=neon")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
