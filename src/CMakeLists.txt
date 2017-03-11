set(TARGET_NAME ${LIB_NAME})

# DEFAULT
file(GLOB_RECURSE DEFAULT_SOURCES
    ${CMAKE_SOURCE_DIR}/src/default/*default.cpp)
set(DEFAULT_OBJ_LIB default)

add_library(${DEFAULT_OBJ_LIB} OBJECT ${DEFAULT_SOURCES})
set_target_properties(${DEFAULT_OBJ_LIB}
    PROPERTIES COMPILE_FLAGS "-fPIC")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")

# AVX
    file(GLOB_RECURSE AVX_SOURCES
        ${CMAKE_SOURCE_DIR}/src/x86/*avx.cpp)
    set(AVX_OBJ_LIB x86_avx)

    add_library(${AVX_OBJ_LIB} OBJECT ${AVX_SOURCES})
    set_target_properties(${AVX_OBJ_LIB}
        PROPERTIES COMPILE_FLAGS "-fPIC -mavx -mpopcnt")

    set(OBJ_LIBS
        $<TARGET_OBJECTS:${DEFAULT_OBJ_LIB}>
        $<TARGET_OBJECTS:${AVX_OBJ_LIB}>)

else(CMAKE_SYSTEM_PROCESSOR MATCHES "(armv7)|(armv8)")

# NEON
    file(GLOB_RECURSE NEON_SOURCES
        ${CMAKE_SOURCE_DIR}/src/arm/*neon.cpp)
    set(NEON_OBJ_LIB arm_neon)

    add_library(${NEON_OBJ_LIB} OBJECT ${NEON_SOURCES})
    set_target_properties(${NEON_OBJ_LIB}
        PROPERTIES COMPILE_FLAGS "-fPIC -mfpu=neon")

    set(OBJ_LIBS
        $<TARGET_OBJECTS:${DEFAULT_OBJ_LIB}>
        $<TARGET_OBJECTS:${NEON_OBJ_LIB}>)

endif()

# TODO: do we really need headers here?
file(GLOB_RECURSE HEADERS
    ${CMAKE_SOURCE_DIR}/include/*.h
    ${CMAKE_SOURCE_DIR}/include/*.hpp
)

file(GLOB SOURCES
    ${CMAKE_SOURCE_DIR}/src/*.c
    ${CMAKE_SOURCE_DIR}/src/*.cpp
)
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/utils
)

add_library(${TARGET_NAME} SHARED ${HEADERS} ${SOURCES} ${OBJ_LIBS})

set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
