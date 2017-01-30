set(TARGET_NAME ${LIB_NAME})

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")

    file(GLOB_RECURSE AVX_SOURCES
        ${CMAKE_SOURCE_DIR}/src/x86/*avx.cpp)
    set(AVX_OBJ_LIB x86_avx)

    add_library(${AVX_OBJ_LIB} OBJECT ${AVX_SOURCES})
    set_target_properties(${AVX_OBJ_LIB}
        PROPERTIES COMPILE_FLAGS "-fPIC -mavx -mpopcnt -DVLEN=256")

    add_library(${AVX_OBJ_LIB}_templated OBJECT ${AVX_SOURCES})
    set_target_properties(${AVX_OBJ_LIB}_templated
        PROPERTIES COMPILE_FLAGS "-fPIC -mavx -mpopcnt -DVLEN=256 -DTEMPLATED")


    file(GLOB_RECURSE DEFAULT_SOURCES
        ${CMAKE_SOURCE_DIR}/src/default/*default.cpp)
    set(DEFAULT_OBJ_LIB x86_default)

    add_library(${DEFAULT_OBJ_LIB} OBJECT ${DEFAULT_SOURCES})
    set_target_properties(${DEFAULT_OBJ_LIB}
        PROPERTIES COMPILE_FLAGS "-fPIC -msse4 -mpopcnt -DVLEN=128")
    add_library(${DEFAULT_OBJ_LIB}_templated OBJECT ${DEFAULT_SOURCES})
    set_target_properties(${DEFAULT_OBJ_LIB}_templated
        PROPERTIES COMPILE_FLAGS "-fPIC -msse4 -mpopcnt -DVLEN=128 -DTEMPLATED")


    set(OBJ_LIBS $<TARGET_OBJECTS:${DEFAULT_OBJ_LIB}>
        $<TARGET_OBJECTS:${DEFAULT_OBJ_LIB}_templated>
        $<TARGET_OBJECTS:${AVX_OBJ_LIB}>
        $<TARGET_OBJECTS:${AVX_OBJ_LIB}_templated>)

else(CMAKE_SYSTEM_PROCESSOR MATCHES "(armv7)|(armv8)")

    file(GLOB_RECURSE NEON_SOURCES
        ${CMAKE_SOURCE_DIR}/src/arm/*neon.cpp)
    set(NEON_OBJ_LIB arm_neon)

    add_library(${NEON_OBJ_LIB} OBJECT ${NEON_SOURCES})
    set_target_properties(${NEON_OBJ_LIB}
        PROPERTIES COMPILE_FLAGS "-fPIC -mfpu=neon -DVLEN=128")

    add_library(${NEON_OBJ_LIB}_templated OBJECT ${NEON_SOURCES})
    set_target_properties(${NEON_OBJ_LIB}_templated
        PROPERTIES COMPILE_FLAGS "-fPIC -mfpu=neon -DVLEN=128 -DTEMPLATED")

    file(GLOB_RECURSE DEFAULT_SOURCES
        ${CMAKE_SOURCE_DIR}/src/default/*default.cpp)
    set(DEFAULT_OBJ_LIB arm_default)

    add_library(${DEFAULT_OBJ_LIB} OBJECT ${DEFAULT_SOURCES})
    set_target_properties(${DEFAULT_OBJ_LIB}
        PROPERTIES COMPILE_FLAGS "-fPIC -DVLEN=64")

    add_library(${DEFAULT_OBJ_LIB}_templated OBJECT ${DEFAULT_SOURCES})
    set_target_properties(${DEFAULT_OBJ_LIB}_templated
        PROPERTIES COMPILE_FLAGS "-fPIC -DVLEN=64 -DTEMPLATED")


    set(OBJ_LIBS $<TARGET_OBJECTS:${DEFAULT_OBJ_LIB}>
        $<TARGET_OBJECTS:${DEFAULT_OBJ_LIB}_templated>
        $<TARGET_OBJECTS:${NEON_OBJ_LIB}>
        $<TARGET_OBJECTS:${NEON_OBJ_LIB}_templated>)

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
