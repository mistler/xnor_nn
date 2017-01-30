set(TARGET_NAME ${LIB_NAME})

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    set(CCXX_ARCH_FLAGS "-mpopcnt -mavx")
    include_directories(${CMAKE_SOURCE_DIR}/src/x86)
else(CMAKE_SYSTEM_PROCESSOR MATCHES "(armv7)|(armv8)")
    set(CCXX_ARCH_FLAGS "-mfpu=neon")
    include_directories(${CMAKE_SOURCE_DIR}/src/arm)
endif()

file(GLOB_RECURSE HEADERS
    ${CMAKE_SOURCE_DIR}/include/*.h
    ${CMAKE_SOURCE_DIR}/include/*.hpp
)

file(GLOB_RECURSE SOURCES
    ${CMAKE_SOURCE_DIR}/src/*.c
    ${CMAKE_SOURCE_DIR}/src/*.cpp
    ${CMAKE_SOURCE_DIR}/src/*.h
    ${CMAKE_SOURCE_DIR}/src/*.hpp
)
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}/src/default
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CCXX_ARCH_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} CCXX_ARCH_FLAGS")

add_library(${TARGET_NAME} SHARED ${HEADERS} ${SOURCES})
target_link_libraries(${TARGET_NAME} ${${TARGET_NAME}_LINKER_LIBS})
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
