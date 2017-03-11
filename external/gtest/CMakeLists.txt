set(TARGET_NAME gtest)
find_package(Threads REQUIRED)

set(MAIN_SRC "gtest-all.cc")

include_directories(${CMAKE_SOURCE_DIR}/external/gtest)
source_group("" FILES ${MAIN_SRC})

add_library(${TARGET_NAME} STATIC ${MAIN_SRC})
target_link_libraries(${TARGET_NAME} ${CMAKE_THREAD_LIBS_INIT})
