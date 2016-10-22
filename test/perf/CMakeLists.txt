set(APP_NAME "test_simple_net")

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/utils
)

add_executable(
    ${APP_NAME}
    ${CMAKE_SOURCE_DIR}/test/perf/test_simple_net.cpp
)
target_link_libraries(${APP_NAME} xnor_nn)

add_custom_target(
    perf
    COMMAND ./${APP_NAME}
    DEPENDS ${APP_NAME}
)
