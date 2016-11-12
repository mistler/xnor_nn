set(APP_NAME "test_simple_net")

string(REPLACE "-O3" "-O0" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

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
    DEPENDS ${LIB_NAME}
)


set(APP_NAME "test_only_convolution")
add_executable(
    ${APP_NAME}
    ${CMAKE_SOURCE_DIR}/test/perf/test_only_convolution.cpp
)
target_link_libraries(${APP_NAME} xnor_nn)

