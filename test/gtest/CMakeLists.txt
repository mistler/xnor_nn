set(APP_NAME "test_gtest")

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/external/gtest)

file(GLOB PRIM_TEST_CASES_SRC test_*.cpp)

add_executable(${APP_NAME} main.cpp ${PRIM_TEST_CASES_SRC})
set_property(TARGET simple_convolution PROPERTY C_STANDARD 99)
target_link_libraries(${APP_NAME} xnor_nn gtest)
add_test(${APP_NAME} ${APP_NAME})
