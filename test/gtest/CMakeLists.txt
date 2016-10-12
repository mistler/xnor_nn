set(APP_NAME "gtest")

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/external/gtest)

file(GLOB PRIM_TEST_CASES_SRC RELATIVE
    ${CMAKE_SOURCE_DIR}/test/gtest test_*.cpp)

foreach(TEST_FILE ${PRIM_TEST_CASES_SRC})
    set(CURR_APP_NAME ${APP_NAME}_${TEST_FILE})
    add_executable(${CURR_APP_NAME} main.cpp ${TEST_FILE})
    target_link_libraries(${CURR_APP_NAME} xnor_nn gtest)
    add_test(${CURR_APP_NAME} ${CURR_APP_NAME})
endforeach()
