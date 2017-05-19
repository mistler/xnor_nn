set(APP_NAME "gtest")

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}/external/gtest
    )

file(GLOB PRIM_TEST_CASES_SRC RELATIVE ${CMAKE_SOURCE_DIR}/test/gtest
    test_data_bin_reference.cpp
    test_weights_bin_reference.cpp

    test_convolution_reference_forward.cpp

    test_binarization.cpp
    test_convolution_compare_with_reference.cpp
)

foreach(TEST_FILE ${PRIM_TEST_CASES_SRC})
    get_filename_component (TEST_FILE_WE ${TEST_FILE} NAME_WE)
    set(CURR_APP_NAME ${TEST_FILE_WE})
    add_executable(${CURR_APP_NAME} main.cpp ${TEST_FILE})
    target_link_libraries(${CURR_APP_NAME} ${LIB_NAME} gtest)
    add_test(${CURR_APP_NAME} ${CURR_APP_NAME})
endforeach()
