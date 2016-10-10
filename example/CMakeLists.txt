include_directories(
    ${CMAKE_SOURCE_DIR}/include
    )

add_executable(simple_convolution simple_convolution.c)
set_property(TARGET simple_convolution PROPERTY C_STANDARD 99)
target_link_libraries(simple_convolution ${LIB_NAME})
add_test(simple_convolution simple_convolution)
