include_directories(
    ${CMAKE_SOURCE_DIR}/include
    )

add_executable(simple_convolution_c simple_convolution.c)
set_property(TARGET simple_convolution_c PROPERTY C_STANDARD 99)
target_link_libraries(simple_convolution_c ${LIB_NAME})
add_test(simple_convolution_c simple_convolution_c)

add_executable(simple_convolution_cpp simple_convolution.cpp)
set_property(TARGET simple_convolution_cpp PROPERTY CXX_STANDARD 11)
target_link_libraries(simple_convolution_cpp ${LIB_NAME})
add_test(simple_convolution_cpp simple_convolution_cpp)
