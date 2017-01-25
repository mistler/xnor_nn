cmake_minimum_required(VERSION 2.8)

add_subdirectory(gtest)

# TODO: use external project
set(CAFFE_SOURCE_DIR "${PROJECT_SOURCE_DIR}/external/caffe")
set(CAFFE_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/caffe")
set(CAFFE_RUNNER "${CAFFE_BUILD_DIR}/tools/caffe")

add_custom_command(OUTPUT ${CAFFE_RUNNER}
    COMMAND
        XNOR_NN_CAFFE_BUILD_DIR=${CAFFE_BUILD_DIR}
        XNOR_NN_INCLUDE="${PROJECT_SOURCE_DIR}/include"
        XNOR_NN_LIB="${CMAKE_BINARY_DIR}/src"
        make all
    WORKING_DIRECTORY "${CAFFE_SOURCE_DIR}"
)

add_custom_target(
    caffe
    DEPENDS ${CAFFE_RUNNER}
)

add_custom_target(
    caffe_clean
    COMMAND make clean
    WORKING_DIRECTORY "${CAFFE_SOURCE_DIR}"
)

add_custom_target(
    caffe_patch
    COMMAND rm ${CAFFE_RUNNER}
)
