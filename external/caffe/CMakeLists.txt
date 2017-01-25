# TODO: use external project
set(CAFFE_SOURCE_DIR "${PROJECT_SOURCE_DIR}/external/caffe")
set(CAFFE_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(CAFFE_RUNNER "${CAFFE_BUILD_DIR}/tools/caffe")

add_custom_command(OUTPUT ${CAFFE_RUNNER}
    COMMAND
        XNOR_NN_CAFFE_BUILD_DIR=${CAFFE_BUILD_DIR}
        XNOR_NN_INCLUDE="${PROJECT_SOURCE_DIR}/include"
        XNOR_NN_LIB="${CMAKE_BINARY_DIR}/src"
        make all
    WORKING_DIRECTORY "${CAFFE_SOURCE_DIR}"
    DEPENDS ${LIB_NAME}
)

add_custom_target(
    caffe
    DEPENDS ${CAFFE_RUNNER}
)

add_custom_target(
    caffe_clean
    COMMAND rm -rf ${CAFFE_BUILD_DIR}
        ${CAFFE_BUILD_DIR}_debug ${CAFFE_BUILD_DIR}_release
    WORKING_DIRECTORY "${CAFFE_SOURCE_DIR}"
)

add_custom_target(
    caffe_patch
    COMMAND rm ${CAFFE_RUNNER}
)

set(CAFFE_TIME_VARS KMP_AFFINITY=compact,1,0,granularity=fine OMP_NUM_THREADS=1)
set(CAFFE_TIME_RUN ${CAFFE_RUNNER} time)
set(CAFFE_MODEL -model=${CAFFE_SOURCE_DIR}/models)
set(CAFFE_GREP 2>&1 | grep -E 'conv.*forward: |Average Forward pass: '
    | sed 's/.*]//g')

add_custom_target(
    caffe_time

    COMMAND echo ${CPU_ARCHITECTURE}
    COMMAND echo caffe_task
    COMMAND ${CAFFE_TIME_VARS} ${CAFFE_TIME_RUN}
        -iterations=2500 ${CAFFE_MODEL}/caffe_task/train_val.prototxt
        ${CAFFE_GREP}

    COMMAND echo && echo xnor_nn_task
    COMMAND ${CAFFE_TIME_VARS} ${CAFFE_TIME_RUN}
        -iterations=2500 ${CAFFE_MODEL}/xnor_nn_task/train_val.prototxt
        ${CAFFE_GREP}

    COMMAND echo && echo caffe_alexnet
    COMMAND ${CAFFE_TIME_VARS} ${CAFFE_TIME_RUN}
        -iterations=25 ${CAFFE_MODEL}/caffe_alexnet/train_val.prototxt
        ${CAFFE_GREP}

    COMMAND echo && echo xnor_nn_alexnet
    COMMAND ${CAFFE_TIME_VARS} ${CAFFE_TIME_RUN}
        -iterations=25 ${CAFFE_MODEL}/xnor_nn_alexnet/train_val.prototxt
        ${CAFFE_GREP}

    WORKING_DIRECTORY ${CAFFE_BUILD_DIR}
    DEPENDS ${CAFFE_RUNNER}
)
