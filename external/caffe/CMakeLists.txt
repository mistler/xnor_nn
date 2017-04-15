# TODO: use external project
set(CAFFE_SOURCE_DIR "${PROJECT_SOURCE_DIR}/external/caffe")
set(CAFFE_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(CAFFE_RUNNER "${CAFFE_BUILD_DIR}/tools/caffe")

add_custom_command(OUTPUT ${CAFFE_RUNNER}
    COMMAND
        XNOR_NN_CAFFE_BUILD_DIR=${CAFFE_BUILD_DIR}
        XNOR_NN_INCLUDE="${PROJECT_SOURCE_DIR}/include"
        XNOR_NN_LIB="${CMAKE_BINARY_DIR}/src"
        XNOR_NN_DEBUG=${XNOR_NN_DEBUG}
        $(MAKE) all
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

set(CAFFE_TIME_RUN ${CAFFE_RUNNER} time)
set(CAFFE_TRAIN_RUN ${CAFFE_RUNNER} train)

set(CAFFE_MODEL_PARAM -model=${CAFFE_SOURCE_DIR}/models)
set(CAFFE_SOLVER_PARAM -solver=${CAFFE_SOURCE_DIR}/models)

set(CAFFE_GREP 2>&1 | grep -E 'conv.*forward: |Average Forward pass: '
    | sed 's/.*]//g')

MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
      LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()

SUBDIRLIST(CAFFE_MODELS ${CAFFE_SOURCE_DIR}/models)

foreach(CURRENT_MODEL ${CAFFE_MODELS})
    add_custom_target(
        caffe_train_${CURRENT_MODEL}
        WORKING_DIRECTORY ${CAFFE_SOURCE_DIR}
        DEPENDS ${CAFFE_RUNNER}

        COMMAND ${CAFFE_TRAIN_RUN}
            ${CAFFE_SOLVER_PARAM}/${CURRENT_MODEL}/solver.prototxt
    )
endforeach()

add_custom_target(
    caffe_time
    WORKING_DIRECTORY ${CAFFE_BUILD_DIR}
    DEPENDS ${CAFFE_RUNNER}

    COMMAND echo ___${CMAKE_SYSTEM_PROCESSOR}_caffe_time___
    COMMAND echo caffe_task
    COMMAND ${CAFFE_TIME_RUN}
        -iterations=800 ${CAFFE_MODEL_PARAM}/caffe_task/time.prototxt
        ${CAFFE_GREP}

    COMMAND echo && echo xnor_nn_task
    COMMAND ${CAFFE_TIME_RUN}
        -iterations=800 ${CAFFE_MODEL_PARAM}/xnor_nn_task/time.prototxt
        ${CAFFE_GREP}

    COMMAND echo && echo caffe_alexnet
    COMMAND ${CAFFE_TIME_RUN}
        -iterations=8 ${CAFFE_MODEL_PARAM}/caffe_alexnet/time.prototxt
        ${CAFFE_GREP}

    COMMAND echo && echo xnor_nn_alexnet
    COMMAND ${CAFFE_TIME_RUN}
        -iterations=8 ${CAFFE_MODEL_PARAM}/xnor_nn_alexnet/time.prototxt
        ${CAFFE_GREP}

    COMMAND echo && echo caffe_cifar10
    COMMAND ${CAFFE_TIME_RUN}
        -iterations=800 ${CAFFE_MODEL_PARAM}/caffe_cifar10/time.prototxt
        ${CAFFE_GREP}

    COMMAND echo && echo xnor_nn_cifar10
    COMMAND ${CAFFE_TIME_RUN}
        -iterations=800 ${CAFFE_MODEL_PARAM}/xnor_nn_cifar10/time.prototxt
        ${CAFFE_GREP}

    COMMAND echo && echo caffe_mnist
    COMMAND ${CAFFE_TIME_RUN}
        -iterations=800 ${CAFFE_MODEL_PARAM}/caffe_mnist/time.prototxt
        ${CAFFE_GREP}

    COMMAND echo && echo xnor_nn_mnist
    COMMAND ${CAFFE_TIME_RUN}
        -iterations=800 ${CAFFE_MODEL_PARAM}/xnor_nn_mnist/time.prototxt
        ${CAFFE_GREP}
)
