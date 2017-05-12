# TODO: use external project
set(TENSORFLOW_SOURCE_DIR "${PROJECT_SOURCE_DIR}/external/tensorflow")
set(TENSORFLOW_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(TENSORFLOW_BAZEL_CONFIG "${TENSORFLOW_SOURCE_DIR}/.tf_configure.bazelrc")
set(TENSORFLOW_XNORNN_SOURCE "${TENSORFLOW_SOURCE_DIR}/tensorflow/core/user_ops/xnornn_convolution.cc")
set(TENSORFLOW_XNORNN_LIBRARY "${TENSORFLOW_BUILD_DIR}/xnornn_convolution.so")

add_custom_command(OUTPUT ${TENSORFLOW_BAZEL_CONFIG}
    COMMAND
        XNOR_NN_TENSORFLOW_BUILD_DIR=${TENSORFLOW_BUILD_DIR}
        XNOR_NN_INCLUDE="${PROJECT_SOURCE_DIR}/include"
        XNOR_NN_LIB="${CMAKE_BINARY_DIR}/src"
        XNOR_NN_DEBUG=${XNOR_NN_DEBUG}
        ./configure
    WORKING_DIRECTORY "${TENSORFLOW_SOURCE_DIR}"
)
add_custom_target(
    tensorflow_config
    DEPENDS ${TENSORFLOW_BAZEL_CONFIG}
)


add_custom_target(
    tensorflow_build
    COMMAND bazel build --local_resources 2048,2.0,1.0 --config=opt //tensorflow/tools/pip_package:build_pip_package
    WORKING_DIRECTORY "${TENSORFLOW_SOURCE_DIR}"
)

add_custom_target(
    tensorflow_install
    COMMAND bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
        && sudo -H pip install --upgrade /tmp/tensorflow_pkg/*.whl
    WORKING_DIRECTORY "${TENSORFLOW_SOURCE_DIR}"
    DEPENDS tensorflow_build
)

add_custom_command(OUTPUT ${TENSORFLOW_XNORNN_LIBRARY}
    VERBATIM COMMAND sh -c "export TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())') && export LIBRARY_PATH=${CMAKE_BINARY_DIR}/src:$LIBRARY_PATH && ${CMAKE_CXX_COMPILER} -shared ${TENSORFLOW_XNORNN_SOURCE} -o ${TENSORFLOW_XNORNN_LIBRARY} -fPIC -I $TF_INC -I ${PROJECT_SOURCE_DIR}/include ${CMAKE_CXX_FLAGS} -lxnor_nn"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS ${TENSORFLOW_XNORNN_SOURCE}
)
add_custom_target(
    tensorflow_xnornn
    DEPENDS ${TENSORFLOW_XNORNN_LIBRARY}
)

add_custom_target(
    tensorflow_test
    COMMAND
        export LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/src:$LD_LIBRARY_PATH
        && python ${TENSORFLOW_SOURCE_DIR}/tensorflow/core/user_ops/test_xnornn_convolution.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS ${TENSORFLOW_XNORNN_LIBRARY}
)

add_custom_target(
    tensorflow_clean
    COMMAND rm -rf /tmp/tensorflow_pkg && bazel clean
    COMMAND rm ${TENSORFLOW_XNORNN_LIBRARY}
    WORKING_DIRECTORY "${TENSORFLOW_SOURCE_DIR}"
)
