cmake_minimum_required(VERSION 2.8.3)
add_compile_options(-std=c++11)
project(yolo2lidar)

set(DARKNET_DIR "darknet")
set(DARKNET_SRC_DIR "${DARKNET_DIR}/src")
add_definitions(-DDARKNET_FILE_PATH="${DARKNET_DIR}")
set(INCLUDE_SRC "src")

# Find CUDA
find_package(CUDA QUIET)
if (CUDA_FOUND)
  find_package(CUDA REQUIRED)
  message(STATUS "CUDA Version: ${CUDA_VERSION_STRINGS}")
  message(STATUS "CUDA Libararies: ${CUDA_LIBRARIES}")
  set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3
    -gencode arch=compute_30,code=sm_30
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_50,code=[sm_50,compute_50]
    -gencode arch=compute_52,code=[sm_52,compute_52]
    -gencode arch=compute_61,code=sm_61
    -gencode arch=compute_62,code=sm_62
  )
  add_definitions(-DGPU)
else()
  list(APPEND LIBRARIES "m")
endif()

# Find X11
message ( STATUS "Searching for X11..." )
find_package ( X11 REQUIRED )
if ( X11_FOUND )
  include_directories ( ${X11_INCLUDE_DIR} )
  link_libraries ( ${X11_LIBRARIES} )
  message ( STATUS " X11_INCLUDE_DIR: " ${X11_INCLUDE_DIR} )
  message ( STATUS " X11_LIBRARIES: " ${X11_LIBRARIES} )
endif ( X11_FOUND )

# Find rquired packeges
find_package(Boost REQUIRED COMPONENTS thread)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(catkin REQUIRED
  COMPONENTS
    cv_bridge
    roscpp
    std_msgs
    image_transport
    actionlib
    pcl_ros
    darknet_ros_msgs
)

# Enable OPENCV in darknet
add_definitions(-DOPENCV)
add_definitions(-O4 -g)


catkin_package(
  INCLUDE_DIRS
    include
  LIBRARIES
    ${PROJECT_NAME}_lib
  CATKIN_DEPENDS
    cv_bridge
    roscpp
    std_msgs
    image_transport
    actionlib
    pcl_ros
  DEPENDS
    Boost
)

include_directories(
  ${DARKNET_DIR}/include
  ${DARKNET_SRC_DIR}
  include
  ${Boost_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

if (CUDA_FOUND)

  link_directories(
    /usr/local/cuda/lib64
  )

  cuda_add_library(${PROJECT_NAME}_lib
    include/yolo2lidar.h
    src/yolo2lidar.cpp

    ${DARKNET_SRC_DIR}/activation_layer.c        ${DARKNET_SRC_DIR}/im2col.c
    ${DARKNET_SRC_DIR}/activations.c             ${DARKNET_SRC_DIR}/image.c
    ${DARKNET_SRC_DIR}/avgpool_layer.c           ${DARKNET_SRC_DIR}/layer.c
    ${DARKNET_SRC_DIR}/batchnorm_layer.c         ${DARKNET_SRC_DIR}/list.c
    ${DARKNET_SRC_DIR}/blas.c                    ${DARKNET_SRC_DIR}/local_layer.c
    ${DARKNET_SRC_DIR}/box.c                     ${DARKNET_SRC_DIR}/lstm_layer.c
    ${DARKNET_SRC_DIR}/classifier.c              ${DARKNET_SRC_DIR}/matrix.c
    ${DARKNET_SRC_DIR}/col2im.c                  ${DARKNET_SRC_DIR}/maxpool_layer.c
    ${DARKNET_SRC_DIR}/connected_layer.c         ${DARKNET_SRC_DIR}/network.c
    ${DARKNET_SRC_DIR}/conv_lstm_layer.c         ${DARKNET_SRC_DIR}/normalization_layer.c
    ${DARKNET_SRC_DIR}/convolutional_layer.c     ${DARKNET_SRC_DIR}/option_list.c
    ${DARKNET_SRC_DIR}/cost_layer.c              ${DARKNET_SRC_DIR}/parser.c
    ${DARKNET_SRC_DIR}/crnn_layer.c              ${DARKNET_SRC_DIR}/region_layer.c
    ${DARKNET_SRC_DIR}/crop_layer.c              ${DARKNET_SRC_DIR}/reorg_layer.c
    ${DARKNET_SRC_DIR}/dark_cuda.c               ${DARKNET_SRC_DIR}/reorg_old_layer.c
    ${DARKNET_SRC_DIR}/data.c                    ${DARKNET_SRC_DIR}/rnn_layer.c
    ${DARKNET_SRC_DIR}/deconvolutional_layer.c   ${DARKNET_SRC_DIR}/route_layer.c
    ${DARKNET_SRC_DIR}/demo.c                    ${DARKNET_SRC_DIR}/sam_layer.c
    ${DARKNET_SRC_DIR}/detection_layer.c         ${DARKNET_SRC_DIR}/scale_channels_layer.c
    ${DARKNET_SRC_DIR}/dropout_layer.c           ${DARKNET_SRC_DIR}/shortcut_layer.c
    ${DARKNET_SRC_DIR}/gaussian_yolo_layer.c     ${DARKNET_SRC_DIR}/softmax_layer.c
    ${DARKNET_SRC_DIR}/gemm.c                    ${DARKNET_SRC_DIR}/tree.c
    ${DARKNET_SRC_DIR}/getopt.c                  ${DARKNET_SRC_DIR}/utils.c
    ${DARKNET_SRC_DIR}/gettimeofday.c            ${DARKNET_SRC_DIR}/upsample_layer.c
    ${DARKNET_SRC_DIR}/gru_layer.c               ${DARKNET_SRC_DIR}/yolo_layer.c

    ${DARKNET_SRC_DIR}/art.c                     ${DARKNET_SRC_DIR}/go.c
    ${DARKNET_SRC_DIR}/cpu_gemm.c                ${DARKNET_SRC_DIR}/nightmare.c
    ${DARKNET_SRC_DIR}/captcha.c                 ${DARKNET_SRC_DIR}/rnn_vid.c
    ${DARKNET_SRC_DIR}/cifar.c                   ${DARKNET_SRC_DIR}/rnn.c
    ${DARKNET_SRC_DIR}/compare.c                 ${DARKNET_SRC_DIR}/swag.c
    ${DARKNET_SRC_DIR}/coco.c                    ${DARKNET_SRC_DIR}/super.c
    ${DARKNET_SRC_DIR}/darknet.c                 ${DARKNET_SRC_DIR}/tag.c
    ${DARKNET_SRC_DIR}/detector.c                ${DARKNET_SRC_DIR}/yolo.c
    ${DARKNET_SRC_DIR}/dice.c                    ${DARKNET_SRC_DIR}/voxel.c
    ${DARKNET_SRC_DIR}/writing.c                 ${DARKNET_SRC_DIR}/image_opencv.cpp

    ${DARKNET_SRC_DIR}/activation_kernels.cu     ${DARKNET_SRC_DIR}/crop_layer_kernels.cu
    ${DARKNET_SRC_DIR}/avgpool_layer_kernels.cu  ${DARKNET_SRC_DIR}/deconvolutional_kernels.cu
    ${DARKNET_SRC_DIR}/blas_kernels.cu           ${DARKNET_SRC_DIR}/dropout_layer_kernels.cu
    ${DARKNET_SRC_DIR}/col2im_kernels.cu         ${DARKNET_SRC_DIR}/im2col_kernels.cu
    ${DARKNET_SRC_DIR}/convolutional_kernels.cu  ${DARKNET_SRC_DIR}/maxpool_layer_kernels.cu
    ${DARKNET_SRC_DIR}/network_kernels.cu        ${DARKNET_SRC_DIR}/http_stream.cpp
    ${DARKNET_SRC_DIR}/image_interface.c         ${DARKNET_SRC_DIR}/stb_image.h
  )

  target_link_libraries(${PROJECT_NAME}_lib
    m
    pthread
    stdc++
    cuda
    cudart
    cublas
    curand
    ${Boost_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
  )

  cuda_add_executable(${PROJECT_NAME}
    ${project_HEADERS}
    src/main.cpp
  )


else()

  add_library(${PROJECT_NAME}_lib
    include/yolo2lidar.h
    src/yolo2lidar.cpp

    ${DARKNET_SRC_DIR}/activation_layer.c        ${DARKNET_SRC_DIR}/im2col.c
    ${DARKNET_SRC_DIR}/activations.c             ${DARKNET_SRC_DIR}/image.c
    ${DARKNET_SRC_DIR}/avgpool_layer.c           ${DARKNET_SRC_DIR}/layer.c
    ${DARKNET_SRC_DIR}/batchnorm_layer.c         ${DARKNET_SRC_DIR}/list.c
    ${DARKNET_SRC_DIR}/blas.c                    ${DARKNET_SRC_DIR}/local_layer.c
    ${DARKNET_SRC_DIR}/box.c                     ${DARKNET_SRC_DIR}/lstm_layer.c
    ${DARKNET_SRC_DIR}/classifier.c              ${DARKNET_SRC_DIR}/matrix.c
    ${DARKNET_SRC_DIR}/col2im.c                  ${DARKNET_SRC_DIR}/maxpool_layer.c
    ${DARKNET_SRC_DIR}/connected_layer.c         ${DARKNET_SRC_DIR}/network.c
    ${DARKNET_SRC_DIR}/conv_lstm_layer.c         ${DARKNET_SRC_DIR}/normalization_layer.c
    ${DARKNET_SRC_DIR}/convolutional_layer.c     ${DARKNET_SRC_DIR}/option_list.c
    ${DARKNET_SRC_DIR}/cost_layer.c              ${DARKNET_SRC_DIR}/parser.c
    ${DARKNET_SRC_DIR}/crnn_layer.c              ${DARKNET_SRC_DIR}/region_layer.c
    ${DARKNET_SRC_DIR}/crop_layer.c              ${DARKNET_SRC_DIR}/reorg_layer.c
    ${DARKNET_SRC_DIR}/dark_cuda.c               ${DARKNET_SRC_DIR}/reorg_old_layer.c
    ${DARKNET_SRC_DIR}/data.c                    ${DARKNET_SRC_DIR}/rnn_layer.c
    ${DARKNET_SRC_DIR}/deconvolutional_layer.c   ${DARKNET_SRC_DIR}/route_layer.c
    ${DARKNET_SRC_DIR}/demo.c                    ${DARKNET_SRC_DIR}/sam_layer.c
    ${DARKNET_SRC_DIR}/detection_layer.c         ${DARKNET_SRC_DIR}/scale_channels_layer.c
    ${DARKNET_SRC_DIR}/dropout_layer.c           ${DARKNET_SRC_DIR}/shortcut_layer.c
    ${DARKNET_SRC_DIR}/gaussian_yolo_layer.c     ${DARKNET_SRC_DIR}/softmax_layer.c
    ${DARKNET_SRC_DIR}/gemm.c                    ${DARKNET_SRC_DIR}/tree.c
    ${DARKNET_SRC_DIR}/getopt.c                  ${DARKNET_SRC_DIR}/utils.c
    ${DARKNET_SRC_DIR}/gettimeofday.c            ${DARKNET_SRC_DIR}/upsample_layer.c
    ${DARKNET_SRC_DIR}/gru_layer.c               ${DARKNET_SRC_DIR}/yolo_layer.c
    ${DARKNET_SRC_DIR}/http_stream.cpp
    ${DARKNET_SRC_DIR}/art.c                     ${DARKNET_SRC_DIR}/go.c
    ${DARKNET_SRC_DIR}/cpu_gemm.c                ${DARKNET_SRC_DIR}/nightmare.c
    ${DARKNET_SRC_DIR}/captcha.c                 ${DARKNET_SRC_DIR}/rnn_vid.c
    ${DARKNET_SRC_DIR}/cifar.c                   ${DARKNET_SRC_DIR}/rnn.c
    ${DARKNET_SRC_DIR}/compare.c                 ${DARKNET_SRC_DIR}/swag.c
    ${DARKNET_SRC_DIR}/coco.c                    ${DARKNET_SRC_DIR}/super.c
    ${DARKNET_SRC_DIR}/darknet.c                 ${DARKNET_SRC_DIR}/tag.c
    ${DARKNET_SRC_DIR}/detector.c                ${DARKNET_SRC_DIR}/yolo.c
    ${DARKNET_SRC_DIR}/dice.c                    ${DARKNET_SRC_DIR}/voxel.c
    ${DARKNET_SRC_DIR}/writing.c                 ${DARKNET_SRC_DIR}/image_opencv.cpp
    ${DARKNET_SRC_DIR}/image_interface.c         ${DARKNET_SRC_DIR}/stb_image.h
  )

  target_link_libraries(${PROJECT_NAME}_lib
    m
    pthread
    stdc++
    ${Boost_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
  )

  add_executable(${PROJECT_NAME}
    ${project_HEADERS}
    src/main.cpp
  )



endif()

add_dependencies(${PROJECT_NAME}_lib ${catkin_EXPORTED_TARGETS})

target_link_libraries(${PROJECT_NAME}
  ${PROJECT_NAME}_lib
)
