cmake_minimum_required(VERSION 3.10.2)
project(RealtimeDemo)

# 设置包含目录
set( Torch_DIR /mnt/anaconda3/envs/pt1/lib/python3.8/site-packages/torch/share/cmake/Torch )
find_package(Torch REQUIRED)     # // 这里使用命令查找libtorch库

set(OpenCV_DIR /mnt/fengyuan/opencv/opencv3.3/share/OpenCV/)
find_package(OpenCV REQUIRED)    # // 这里使用命名查找OpenCV库

# 查找CUDA
set(CUDA usr/local/cuda)
find_package(CUDA REQUIRED)

find_package(Boost REQUIRED COMPONENTS system filesystem)
find_package(Threads REQUIRED)

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "[/mnt/anaconda3/envs/pt1/lib/python3.8/site-packages/torch/lib:/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64:/mnt/anaconda3/envs/pt1/lib:/mnt/fengyuan/opencv/opencv3.3/lib")
# //解决cmake的warning
include_directories( 
	${Boost_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/include/
)

set(CMAKE_CXX_STANDARD 11)

# 设置库目录
# link_directories("/usr/local/lib")
link_directories(
        ${Boost_LIBRARIES_DIRS}
        lib
)

# 声明源文件列表
set(SOURCE_FILES
    src/main.cpp
    include/yolov7_detector.cpp
    
)

# 创建可执行文件
add_executable(main ${SOURCE_FILES})

# 链接库
target_link_libraries(main
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
        ${TORCH_LIBRARIES}       
        
)
set_property(TARGET main PROPERTY CXX_STANDARD 14)