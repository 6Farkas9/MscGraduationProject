cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(BackEnd)

# set(CMAKE_GENERATOR_TOOLSET "version=17.0" CACHE STRING "VS 2022 Toolset" FORCE)
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc.exe")
enable_language(CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)

# 查找 libtorch 包
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libs/libtorch-release")
find_package(Torch REQUIRED)

find_library(NVTOOLSEXT_LIBRARY NAMES nvToolsExt64_1 PATHS "C:/Program Files/NVIDIA Corporation/NvToolsExt/lib/x64")
if (NVTOOLSEXT_LIBRARY)
    message(STATUS "Found nvToolsExt: ${NVTOOLSEXT_LIBRARY}")
else()
    message(FATAL_ERROR "nvToolsExt library not found")
endif()

# 添加可执行文件
add_executable(${PROJECT_NAME} main.cpp)

# 链接 libtorch 库
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})

# 启用 C++14
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)