cmake_minimum_required(VERSION 3.12)
project(cosh_ut_test)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 在昇腾环境中，GTest可能已经预装，我们手动设置路径
set(ASCEND_HOME $ENV{ASCEND_HOME})
if(NOT ASCEND_HOME)
    set(ASCEND_HOME "/usr/local/Ascend")
endif()

# 设置包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ASCEND_HOME}/include
    ${ASCEND_HOME}/tools/include
)

# 设置链接目录
link_directories(
    ${ASCEND_HOME}/lib64
    ${ASCEND_HOME}/tools/lib64
)

# 添加可执行文件
add_executable(test_cosh test_cosh.cpp)

# 链接必要的库
target_link_libraries(test_cosh
    tikicpulib
    gtest
    gtest_main
    pthread
    stdc++
)

# 添加编译选项
target_compile_options(test_cosh PRIVATE -Wno-deprecated-declarations)