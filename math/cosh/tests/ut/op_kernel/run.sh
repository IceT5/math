#!/bin/bash

set -e

echo "=== Cosh UT Test Setup ==="

# 创建数据目录
mkdir -p data
chmod -R 755 data

# 生成测试数据
echo "Generating test data..."
cd data
python3 gen_data.py
cd ..

echo "Test data generated successfully!"

# 创建构建目录
mkdir -p build
cd build

# 配置和编译
echo "Configuring and building..."
cmake ..
make -j$(nproc)

# 运行测试
echo "Running tests..."
./test_cosh

echo "=== All tests passed! ==="