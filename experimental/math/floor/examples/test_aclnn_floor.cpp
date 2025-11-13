/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <limits>
#include <cstdint>
#include "acl/acl.h"
#include "aclnnop/aclnn_floor.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

// Float16 转换辅助函数
uint16_t float_to_float16(float value) {
    uint32_t f32 = *reinterpret_cast<uint32_t*>(&value);
    uint16_t sign = (f32 >> 16) & 0x8000;
    int32_t exponent = ((f32 >> 23) & 0xff) - 127 + 15;
    uint32_t mantissa = f32 & 0x007fffff;
    
    if (exponent <= 0) {
        return sign;
    } else if (exponent >= 31) {
        return sign | 0x7c00;
    }
    
    return sign | (exponent << 10) | (mantissa >> 13);
}

float float16_to_float(uint16_t value) {
    uint32_t sign = (value & 0x8000) << 16;
    uint32_t exponent = (value & 0x7c00) >> 10;
    uint32_t mantissa = value & 0x03ff;
    
    if (exponent == 0) {
        if (mantissa == 0) {
            uint32_t result = sign;
            return *reinterpret_cast<float*>(&result);
        }
        while (!(mantissa & 0x0400)) {
            mantissa <<= 1;
            exponent--;
        }
        exponent++;
        mantissa &= ~0x0400;
    } else if (exponent == 31) {
        uint32_t result = sign | 0x7f800000 | (mantissa << 13);
        return *reinterpret_cast<float*>(&result);
    }
    
    exponent = exponent - 15 + 127;
    uint32_t result = sign | (exponent << 23) | (mantissa << 13);
    return *reinterpret_cast<float*>(&result);
}

// BFloat16 转换辅助函数
uint16_t float_to_bfloat16(float value) {
    uint32_t f32 = *reinterpret_cast<uint32_t*>(&value);
    return (f32 >> 16) & 0xffff;
}

float bfloat16_to_float(uint16_t value) {
    uint32_t f32 = static_cast<uint32_t>(value) << 16;
    return *reinterpret_cast<float*>(&f32);
}

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

// CPU参考实现
void CpuFloor(const std::vector<float>& input, std::vector<float>& output)
{
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::floor(input[i]);
    }
}

// 验证精度
bool VerifyResult(const std::vector<float>& gpuResult, const std::vector<float>& cpuResult)
{
    if (gpuResult.size() != cpuResult.size()) {
        return false;
    }
    
    for (size_t i = 0; i < gpuResult.size(); ++i) {
        if (gpuResult[i] != cpuResult[i]) {
            LOG_PRINT("Verification failed at index %zu: GPU=%.10f, CPU=%.10f\n", 
                     i, gpuResult[i], cpuResult[i]);
            return false;
        }
    }
    return true;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

// 统一的测试函数
int RunTest(aclrtStream stream, aclDataType dataType, const char* typeName)
{
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-100.0f, 100.0f);
    
    std::vector<int64_t> shape = {1024, 1024, 4};
    const int totalIters = 11;
    const int warmupIters = 1;
    const int testIters = totalIters - warmupIters;
    int correctCount = 0;
    double totalTime = 0.0;
    double minTime = std::numeric_limits<double>::max();
    double maxTime = 0.0;
    std::vector<double> times;
    times.reserve(testIters);
    
    int64_t totalElements = GetShapeSize(shape);

    for (int i = 0; i < totalIters; ++i) {
        std::vector<float> selfHostDataFloat(totalElements);
        std::vector<float> cpuResult(totalElements);
        
        void* selfDeviceAddr = nullptr;
        void* outDeviceAddr = nullptr;
        aclTensor* self = nullptr;
        aclTensor* out = nullptr;
        int ret = 0;
        
        if (dataType == ACL_FLOAT) {
            // Float32 处理逻辑
            std::vector<float> selfHostData(totalElements);
            std::vector<float> outHostData(totalElements, 0);
            
            for (int64_t j = 0; j < totalElements; ++j) {
                selfHostData[j] = dis(gen);
            }
            
            CpuFloor(selfHostData, cpuResult);
            
            ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, ACL_FLOAT, &self);
            CHECK_RET(ret == ACL_SUCCESS, return ret);
            ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, ACL_FLOAT, &out);
            CHECK_RET(ret == ACL_SUCCESS, return ret);
        } else {
            // Float16/BFloat16 处理逻辑
            std::vector<uint16_t> selfHostData(totalElements);
            std::vector<uint16_t> outHostData(totalElements, 0);
            
            for (int64_t j = 0; j < totalElements; ++j) {
                float rawValue = dis(gen);
                if (dataType == ACL_FLOAT16) {
                    uint16_t converted = float_to_float16(rawValue);
                    selfHostDataFloat[j] = float16_to_float(converted);
                    selfHostData[j] = converted;
                } else if (dataType == ACL_BF16) {
                    uint16_t converted = float_to_bfloat16(rawValue);
                    selfHostDataFloat[j] = bfloat16_to_float(converted);
                    selfHostData[j] = converted;
                }
            }
            
            CpuFloor(selfHostDataFloat, cpuResult);
            
            ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, dataType, &self);
            CHECK_RET(ret == ACL_SUCCESS, return ret);
            ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, dataType, &out);
            CHECK_RET(ret == ACL_SUCCESS, return ret);
        }

        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
        auto start = std::chrono::high_resolution_clock::now();

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        void* workspaceAddr = nullptr;

        ret = aclnnFloorGetWorkspaceSize(self, out, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFloorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);

        ret = aclnnFloor(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFloor failed. ERROR: %d\n", ret); return ret);
        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        double iterTime = duration.count();
        
        // 验证结果
        std::vector<float> gpuResult(totalElements);
        
        if (dataType == ACL_FLOAT) {
            ret = aclrtMemcpy(gpuResult.data(), gpuResult.size() * sizeof(float), 
                             outDeviceAddr, totalElements * sizeof(float), 
                             ACL_MEMCPY_DEVICE_TO_HOST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", ret); return ret);
        } else {
            std::vector<uint16_t> gpuResultRaw(totalElements);
            ret = aclrtMemcpy(gpuResultRaw.data(), gpuResultRaw.size() * sizeof(uint16_t), 
                             outDeviceAddr, totalElements * sizeof(uint16_t), 
                             ACL_MEMCPY_DEVICE_TO_HOST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", ret); return ret);
            
            for (int64_t j = 0; j < totalElements; ++j) {
                if (dataType == ACL_FLOAT16) {
                    gpuResult[j] = float16_to_float(gpuResultRaw[j]);
                } else if (dataType == ACL_BF16) {
                    gpuResult[j] = bfloat16_to_float(gpuResultRaw[j]);
                }
            }
        }
        
        if (!VerifyResult(gpuResult, cpuResult)) {
            LOG_PRINT("ERROR: Verification failed at iteration %d for %s\n", i, typeName);
            return -1;
        }

        if (i >= warmupIters) {
            totalTime += iterTime;
            times.push_back(iterTime);
            minTime = std::min(minTime, iterTime);
            maxTime = std::max(maxTime, iterTime);
            correctCount++;
        }

        if (workspaceSize > static_cast<uint64_t>(0)) {
            aclrtFree(workspaceAddr);
        }
        
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(selfDeviceAddr);
        aclrtFree(outDeviceAddr);
    }

    double avgTime = totalTime / testIters;
    double variance = 0.0;
    for (double t : times) {
        variance += (t - avgTime) * (t - avgTime);
    }
    variance /= testIters;
    double stddev = std::sqrt(variance);

    LOG_PRINT("\n==================== Performance Summary (%s) ====================\n", typeName);
    LOG_PRINT("Correct count: %d / %d\n", correctCount, testIters);
    LOG_PRINT("Total time: %.3f ms\n", totalTime);
    LOG_PRINT("Min time: %.6f ms\n", minTime);
    LOG_PRINT("Max time: %.6f ms\n", maxTime);
    LOG_PRINT("Avg time: %.6f ms\n", avgTime);
    LOG_PRINT("Std dev: %.6f ms\n", stddev);
    LOG_PRINT("Variance: %.6f ms^2\n", variance);

    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 测试 Float32
    ret = RunTest(stream, ACL_FLOAT, "Float32");
    if (ret != 0) {
        LOG_PRINT("Float32 test failed\n");
        return ret;
    }

    // 测试 Float16
    ret = RunTest(stream, ACL_FLOAT16, "Float16");
    if (ret != 0) {
        LOG_PRINT("Float16 test failed\n");
        return ret;
    }

    // 测试 BFloat16
    ret = RunTest(stream, ACL_BF16, "BFloat16");
    if (ret != 0) {
        LOG_PRINT("BFloat16 test failed\n");
        return ret;
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("\n========== All Tests Passed ==========\n");

    return 0;
}