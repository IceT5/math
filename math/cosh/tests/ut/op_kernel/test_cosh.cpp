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

/*!
 * \file test_cosh.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_kernel/cosh_tiling_data.h"
#include "../../../op_kernel/cosh_tiling_key.h"
#include "data_utils.h"

using namespace std;
using namespace ge;

extern "C" __global__ __aicore__ void cosh(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling);

class cosh_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "cosh_test SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "cosh_test TearDown" << endl;
    }
};

TEST_F(cosh_test, test_float16_basic)
{
    // 测试用例1: float16基础测试 [8, 2048]
    size_t dataSize = 8 * 2048;  // 16,384
    size_t inputByteSize = dataSize * sizeof(uint16_t);
    size_t outputByteSize = dataSize * sizeof(uint16_t);
    size_t tiling_data_size = sizeof(CoshTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    uint32_t blockDim = 8;  // 使用8个核心

    // 设置tiling参数
    CoshTilingData* tilingData = reinterpret_cast<CoshTilingData*>(tiling);
    memset(tilingData, 0, sizeof(CoshTilingData));
    
    tilingData->totalLength = dataSize;  // 16,384
    tilingData->coreNum = 8;
    tilingData->tile_element_num = 2048;  // 每个tile 2048个元素

    // 均匀分配数据到8个核心: 16,384 / 8 = 2,048
    int32_t elementsPerCore = dataSize / 8;  // 2,048
    
    for (int i = 0; i < 8; i++) {
        tilingData->core_element_start[i] = i * elementsPerCore;
        tilingData->core_element_end[i] = (i + 1) * elementsPerCore - 1;
        tilingData->core_element_count[i] = elementsPerCore;
        
        // 循环次数: 2,048 / 2,048 = 1
        tilingData->core_loop_times[i] = 1;
        
        // 尾部元素: 正好整除，设为tile大小
        tilingData->core_tail_elements[i] = 2048;
    }

    // 设置tiling key为float16模式
    ICPU_SET_TILING_KEY(1);  // ELEMENTWISE_TPL_SCH_MODE_1 = 1

    // 初始化测试数据
    uint16_t* input_data = reinterpret_cast<uint16_t*>(x);
    for (int i = 0; i < dataSize; i++) {
        float value = (i % 100) * 0.01f;  // 循环的简单数据
        input_data[i] = FloatToHalf(value);
    }

    // 执行核函数
    ICPU_RUN_KF(cosh, blockDim, x, z, workspace, tiling);

    // 验证结果（抽样验证）
    uint16_t* output_data = reinterpret_cast<uint16_t*>(z);
    for (int i = 0; i < 10; i++) {  // 只验证前10个元素
        float input_val = (i % 100) * 0.01f;
        float expected = std::cosh(input_val);
        float actual = HalfToFloat(output_data[i]);
        
        EXPECT_NEAR(actual, expected, 0.001f) << "Mismatch at index " << i;
    }

    AscendC::GmFree(x);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(cosh_test, test_float_basic)
{
    // 测试用例2: float基础测试 [8, 2048]  
    size_t dataSize = 8 * 2048;  // 16,384
    size_t inputByteSize = dataSize * sizeof(float);
    size_t outputByteSize = dataSize * sizeof(float);
    size_t tiling_data_size = sizeof(CoshTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    uint32_t blockDim = 8;  // 使用8个核心

    // 设置tiling参数
    CoshTilingData* tilingData = reinterpret_cast<CoshTilingData*>(tiling);
    memset(tilingData, 0, sizeof(CoshTilingData));
    
    tilingData->totalLength = dataSize;  // 16,384
    tilingData->coreNum = 8;
    tilingData->tile_element_num = 2048;

    // 均匀分配数据到8个核心: 16,384 / 8 = 2,048
    int32_t elementsPerCore = dataSize / 8;  // 2,048
    
    for (int i = 0; i < 8; i++) {
        tilingData->core_element_start[i] = i * elementsPerCore;
        tilingData->core_element_end[i] = (i + 1) * elementsPerCore - 1;
        tilingData->core_element_count[i] = elementsPerCore;
        tilingData->core_loop_times[i] = 1;
        tilingData->core_tail_elements[i] = 2048;
    }

    // 设置tiling key为float模式
    ICPU_SET_TILING_KEY(0);  // ELEMENTWISE_TPL_SCH_MODE_0 = 0

    // 初始化测试数据
    float* input_data = reinterpret_cast<float*>(x);
    for (int i = 0; i < dataSize; i++) {
        input_data[i] = (i % 100) * 0.01f;
    }

    // 执行核函数
    ICPU_RUN_KF(cosh, blockDim, x, z, workspace, tiling);

    // 验证结果（抽样验证）
    float* output_data = reinterpret_cast<float*>(z);
    for (int i = 0; i < 10; i++) {
        float input_val = (i % 100) * 0.01f;
        float expected = std::cosh(input_val);
        float actual = output_data[i];
        
        EXPECT_NEAR(actual, expected, 0.001f) << "Mismatch at index " << i;
    }

    AscendC::GmFree(x);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}