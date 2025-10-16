/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/cosh_tiling_data.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace CoshNs;

class CoshTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "CoshTiling SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "CoshTiling TearDown " << endl;
    }
};

// 基础功能测试用例 - FP16
TEST_F(CoshTiling, ascend9101_test_tiling_fp16_001)
{
    optiling::CoshCompileInfo compileInfo = {32, 262144, true}; // blockDim=32
    gert::TilingContextPara tilingContextPara(
        "Cosh",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    
    // 根据tiling参数计算期望值
    // totalLength = 1 * 64 * 2 * 64 = 8192
    // coreNum = 8 (假设)
    // tile_element_num = 基于UB大小计算
    uint64_t expectTilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1); // FP16模式
    string expectTilingData = "8192 8 "; // totalLength, coreNum
    // 核间划分和核内划分数据会根据实际计算填充
    
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// FP32数据类型测试
TEST_F(CoshTiling, ascend9101_test_tiling_fp32_002)
{
    optiling::CoshCompileInfo compileInfo = {32, 262144, true};
    gert::TilingContextPara tilingContextPara(
        "Cosh",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    
    uint64_t expectTilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0); // FP32模式
    string expectTilingData = "8192 8 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 大张量测试 - 测试核间划分
TEST_F(CoshTiling, ascend9101_test_tiling_large_tensor_003)
{
    optiling::CoshCompileInfo compileInfo = {32, 1048576, true}; // 更大形状
    gert::TilingContextPara tilingContextPara(
        "Cosh",
        {
            {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    
    uint64_t expectTilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
    string expectTilingData = "1048576 32 "; // 使用最大核心数
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 小张量测试 - 测试核内划分
TEST_F(CoshTiling, ascend9101_test_tiling_small_tensor_004)
{
    optiling::CoshCompileInfo compileInfo = {32, 16, true};
    gert::TilingContextPara tilingContextPara(
        "Cosh",
        {
            {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    
    uint64_t expectTilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
    string expectTilingData = "16 1 "; // 单核心处理
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// BF16数据类型测试
TEST_F(CoshTiling, ascend9101_test_tiling_bf16_005)
{
    optiling::CoshCompileInfo compileInfo = {32, 8192, true};
    gert::TilingContextPara tilingContextPara(
        "Cosh",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        &compileInfo);
    
    uint64_t expectTilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1); // 使用FP16模式
    string expectTilingData = "8192 8 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 失败测试用例 - 空张量
TEST_F(CoshTiling, ascend9101_test_tiling_failed_empty_tensor_006)
{
    optiling::CoshCompileInfo compileInfo = {32, 0, true};
    gert::TilingContextPara tilingContextPara(
        "Cosh",
        {
            {{{1, 0, 2, 64}, {1, 0, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 0, 2, 64}, {1, 0, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara);
}

// 失败测试用例 - 不支持的输入类型
TEST_F(CoshTiling, ascend9101_test_tiling_failed_unsupported_input_type_007)
{
    optiling::CoshCompileInfo compileInfo = {32, 8192, true};
    gert::TilingContextPara tilingContextPara(
        "Cosh",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara);
}

// 失败测试用例 - 输入输出数据类型不匹配
TEST_F(CoshTiling, ascend9101_test_tiling_failed_dtype_mismatch_008)
{
    optiling::CoshCompileInfo compileInfo = {32, 8192, true};
    gert::TilingContextPara tilingContextPara(
        "Cosh",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara);
}

// 失败测试用例 - 输入输出形状不匹配
TEST_F(CoshTiling, ascend9101_test_tiling_failed_shape_mismatch_009)
{
    optiling::CoshCompileInfo compileInfo = {32, 8192, true};
    gert::TilingContextPara tilingContextPara(
        "Cosh",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 3, 64}, {1, 64, 3, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara);
}

// 边界值测试 - 最小tile大小
TEST_F(CoshTiling, ascend9101_test_tiling_min_tile_010)
{
    optiling::CoshCompileInfo compileInfo = {32, 16, true};
    gert::TilingContextPara tilingContextPara(
        "Cosh",
        {
            {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    
    uint64_t expectTilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
    string expectTilingData = "16 1 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 测试动态形状
TEST_F(CoshTiling, ascend9101_test_tiling_dynamic_shape_011)
{
    optiling::CoshCompileInfo compileInfo = {32, 8192, true};
    gert::TilingContextPara tilingContextPara(
        "Cosh",
        {
            {{{1, -1, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}, // 动态维度
        },
        {
            {{{1, -1, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    
    uint64_t expectTilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
    string expectTilingData = "8192 8 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 测试不同格式
TEST_F(CoshTiling, ascend9101_test_tiling_different_format_012)
{
    optiling::CoshCompileInfo compileInfo = {32, 8192, true};
    gert::TilingContextPara tilingContextPara(
        "Cosh",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        &compileInfo);
    
    uint64_t expectTilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
    string expectTilingData = "8192 8 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}