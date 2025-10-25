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

/*!
 * \file test_cosh_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_kernel/cosh_tiling_data.h"
#include "../../../op_kernel/cosh_tiling_key.h"

using namespace std;
using namespace ge;

class CoshTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "CoshTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "CoshTilingTest TearDown" << std::endl;
    }
};

TEST_F(CoshTilingTest, cosh_tiling_float16_001)
{
    // 测试用例1: float16数据类型, shape [8, 2048]
    optiling::CoshCompileInfo compileInfo = {32, 191 * 1024};  // 32核心, 191KB UB
    gert::TilingContextPara tilingContextPara(
        "Cosh",
        {
            {{{8, 2048}, {8, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8, 2048}, {8, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    
    uint64_t expectTilingKey = 1;  // ELEMENTWISE_TPL_SCH_MODE_1 for float16
    
    // 基于之前计算的tiling参数序列化
    // totalLength=16384, coreNum=8, tile_element_num=2048
    // 每个核心: start=[0,2048,4096,...], end=[2047,4095,6143,...], count=2048, loop_times=1, tail_elements=2048
    string expectTilingData = 
        "16384 8 "                    // totalLength, coreNum
        "0 2048 4096 6144 8192 10240 12288 14336 "  // core_element_start[0-7]
        "0 0 0 0 0 0 0 0 "            // core_element_start[8-15] (未使用)
        "0 0 0 0 0 0 0 0 "            // core_element_start[16-23] (未使用)
        "0 0 0 0 0 0 0 0 "            // core_element_start[24-31] (未使用)
        "2047 4095 6143 8191 10239 12287 14335 16383 "  // core_element_end[0-7]
        "-1 -1 -1 -1 -1 -1 -1 -1 "    // core_element_end[8-15] (未使用)
        "-1 -1 -1 -1 -1 -1 -1 -1 "    // core_element_end[16-23] (未使用)  
        "-1 -1 -1 -1 -1 -1 -1 -1 "    // core_element_end[24-31] (未使用)
        "2048 2048 2048 2048 2048 2048 2048 2048 "  // core_element_count[0-7]
        "0 0 0 0 0 0 0 0 "            // core_element_count[8-15] (未使用)
        "0 0 0 0 0 0 0 0 "            // core_element_count[16-23] (未使用)
        "0 0 0 0 0 0 0 0 "            // core_element_count[24-31] (未使用)
        "2048 "                        // tile_element_num
        "1 1 1 1 1 1 1 1 "            // core_loop_times[0-7]
        "0 0 0 0 0 0 0 0 "            // core_loop_times[8-15] (未使用)
        "0 0 0 0 0 0 0 0 "            // core_loop_times[16-23] (未使用)
        "0 0 0 0 0 0 0 0 "            // core_loop_times[24-31] (未使用)
        "2048 2048 2048 2048 2048 2048 2048 2048 "  // core_tail_elements[0-7]
        "0 0 0 0 0 0 0 0 "            // core_tail_elements[8-15] (未使用)
        "0 0 0 0 0 0 0 0 "            // core_tail_elements[16-23] (未使用)
        "0 0 0 0 0 0 0 0 ";           // core_tail_elements[24-31] (未使用)
    
    std::vector<size_t> expectWorkspaces = {0};
    
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, 
                    expectTilingData, expectWorkspaces);
}