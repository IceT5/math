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
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_kernel/cosh_tiling_data.h"

using namespace std;
using namespace ge;

namespace {
    std::string BuildCoshTilingDataForSmallTensor(int64_t totalElements, int32_t coreNum, 
                                                 int32_t elementsPerCore, int32_t tileSize) {
        std::string result;
        
        // 基础字段
        result += std::to_string(totalElements) + " ";
        result += std::to_string(coreNum) + " ";
        
        // 核间划分数组 - 均匀分配
        for (int i = 0; i < 32; i++) {
            if (i < coreNum) {
                result += std::to_string(i * elementsPerCore) + " ";
            } else {
                result += "0 ";
            }
        }
        
        for (int i = 0; i < 32; i++) {
            if (i < coreNum) {
                result += std::to_string(i * elementsPerCore + elementsPerCore - 1) + " ";
            } else {
                result += "0 ";
            }
        }
        
        for (int i = 0; i < 32; i++) {
            if (i < coreNum) {
                result += std::to_string(elementsPerCore) + " ";
            } else {
                result += "0 ";
            }
        }
        
        // 核内划分
        result += std::to_string(tileSize) + " ";
        
        // 循环次数数组
        for (int i = 0; i < 32; i++) {
            if (i < coreNum) {
                result += "1 ";  // 每个核循环1次
            } else {
                result += "0 ";
            }
        }
        
        // 尾部元素数组
        for (int i = 0; i < 32; i++) {
            if (i < coreNum) {
                result += std::to_string(elementsPerCore) + " ";  // 尾部元素数=每核元素数
            } else {
                result += "0 ";
            }
        }
        
        return result;
    }
}

class CoshTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "CoshTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "CoshTilingTest TearDown" << std::endl;
    }
};

// 基础功能测试 - FP16小张量
TEST_F(CoshTilingTest, ascend9101_test_tiling_fp16_001) {
    optiling::CoshCompileInfo compileInfo = {32, 196608};  // 核心数32, ubSize = 192KB
    gert::TilingContextPara tilingContextPara(
        "Cosh",
        {
            {{{128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // 1D 小张量
        },
        {
            {{{128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);

    // tilingKey
    uint64_t expectTilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);

    // tilingdata
    string expectTilingData = BuildCoshTilingDataForSmallTensor(
        128,    // totalElements
        32,     // coreNum  
        4,      // elementsPerCore
        4       // tileSize
    );

    std::vector<size_t> expectWorkspaces = {0};  // Cosh不需要workspace
    
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// // 基础功能测试 - FP32小张量
// TEST_F(CoshTilingTest, BasicFP32SmallTensor) {
//     optiling::CoshCompileInfo compileInfo = {};
//     gert::TilingContextPara tilingContextPara(
//         "Cosh",
//         {
//             {{{256}}, ge::DT_FLOAT, ge::FORMAT_ND},  // 1D 小张量
//         },
//         {
//             {{{256}}, ge::DT_FLOAT, ge::FORMAT_ND},
//         },
//         &compileInfo);

//     uint64_t expectTilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
//     std::vector<size_t> expectWorkspaces = {0};
    
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
// }

// // 多核分配测试 - 触发大核小核策略
// TEST_F(CoshTilingTest, MultiCoreDistribution) {
//     optiling::CoshCompileInfo compileInfo = {};
//     gert::TilingContextPara tilingContextPara(
//         "Cosh",
//         {
//             {{{10000}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // 大张量，触发多核
//         },
//         {
//             {{{10000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//         },
//         &compileInfo);

//     uint64_t expectTilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
//     std::vector<size_t> expectWorkspaces = {0};
    
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
// }

// // 边界测试 - 元素数刚好整除核数
// TEST_F(CoshTilingTest, BoundaryDivisibleElements) {
//     optiling::CoshCompileInfo compileInfo = {};
//     gert::TilingContextPara tilingContextPara(
//         "Cosh",
//         {
//             {{{1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // 1024 = 32 * 32
//         },
//         {
//             {{{1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//         },
//         &compileInfo);

//     uint64_t expectTilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
//     std::vector<size_t> expectWorkspaces = {0};
    
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
// }

// // 边界测试 - 质数元素数（最难均衡情况）
// TEST_F(CoshTilingTest, PrimeNumberElements) {
//     optiling::CoshCompileInfo compileInfo = {};
//     gert::TilingContextPara tilingContextPara(
//         "Cosh",
//         {
//             {{{1009}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // 质数
//         },
//         {
//             {{{1009}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//         },
//         &compileInfo);

//     uint64_t expectTilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
//     std::vector<size_t> expectWorkspaces = {0};
    
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
// }

// // 多维张量测试
// TEST_F(CoshTilingTest, MultiDimensionalTensor) {
//     optiling::CoshCompileInfo compileInfo = {};
//     gert::TilingContextPara tilingContextPara(
//         "Cosh",
//         {
//             {{{2, 512, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // 3D张量
//         },
//         {
//             {{{2, 512, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//         },
//         &compileInfo);

//     uint64_t expectTilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
//     std::vector<size_t> expectWorkspaces = {0};
    
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
// }

// // 最小tile大小测试
// TEST_F(CoshTilingTest, MinimumTileSize) {
//     optiling::CoshCompileInfo compileInfo = {};
//     gert::TilingContextPara tilingContextPara(
//         "Cosh",
//         {
//             {{{16}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // 最小tile大小边界
//         },
//         {
//             {{{16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//         },
//         &compileInfo);

//     uint64_t expectTilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
//     std::vector<size_t> expectWorkspaces = {0};
    
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
// }

// // 最大核数限制测试
// TEST_F(CoshTilingTest, MaxCoreNumLimit) {
//     optiling::CoshCompileInfo compileInfo = {};
//     gert::TilingContextPara tilingContextPara(
//         "Cosh",
//         {
//             {{{50000}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // 大张量，测试核数限制
//         },
//         {
//             {{{50000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//         },
//         &compileInfo);

//     uint64_t expectTilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
//     std::vector<size_t> expectWorkspaces = {0};
    
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
// }

// // 错误处理测试 - 不支持的数据类型
// TEST_F(CoshTilingTest, InvalidDataType) {
//     optiling::CoshCompileInfo compileInfo = {};
//     gert::TilingContextPara tilingContextPara(
//         "Cosh",
//         {
//             {{{256}}, ge::DT_INT32, ge::FORMAT_ND},  // 不支持的INT32
//         },
//         {
//             {{{256}}, ge::DT_INT32, ge::FORMAT_ND},
//         },
//         &compileInfo);

//     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
// }

// // 错误处理测试 - 零元素张量
// TEST_F(CoshTilingTest, ZeroElements) {
//     optiling::CoshCompileInfo compileInfo = {};
//     gert::TilingContextPara tilingContextPara(
//         "Cosh",
//         {
//             {{{0}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // 0元素
//         },
//         {
//             {{{0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//         },
//         &compileInfo);

//     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
// }

// // 错误处理测试 - 空形状
// TEST_F(CoshTilingTest, EmptyShape) {
//     optiling::CoshCompileInfo compileInfo = {};
//     gert::TilingContextPara tilingContextPara(
//         "Cosh",
//         {
//             {{{}}},  // 空形状
//         },
//         {
//             {{{}}},
//         },
//         &compileInfo);

//     ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
// }

// // 混合数据类型测试 - 输入输出类型匹配
// TEST_F(CoshTilingTest, MixedDataTypes) {
//     optiling::CoshCompileInfo compileInfo = {};
    
//     // 测试FP16路径
//     gert::TilingContextPara tilingContextParaFP16(
//         "Cosh",
//         {
//             {{{512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//         },
//         {
//             {{{512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//         },
//         &compileInfo);

//     uint64_t expectTilingKeyFP16 = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
//     ExecuteTestCase(tilingContextParaFP16, ge::GRAPH_SUCCESS, expectTilingKeyFP16, "", {0});

//     // 测试FP32路径
//     gert::TilingContextPara tilingContextParaFP32(
//         "Cosh",
//         {
//             {{{512}}, ge::DT_FLOAT, ge::FORMAT_ND},
//         },
//         {
//             {{{512}}, ge::DT_FLOAT, ge::FORMAT_ND},
//         },
//         &compileInfo);

//     uint64_t expectTilingKeyFP32 = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
//     ExecuteTestCase(tilingContextParaFP32, ge::GRAPH_SUCCESS, expectTilingKeyFP32, "", {0});
// }

// // 性能边界测试 - 刚好超过单核处理能力
// TEST_F(CoshTilingTest, PerformanceBoundary) {
//     optiling::CoshCompileInfo compileInfo = {};
//     gert::TilingContextPara tilingContextPara(
//         "Cosh",
//         {
//             {{{4097}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // 刚好超过单核边界
//         },
//         {
//             {{{4097}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//         },
//         &compileInfo);

//     uint64_t expectTilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
//     std::vector<size_t> expectWorkspaces = {0};
    
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
// }

// // 复杂形状测试 - 不规则多维形状
// TEST_F(CoshTilingTest, ComplexIrregularShape) {
//     optiling::CoshCompileInfo compileInfo = {};
//     gert::TilingContextPara tilingContextPara(
//         "Cosh",
//         {
//             {{{3, 17, 89, 23}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // 不规则多维
//         },
//         {
//             {{{3, 17, 89, 23}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//         },
//         &compileInfo);

//     uint64_t expectTilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
//     std::vector<size_t> expectWorkspaces = {0};
    
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
// }

// // 内存对齐边界测试
// TEST_F(CoshTilingTest, MemoryAlignmentBoundary) {
//     optiling::CoshCompileInfo compileInfo = {};
    
//     // 测试各种对齐边界情况
//     std::vector<int64_t> testSizes = {31, 32, 33, 63, 64, 65, 127, 128, 129};
    
//     for (auto size : testSizes) {
//         gert::TilingContextPara tilingContextPara(
//             "Cosh",
//             {
//                 {{{size}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//             },
//             {
//                 {{{size}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//             },
//             &compileInfo);

//         uint64_t expectTilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
//         ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", {0});
//     }
// }

// // 大规模数据测试
// TEST_F(CoshTilingTest, LargeScaleData) {
//     optiling::CoshCompileInfo compileInfo = {};
//     gert::TilingContextPara tilingContextPara(
//         "Cosh",
//         {
//             {{{1000000}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // 百万级元素
//         },
//         {
//             {{{1000000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//         },
//         &compileInfo);

//     uint64_t expectTilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
//     std::vector<size_t> expectWorkspaces = {0};
    
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
// }

// // 核数边界测试
// TEST_F(CoshTilingTest, CoreNumBoundary) {
//     optiling::CoshCompileInfo compileInfo = {};
    
//     // 测试刚好使用最大核数的情况
//     gert::TilingContextPara tilingContextPara(
//         "Cosh",
//         {
//             {{{32000}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // 需要接近32核
//         },
//         {
//             {{{32000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//         },
//         &compileInfo);

//     uint64_t expectTilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
//     std::vector<size_t> expectWorkspaces = {0};
    
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
// }