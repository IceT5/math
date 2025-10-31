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
 * \file square_v2_tiling.cpp
 * \brief
*/

#include "log/log.h"
#include "util/math_util.h"
#include "tiling_base/tiling_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "math/square_v2/op_kernel/square_v2_tiling_data.h"
#include "math/square_v2/op_kernel/square_v2_tiling_key.h"

namespace optiling {

using namespace Ops::Math::OpTiling;

const uint32_t BLOCK_SIZE = 32;
const int32_t MAX_CORE_NUM = 32;
const uint32_t MIN_TILE_SIZE = 16;
const uint32_t BUFFER_NUM = 2;
const uint32_t VEC_LEN = 8;
const uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;
constexpr int32_t ATTRPOS0 = 0;
constexpr uint32_t INDEXZERO = 0;
constexpr uint32_t INDEXONE = 1;
constexpr uint32_t INDEXTWO = 2;
constexpr uint32_t INDEXTHREE = 3;

struct SquareV2CompileInfo {};

static uint32_t AlignUp(uint32_t a, uint32_t b) 
{
    if (b == 0)
        return a;
    return (a + b - 1) / b * b;
}

// 获取平台信息如ubSize, coreNum
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    // 获取ubsize coreNum
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();

    // 限制核数不超过最大值
    if (coreNum > MAX_CORE_NUM) {
        coreNum = MAX_CORE_NUM;
    }

    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 获取形状和数据类型信息
static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalElements, 
                                       ge::DataType& dataType, uint32_t& typeSize)
{
    // 获取输入shape信息
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    // 如果输入shape 是标量 转换为{1}，否则保持原 shape 不变
    auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());

    auto outZ = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outZ);

    auto dimNum = inputShapeX.GetDimNum();
    OP_CHECK_IF(dimNum == 0, OP_LOGE(context, "SquareV2: input shape dim is 0"), return ge::GRAPH_FAILED);

    // 计算总元素数量
    totalElements = 1;
    for (size_t i = 0; i < dimNum; ++i) {
        totalElements *= inputShapeX.GetDim(i);
    }
    
    OP_CHECK_IF(totalElements == 0, OP_LOGE(context, "SquareV2: total elements is 0"), return ge::GRAPH_FAILED);

    // dtype校验
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "invalid dtype");
        return ge::GRAPH_FAILED;
    }

    // 获取数据类型    
    switch (dataType) {
        case ge::DT_FLOAT:
            typeSize = 4;
            break;
        case ge::DT_FLOAT16:
            typeSize = 2;
            break;
        default:
            OP_LOGE(context, "SquareV2: unsupported data type");
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// UB内存需求估算
static uint32_t EstimateUBUsage(int32_t elem_num, uint32_t typeSize)
{
    // 输入数据缓存：需要缓存可能被重复访问的输入区域
    uint32_t inputBytes = AlignUp(elem_num * typeSize, BLOCK_SIZE); // 预估4倍输入数据
    // 输出数据
    uint32_t outputBytes = AlignUp(elem_num * typeSize, BLOCK_SIZE);
    // 双缓冲总需求
    uint32_t total = BUFFER_NUM * (inputBytes + outputBytes);
    
    return total;
}

// 计算核间划分 - 大核小核分配策略
static ge::graphStatus CalculateCoreDistribution(int64_t totalElements, int64_t coreNum,
                                               SquareV2TilingData& tilingData)
{
    // 大核小核分配
    int32_t baseElementsPerCore = totalElements / coreNum;
    int32_t bigCoreNum = totalElements % coreNum; // 大核个数
    int32_t bigCoreElements = baseElementsPerCore + 1; // 大核处理的元素数
    int32_t smallCoreElements = baseElementsPerCore; // 小核处理的元素数

    // 计算每个核的元素分配
    int32_t currentElement = 0;
    for (int32_t coreId = 0; coreId < coreNum; coreId++) {
        if (coreId < bigCoreNum) {
            tilingData.core_element_count[coreId] = bigCoreElements;
        } else {
            tilingData.core_element_count[coreId] = smallCoreElements;
        }
        tilingData.core_element_start[coreId] = currentElement;
        
        if (tilingData.core_element_count[coreId] > 0) {
            tilingData.core_element_end[coreId] = currentElement + tilingData.core_element_count[coreId] - 1;
            currentElement += tilingData.core_element_count[coreId];
        } else {
            tilingData.core_element_end[coreId] = currentElement - 1;
        }
    }

    // 对于未使用的核，设置默认值
    for (int32_t coreId = coreNum; coreId < MAX_CORE_NUM; coreId++) {
        tilingData.core_element_start[coreId] = 0;
        tilingData.core_element_end[coreId] = -1;
        tilingData.core_element_count[coreId] = 0;
    }

    return ge::GRAPH_SUCCESS;
}

// 计算核内划分参数
static ge::graphStatus CalculateIntraCorePartition(int64_t coreNum, 
                                                 const SquareV2TilingData& tilingData,
                                                 uint64_t ubSize, uint32_t typeSize,
                                                 int32_t& tileElementNum)
{
    // 计算最大每个核处理的元素数
    int32_t maxElementsPerCore = 0;
    for (int32_t coreId = 0; coreId < coreNum; coreId++) {
        if (tilingData.core_element_count[coreId] > maxElementsPerCore) {
            maxElementsPerCore = tilingData.core_element_count[coreId];
        }
    }

    // 动态调整tile大小以充分利用UB内存
    tileElementNum = 1;
    while (tileElementNum * 2 <= maxElementsPerCore && 
           EstimateUBUsage(tileElementNum * 2, typeSize) <= ubSize * 90 / 100) {
        tileElementNum *= 2;
    }

    // 确保最小tile大小
    if ((uint32_t)tileElementNum < MIN_TILE_SIZE) {
        tileElementNum = std::min(MIN_TILE_SIZE, static_cast<uint32_t>(maxElementsPerCore));
    }

    return ge::GRAPH_SUCCESS;
}

// 计算每个核的循环参数
static void CalculateCoreLoopParams(int64_t coreNum, int32_t tileElementNum, SquareV2TilingData& tilingData)
{
    for (int32_t coreId = 0; coreId < coreNum; coreId++) {
        if (tilingData.core_element_count[coreId] > 0) {
            tilingData.core_loop_times[coreId] = 
                (tilingData.core_element_count[coreId] + tileElementNum - 1) / tileElementNum;
            tilingData.core_tail_elements[coreId] = 
                tilingData.core_element_count[coreId] % tileElementNum;
            if (tilingData.core_tail_elements[coreId] == 0) {
                tilingData.core_tail_elements[coreId] = tileElementNum;
            }
        } else {
            tilingData.core_loop_times[coreId] = 0;
            tilingData.core_tail_elements[coreId] = 0;
        }
    }

    // 对于未使用的核，设置默认值
    for (int32_t coreId = coreNum; coreId < MAX_CORE_NUM; coreId++) {
        tilingData.core_loop_times[coreId] = 0;
        tilingData.core_tail_elements[coreId] = 0;
    }
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0; // SquareV2操作通常不需要workspace
    return ge::GRAPH_SUCCESS;
}

// Tiling分发入口
static ge::graphStatus SquareV2TilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台运行信息
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, 
        OP_LOGE(context, "GetPlatformInfo error"), 
        return ge::GRAPH_FAILED);

    // 2. 获取shape、属性信息
    int64_t totalElements;
    ge::DataType dataType;
    uint32_t typeSize;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalElements, dataType, typeSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), 
        return ge::GRAPH_FAILED);

    // 3. 获取WorkspaceSize信息
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, 
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 4. 初始化tiling数据
    SquareV2TilingData* tiling = context->GetTilingData<SquareV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(SquareV2TilingData), 0, sizeof(SquareV2TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), 
        return ge::GRAPH_FAILED);

    // 设置基础参数
    tiling->totalLength = totalElements;
    tiling->coreNum = coreNum;

    // 5. 计算核间划分（大核小核策略）
    OP_CHECK_IF(
        CalculateCoreDistribution(totalElements, coreNum, *tiling) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "CalculateCoreDistribution error"),
        return ge::GRAPH_FAILED);

    // 6. 计算核内tile大小
    int32_t tileElementNum;
    OP_CHECK_IF(
        CalculateIntraCorePartition(coreNum, *tiling, ubSize, typeSize, tileElementNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "CalculateIntraCorePartition error"),
        return ge::GRAPH_FAILED);

    tiling->tile_element_num = tileElementNum;

    // 7. 计算每个核的循环参数
    CalculateCoreLoopParams(coreNum, tileElementNum, *tiling);

    context->SetBlockDim(coreNum);

    // 8. 设置tiling key（根据数据类型）
    uint64_t tilingKey = 0;
    if (dataType == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
        context->SetTilingKey(tilingKey);
    } 
    else if (dataType == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
        context->SetTilingKey(tilingKey);
    }
    else {
        OP_LOGE(context, "Unsupported data type for tiling key");
        return ge::GRAPH_FAILED;
    }
        
    // 打印调试信息
    OP_LOGI(context, "SquareV2 Tiling: totalElements=%ld, coreNum=%ld, tileElementNum=%d", 
           totalElements, coreNum, tileElementNum);
    
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSquareV2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// Tiling注册入口
IMPL_OP_OPTILING(SquareV2).Tiling(SquareV2TilingFunc).TilingParse<SquareV2CompileInfo>(TilingParseForSquareV2);
} // namespace optiling