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
 * \file power_tiling.cpp
 * \brief
*/
#include "log/log.h"
#include "util/math_util.h"
#include "tiling_base/tiling_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "math/power/op_kernel/power_tiling_data.h"
#include "math/power/op_kernel/power_tiling_key.h"

#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include <vector>
#include <algorithm> 
#include "graph/types.h" 

namespace optiling {

using namespace Ops::Math::OpTiling;
const uint32_t BLOCK_SIZE = 32;
const uint32_t BUFFER_NUM = 2;

const uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;
const int32_t DIMS_LIMIT = 10;

struct PowerCompileInfo {};

// 获取平台信息如ubSize, coreNum
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 获取形状和属性信息
ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalElements, 
                                  ge::DataType& dataType0, ge::DataType& dataType1)
{
    // 获取输入形状信息
    auto inputX1 = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX1);
    auto inputX2 = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX2);
    auto outputZ = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputZ);
    
    // 获取输出元素总数
    totalElements = outputZ->GetStorageShape().GetShapeSize();
    
    // 数据类型校验
    const std::set<ge::DataType> supportedDtype = {
        ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_INT32, 
        ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT
    };
    
    auto inputDesc0 = context->GetInputDesc(0);
    auto inputDesc1 = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc1);
    
    dataType0 = inputDesc0->GetDataType();
    dataType1 = inputDesc1->GetDataType();
    
    if (supportedDtype.count(dataType0) == 0 || supportedDtype.count(dataType1) == 0) {
        OP_LOGE(context, "Unsupported dtype in power operator");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0; // Power算子不需要工作空间
    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
static ge::graphStatus PowerTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context, "Begin to do PowerTilingFunc");
    
    // 1. 获取平台信息
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2. 获取形状和数据类型信息
    int64_t totalElements = 0;
    ge::DataType dataType0, dataType1;
    OP_CHECK_IF(GetShapeAttrsInfo(context, totalElements, dataType0, dataType1) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // 3. 获取工作空间大小
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    // 4. 初始化 tiling 数据
    PowerTilingData* tiling = context->GetTilingData<PowerTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(PowerTilingData), 0, sizeof(PowerTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // 5. 在函数内部计算广播信息
    
    {
        const gert::Shape x1ShapeObj = context->GetInputShape(0)->GetStorageShape();
        const gert::Shape x2ShapeObj = context->GetInputShape(1)->GetStorageShape();
        
        uint32_t dimNum1 = x1ShapeObj.GetDimNum();
        uint32_t dimNum2 = x2ShapeObj.GetDimNum();
        uint32_t dimMax = std::max(dimNum1, dimNum2);

        // 对齐后的形状数组
        uint32_t alignedX1[DIMS_LIMIT] = {1};
        uint32_t alignedX2[DIMS_LIMIT] = {1};
        uint32_t alignedY[DIMS_LIMIT] = {1};
        
        // 右对齐形状
        for (uint32_t i = 0; i < dimNum1; ++i)
            alignedX1[dimMax - dimNum1 + i] = static_cast<uint32_t>(x1ShapeObj.GetDim(i));
        for (uint32_t i = 0; i < dimNum2; ++i)
            alignedX2[dimMax - dimNum2 + i] = static_cast<uint32_t>(x2ShapeObj.GetDim(i));
        
        // 计算广播后的输出形状
        for (uint32_t i = 0; i < dimMax; ++i) {
            int idx1 = dimNum1 - 1 - i;
            int idx2 = dimNum2 - 1 - i;
            uint32_t s1 = (idx1 >= 0) ? static_cast<uint32_t>(x1ShapeObj.GetDim(idx1)) : 1;
            uint32_t s2 = (idx2 >= 0) ? static_cast<uint32_t>(x2ShapeObj.GetDim(idx2)) : 1;
            
            if (s1 != s2 && s1 != 1 && s2 != 1) {
                OP_LOGE(context, "Broadcast Fail, Please check your input shape");
                return ge::GRAPH_FAILED;
            }
            alignedY[dimMax - 1 - i] = (s1 > s2) ? s1 : s2;
        }
        
        // 计算步长
        uint32_t strideX1[DIMS_LIMIT] = {0};
        uint32_t strideX2[DIMS_LIMIT] = {0};
        uint32_t strideY[DIMS_LIMIT] = {0};
        
        // 输出步长
        strideY[dimMax - 1] = 1;
        for (int i = dimMax - 2; i >= 0; --i)
            strideY[i] = strideY[i + 1] * alignedY[i + 1];
        
        // 输入步长
        strideX1[dimMax - 1] = 1;
        strideX2[dimMax - 1] = 1;
        for (int i = dimMax - 2; i >= 0; --i) {
            strideX1[i] = strideX1[i + 1] * alignedX1[i + 1];
            strideX2[i] = strideX2[i + 1] * alignedX2[i + 1];
        }
        
        // 有效步长（考虑广播）
        uint32_t effStrideX1[DIMS_LIMIT] = {0};
        uint32_t effStrideX2[DIMS_LIMIT] = {0};
        for (uint32_t i = 0; i < dimMax; ++i) {
            effStrideX1[i] = (alignedX1[i] == 1) ? 0 : strideX1[i];
            effStrideX2[i] = (alignedX2[i] == 1) ? 0 : strideX2[i];
        }
        
        // 设置 tiling 数据
        tiling->x1Dim = dimNum1;
        tiling->x2Dim = dimNum2;
        tiling->yDim = dimMax;
        
        for (uint32_t i = 0; i < DIMS_LIMIT; ++i) {
            tiling->x1Shape[i] = alignedX1[i];
            tiling->x2Shape[i] = alignedX2[i];
            tiling->yShape[i] = alignedY[i];
            tiling->strideX1[i] = effStrideX1[i];
            tiling->strideX2[i] = effStrideX2[i];
            tiling->strideY[i] = strideY[i];
        }
        
        // 判断输入与输出形状是否一致
        bool isSameX1 = true;
        bool isSameX2 = true;
        for (uint32_t i = 0; i < dimMax; ++i) {
            if (alignedX1[i] != alignedY[i]) isSameX1 = false;
            if (alignedX2[i] != alignedY[i]) isSameX2 = false;
        }
        tiling->isSameX1 = isSameX1;
        tiling->isSameX2 = isSameX2;
    }



    // 6. 计算数据类型长度
    uint32_t typeLength0 = 0, typeLength1 = 0, outputTypeLength = 0;
    ge::TypeUtils::GetDataTypeLength(dataType0, typeLength0);
    ge::TypeUtils::GetDataTypeLength(dataType1, typeLength1);
    ge::TypeUtils::GetDataTypeLength(context->GetOutputDesc(0)->GetDataType(), outputTypeLength);
    
    uint32_t maxTypeLength = std::max(std::max(typeLength0, typeLength1), std::max(outputTypeLength, uint32_t(2)));
    uint64_t inputBytes = static_cast<uint64_t>(maxTypeLength);
    uint64_t inputLengthBytes = static_cast<uint64_t>(totalElements) * inputBytes;

    // 7. UB 划分策略
    uint32_t ubDataNumber = (inputBytes == 1ULL) ? 10U : 10U;
    uint64_t tmp = (ubSize / BLOCK_SIZE / BUFFER_NUM);
    uint32_t tileBlockNum = 1U;
    if (tmp > 0) {
        uint64_t tb = tmp / ubDataNumber;
        tileBlockNum = (tb == 0) ? 1U : static_cast<uint32_t>(tb);
    }

    // 每个 tile 包含的元素数
    uint32_t tileDataNum = static_cast<uint32_t>((static_cast<uint64_t>(tileBlockNum) * BLOCK_SIZE) / inputBytes);
    if (tileDataNum == 0U) tileDataNum = 1U;

    // 总 block 数
    uint64_t blocksTotal = (inputLengthBytes + BLOCK_SIZE - 1ULL) / BLOCK_SIZE;
    uint64_t coreNum64 = static_cast<uint64_t>(coreNum);
    if (coreNum64 > blocksTotal) coreNum64 = blocksTotal;
    if (coreNum64 == 0ULL) coreNum64 = 1ULL;
    uint32_t finalCoreNum = static_cast<uint32_t>(coreNum64);

    uint64_t everyCoreInputBlockNum = blocksTotal / coreNum64;
    uint32_t tailBlockNum = static_cast<uint32_t>(blocksTotal % coreNum64);

    // 小核计算
    uint64_t smallCoreDataNum_u = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint32_t smallCoreDataNum = static_cast<uint32_t>(smallCoreDataNum_u);
    uint32_t smallTileNum = static_cast<uint32_t>(everyCoreInputBlockNum / static_cast<uint64_t>(tileBlockNum));
    uint32_t finalSmallTileNum = ((everyCoreInputBlockNum % tileBlockNum) == 0) ? smallTileNum : (smallTileNum + 1);
    int64_t smallTailDataNum_i = static_cast<int64_t>(smallCoreDataNum) - static_cast<int64_t>(tileDataNum) * static_cast<int64_t>(smallTileNum);
    uint32_t smallTailDataNum = (smallTailDataNum_i <= 0) ? tileDataNum : static_cast<uint32_t>(smallTailDataNum_i);

    // 大核计算
    uint64_t bigEveryCoreBlockNum = everyCoreInputBlockNum + 1ULL;
    uint64_t bigCoreDataNum_u = bigEveryCoreBlockNum * BLOCK_SIZE / inputBytes;
    uint32_t bigCoreDataNum = static_cast<uint32_t>(bigCoreDataNum_u);
    uint32_t bigTileNum = static_cast<uint32_t>(bigEveryCoreBlockNum / static_cast<uint64_t>(tileBlockNum));
    uint32_t finalBigTileNum = ((bigEveryCoreBlockNum % tileBlockNum) == 0) ? bigTileNum : (bigTileNum + 1);
    int64_t bigTailDataNum_i = static_cast<int64_t>(bigCoreDataNum) - static_cast<int64_t>(tileDataNum) * static_cast<int64_t>(bigTileNum);
    uint32_t bigTailDataNum = (bigTailDataNum_i <= 0) ? tileDataNum : static_cast<uint32_t>(bigTailDataNum_i);

    // 8. 设置 tiling 字段
    tiling->smallCoreDataNum = static_cast<int64_t>(smallCoreDataNum);
    tiling->bigCoreDataNum = static_cast<int64_t>(bigCoreDataNum);
    tiling->tileDataNum = static_cast<int64_t>(tileDataNum);
    tiling->smallTailDataNum = static_cast<int64_t>(smallTailDataNum);
    tiling->bigTailDataNum = static_cast<int64_t>(bigTailDataNum);
    tiling->finalSmallTileNum = static_cast<int64_t>(finalSmallTileNum);
    tiling->finalBigTileNum = static_cast<int64_t>(finalBigTileNum);
    tiling->tailBlockNum = static_cast<int64_t>(tailBlockNum);

    // 9. 设置数据类型
    tiling->x_dtype = static_cast<int32_t>(dataType0);
    tiling->y_dtype = static_cast<int32_t>(dataType1);
    tiling->z_dtype = static_cast<int32_t>(context->GetOutputDesc(0)->GetDataType());

    // 10. 判断标量输入
    bool is_input0_scalar = (context->GetInputShape(0)->GetStorageShape().GetShapeSize() == 1);
    bool is_input1_scalar = (context->GetInputShape(1)->GetStorageShape().GetShapeSize() == 1);
    tiling->is_input0_scalar = is_input0_scalar;
    tiling->is_input1_scalar = is_input1_scalar;

    // 11. 设置块维度
    context->SetBlockDim(finalCoreNum);



    uint64_t tilingKey = 0;

    // 对应 TILING_KEY_INT8_INT8
    if (dataType0 == ge::DT_INT8 && dataType1 == ge::DT_INT8) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_INT8_UINT8
    else if (dataType0 == ge::DT_INT8 && dataType1 == ge::DT_UINT8) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_INT8_INT16
    else if (dataType0 == ge::DT_INT8 && dataType1 == ge::DT_INT16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_2);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_INT8_INT32
    else if (dataType0 == ge::DT_INT8 && dataType1 == ge::DT_INT32) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_3);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_INT8_FLOAT16
    else if (dataType0 == ge::DT_INT8 && dataType1 == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_4);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_INT8_BF16
    else if (dataType0 == ge::DT_INT8 && dataType1 == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_5);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_INT8_FLOAT32
    else if (dataType0 == ge::DT_INT8 && dataType1 == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_6);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_UINT8_INT8
    else if (dataType0 == ge::DT_UINT8 && dataType1 == ge::DT_INT8) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_7);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_UINT8_UINT8
    else if (dataType0 == ge::DT_UINT8 && dataType1 == ge::DT_UINT8) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_8);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_UINT8_INT16
    else if (dataType0 == ge::DT_UINT8 && dataType1 == ge::DT_INT16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_9);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_UINT8_INT32
    else if (dataType0 == ge::DT_UINT8 && dataType1 == ge::DT_INT32) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_10);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_UINT8_FLOAT16
    else if (dataType0 == ge::DT_UINT8 && dataType1 == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_11);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_UINT8_BF16
    else if (dataType0 == ge::DT_UINT8 && dataType1 == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_12);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_UINT8_FLOAT32
    else if (dataType0 == ge::DT_UINT8 && dataType1 == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_13);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_INT16_INT8
    else if (dataType0 == ge::DT_INT16 && dataType1 == ge::DT_INT8) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_14);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_INT16_UINT8
    else if (dataType0 == ge::DT_INT16 && dataType1 == ge::DT_UINT8) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_15);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_INT16_INT16
    else if (dataType0 == ge::DT_INT16 && dataType1 == ge::DT_INT16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_16);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_INT16_INT32
    else if (dataType0 == ge::DT_INT16 && dataType1 == ge::DT_INT32) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_17);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_INT16_FLOAT16
    else if (dataType0 == ge::DT_INT16 && dataType1 == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_18);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_INT16_BF16
    else if (dataType0 == ge::DT_INT16 && dataType1 == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_19);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_INT16_FLOAT32
    else if (dataType0 == ge::DT_INT16 && dataType1 == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_20);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_INT32_INT8
    else if (dataType0 == ge::DT_INT32 && dataType1 == ge::DT_INT8) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_21);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_INT32_UINT8
    else if (dataType0 == ge::DT_INT32 && dataType1 == ge::DT_UINT8) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_22);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_INT32_INT16
    else if (dataType0 == ge::DT_INT32 && dataType1 == ge::DT_INT16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_23);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_INT32_INT32
    else if (dataType0 == ge::DT_INT32 && dataType1 == ge::DT_INT32) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_24);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_INT32_FLOAT16
    else if (dataType0 == ge::DT_INT32 && dataType1 == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_25);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_INT32_BF16
    else if (dataType0 == ge::DT_INT32 && dataType1 == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_26);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_INT32_FLOAT32
    else if (dataType0 == ge::DT_INT32 && dataType1 == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_27);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_FLOAT16_INT8
    else if (dataType0 == ge::DT_FLOAT16 && dataType1 == ge::DT_INT8) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_28);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_FLOAT16_UINT8
    else if (dataType0 == ge::DT_FLOAT16 && dataType1 == ge::DT_UINT8) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_29);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_FLOAT16_INT16
    else if (dataType0 == ge::DT_FLOAT16 && dataType1 == ge::DT_INT16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_30);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_FLOAT16_INT32
    else if (dataType0 == ge::DT_FLOAT16 && dataType1 == ge::DT_INT32) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_31);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_FLOAT16_FLOAT16
    else if (dataType0 == ge::DT_FLOAT16 && dataType1 == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_32);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_FLOAT16_BF16
    else if (dataType0 == ge::DT_FLOAT16 && dataType1 == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_33);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_FLOAT16_FLOAT32
    else if (dataType0 == ge::DT_FLOAT16 && dataType1 == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_34);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_BF16_INT8
    else if (dataType0 == ge::DT_BF16 && dataType1 == ge::DT_INT8) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_35);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_BF16_UINT8
    else if (dataType0 == ge::DT_BF16 && dataType1 == ge::DT_UINT8) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_36);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_BF16_INT16
    else if (dataType0 == ge::DT_BF16 && dataType1 == ge::DT_INT16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_37);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_BF16_INT32
    else if (dataType0 == ge::DT_BF16 && dataType1 == ge::DT_INT32) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_38);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_BF16_FLOAT16
    else if (dataType0 == ge::DT_BF16 && dataType1 == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_39);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_BF16_BF16
    else if (dataType0 == ge::DT_BF16 && dataType1 == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_40);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_BF16_FLOAT32
    else if (dataType0 == ge::DT_BF16 && dataType1 == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_41);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_FLOAT32_INT8
    else if (dataType0 == ge::DT_FLOAT && dataType1 == ge::DT_INT8) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_42);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_FLOAT32_UINT8
    else if (dataType0 == ge::DT_FLOAT && dataType1 == ge::DT_UINT8) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_43);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_FLOAT32_INT16
    else if (dataType0 == ge::DT_FLOAT && dataType1 == ge::DT_INT16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_44);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_FLOAT32_INT32
    else if (dataType0 == ge::DT_FLOAT && dataType1 == ge::DT_INT32) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_45);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_FLOAT32_FLOAT16
    else if (dataType0 == ge::DT_FLOAT && dataType1 == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_46);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_FLOAT32_BF16
    else if (dataType0 == ge::DT_FLOAT && dataType1 == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_47);
        context->SetTilingKey(tilingKey);
    }
    // 对应 TILING_KEY_FLOAT32_FLOAT32
    else if (dataType0 == ge::DT_FLOAT && dataType1 == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_48);
        context->SetTilingKey(tilingKey);
    } else {
        OP_LOGE(context, "get dtype error");
        return ge::GRAPH_FAILED;
    }
    




    OP_LOGD(context, "End to do PowerTilingFunc");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForPower([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口
IMPL_OP_OPTILING(Power).Tiling(PowerTilingFunc).TilingParse<PowerCompileInfo>(TilingParseForPower);
} // namespace optiling