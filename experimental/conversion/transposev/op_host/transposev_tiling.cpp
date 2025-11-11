/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Liang Yanglin <@liang-yanglin>
 * - Liu Jun <@kbryantttt>
 * - Zhou Jianhua <@LePenseur>
 * - Tu Yuanhang <@TuYHAAAAAA>
 * - Li Xing <@li-xingHIT>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file transposev_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "tiling_base/tiling_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "../op_kernel/transposev_tiling_data.h"
#include "../op_kernel/transposev_tiling_key.h"

namespace optiling {

using namespace Ops::Math::OpTiling;
const uint32_t BLOCK_SIZE = 32;
const uint32_t BUFFER_NUM = 2;
int64_t dim = 8;

const uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;
const int32_t DIMS_LIMIT = 4;

constexpr uint32_t INDEXZERO = 0;
constexpr uint32_t INDEXONE = 1;
constexpr uint32_t INDEXTWO = 2;
constexpr uint32_t INDEXTHREE = 3;
constexpr uint32_t RESERVED_BYTES = 512U;
struct TransposevCompileInfo {};

// 获取平台信息如ubSize, coreNum
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    // 获取ubSize coreNum
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    //coreNum = 1;
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 获取属性，shape信息
ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalIdx, ge::DataType& dataType)
{
    
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    totalIdx = inputX->GetStorageShape().GetShapeSize();
    
    // dtype校验
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_INT32, ge::DT_UINT32, ge::DT_INT64,         // int32对应12种数据类型
                ge::DT_UINT64,ge::DT_FLOAT16, ge::DT_INT16, ge::DT_UINT16, 
                ge::DT_BF16,ge::DT_INT8, ge::DT_UINT8, ge::DT_BOOL};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "invalid dtype");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
static ge::graphStatus TransposevTilingFunc(gert::TilingContext* context)
{
    // 1. platform
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    ubSize = ubSize - RESERVED_BYTES;
    // 2. shapes & dtype
    int64_t totalIdx = 0;
    ge::DataType dataType;
    OP_CHECK_IF(GetShapeAttrsInfo(context, totalIdx, dataType) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);
    
    // 获取x2的数据类型
    auto inputDesc2 = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc2);
    ge::DataType permType = inputDesc2->GetDataType();
    OP_CHECK_IF(permType != ge::DT_INT32 && permType != ge::DT_INT64,
                OP_LOGE(context, "Transposev only supports int32/int64 perm dtype, got %d", permType),
                return ge::GRAPH_FAILED);
    const bool isPermInt64 = (permType == ge::DT_INT64); 

    
    // 3. workspace
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    TransposevTilingData* tiling = context->GetTilingData<TransposevTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(TransposevTilingData), 0, sizeof(TransposevTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // --- safer numeric types ---
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    if (typeLength == 0) {
        OP_LOGE(context, "typeLength is 0");
        return ge::GRAPH_FAILED;
    }
    uint64_t inputBytes = static_cast<uint64_t>(typeLength);
    uint64_t inputLengthBytes = static_cast<uint64_t>(totalIdx) * inputBytes;

    // ub-based tileBlockNum guard (避免为0)
    uint32_t ubDataNumber = 0;
    switch (inputBytes) {
        case 1:  ubDataNumber = 23U; break;
        case 2:  ubDataNumber = 13U; break;
        case 4:  ubDataNumber = 7U;  break;
        case 8:  ubDataNumber = 4U;  break;
        default: ubDataNumber = 23U;  break;   // 未知类型按 1 B 处理
    }
    uint64_t tmp = (ubSize / BLOCK_SIZE / BUFFER_NUM);
    uint32_t tileBlockNum = 1U;
    if (tmp > 0) {
        uint64_t tb = tmp / ubDataNumber;
        tileBlockNum = (tb == 0) ? 1U : static_cast<uint32_t>(tb);
    }

    // 每个 tile 包含的元素数（至少 1）
    uint32_t tileDataNum = static_cast<uint32_t>((static_cast<uint64_t>(tileBlockNum) * BLOCK_SIZE) / inputBytes);
    if (tileDataNum == 0U) tileDataNum = 1U;

    // 总 block 数（向上取整）
    uint64_t blocksTotal = (inputLengthBytes + BLOCK_SIZE - 1ULL) / BLOCK_SIZE;
    uint64_t coreNum64 = static_cast<uint64_t>(coreNum);
    if (coreNum64 > blocksTotal) coreNum64 = blocksTotal;
    if (coreNum64 == 0ULL) coreNum64 = 1ULL; // 最少 1 core
    uint32_t finalCoreNum = static_cast<uint32_t>(coreNum64);

    uint64_t everyCoreInputBlockNum = blocksTotal / coreNum64; // 基本块数
    uint32_t tailBlockNum = static_cast<uint32_t>(blocksTotal % coreNum64); // 前 tailBlockNum 个核是 big-core

    // small-core 数量（元素）
    uint64_t smallCoreDataNum_u = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint32_t smallCoreDataNum = static_cast<uint32_t>(smallCoreDataNum_u);

    uint32_t smallTileNum = static_cast<uint32_t>(everyCoreInputBlockNum / static_cast<uint64_t>(tileBlockNum));
    uint32_t finalSmallTileNum = ((everyCoreInputBlockNum % tileBlockNum) == 0) ? smallTileNum : (smallTileNum + 1);
    int64_t smallTailDataNum_i = static_cast<int64_t>(smallCoreDataNum) - static_cast<int64_t>(tileDataNum) * static_cast<int64_t>(smallTileNum);
    uint32_t smallTailDataNum = (smallTailDataNum_i <= 0) ? tileDataNum : static_cast<uint32_t>(smallTailDataNum_i);

    // big-core（每个多一个 block）
    uint64_t bigEveryCoreBlockNum = everyCoreInputBlockNum + 1ULL;
    uint64_t bigCoreDataNum_u = bigEveryCoreBlockNum * BLOCK_SIZE / inputBytes;
    uint32_t bigCoreDataNum = static_cast<uint32_t>(bigCoreDataNum_u);
    uint32_t bigTileNum = static_cast<uint32_t>(bigEveryCoreBlockNum / static_cast<uint64_t>(tileBlockNum));
    uint32_t finalBigTileNum = ((bigEveryCoreBlockNum % tileBlockNum) == 0) ? bigTileNum : (bigTileNum + 1);
    int64_t bigTailDataNum_i = static_cast<int64_t>(bigCoreDataNum) - static_cast<int64_t>(tileDataNum) * static_cast<int64_t>(bigTileNum);
    uint32_t bigTailDataNum = (bigTailDataNum_i <= 0) ? tileDataNum : static_cast<uint32_t>(bigTailDataNum_i);

    // write back
    tiling->smallCoreDataNum = static_cast<int64_t>(smallCoreDataNum);
    tiling->bigCoreDataNum = static_cast<int64_t>(bigCoreDataNum);
    tiling->tileDataNum = static_cast<int64_t>(tileDataNum);
    tiling->smallTailDataNum = static_cast<int64_t>(smallTailDataNum);
    tiling->bigTailDataNum = static_cast<int64_t>(bigTailDataNum);
    tiling->finalSmallTileNum = static_cast<int64_t>(finalSmallTileNum);
    tiling->finalBigTileNum = static_cast<int64_t>(finalBigTileNum);
    tiling->tailBlockNum = static_cast<int64_t>(tailBlockNum);
    tiling->totalnumber = static_cast<int64_t>(totalIdx);

    const auto xShape = context->GetInputTensor(0)->GetOriginShape();  
    dim = static_cast<int64_t>(xShape.GetDimNum());   
    tiling->dims = static_cast<int64_t>(dim);
    for(int32_t i=0;i<8;i++){                                          
        tiling->shape[i] = static_cast<int64_t>(0);
    }
    for(int32_t i=0;i<dim;i++){                                           
        tiling->shape[i] = static_cast<int64_t>(xShape.GetDim(i));
    }
    
    context->SetBlockDim(finalCoreNum);

    uint64_t tilingKey = 0;
    bool tilingKeyFound = false;
    if(!isPermInt64){ // x2为int32
        if (dataType == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
        context->SetTilingKey(tilingKey);
        tilingKeyFound = true;
        } else if (dataType == ge::DT_INT32) {
            tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
            context->SetTilingKey(tilingKey);
            tilingKeyFound = true;
        } else if (dataType == ge::DT_UINT32) {
            tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_2);
            context->SetTilingKey(tilingKey);
            tilingKeyFound = true;
        } else if (dataType == ge::DT_INT64) {
            tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_3);
            context->SetTilingKey(tilingKey);
            tilingKeyFound = true;
        } else if (dataType == ge::DT_UINT64) {
            tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_4);
            context->SetTilingKey(tilingKey);
            tilingKeyFound = true;
        } else if (dataType == ge::DT_FLOAT16) {
            tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_5);
            context->SetTilingKey(tilingKey);
            tilingKeyFound = true;
        } else if (dataType == ge::DT_INT16) {
            tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_6);
            context->SetTilingKey(tilingKey);
            tilingKeyFound = true;
        }  else if (dataType == ge::DT_UINT16) {
            tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_7);
            context->SetTilingKey(tilingKey);
            tilingKeyFound = true;
        } else if (dataType == ge::DT_BF16) {
            tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_8);
            context->SetTilingKey(tilingKey);
            tilingKeyFound = true;
        } else if (dataType == ge::DT_INT8) {
            tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_9);
            context->SetTilingKey(tilingKey);
            tilingKeyFound = true;
        } else if (dataType == ge::DT_UINT8) {
            tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_10);
            context->SetTilingKey(tilingKey);
            tilingKeyFound = true;
        } else if (dataType == ge::DT_BOOL) {
            tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_11);
            context->SetTilingKey(tilingKey);
            tilingKeyFound = true;
        }else {
            tilingKeyFound = false;
        }
    }
    else{ // x2为int64
        if (dataType == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_12);
        context->SetTilingKey(tilingKey);
        tilingKeyFound = true;
        } else if (dataType == ge::DT_INT32) {
            tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_13);
            context->SetTilingKey(tilingKey);
            tilingKeyFound = true;
        } else if (dataType == ge::DT_UINT32) {
            tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_14);
            context->SetTilingKey(tilingKey);
            tilingKeyFound = true;
        } else if (dataType == ge::DT_INT64) {
            tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_15);
            context->SetTilingKey(tilingKey);
            tilingKeyFound = true;
        } else if (dataType == ge::DT_UINT64) {
            tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_16);
            context->SetTilingKey(tilingKey);
            tilingKeyFound = true;
        } else if (dataType == ge::DT_FLOAT16) {
            tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_17);
            context->SetTilingKey(tilingKey);
            tilingKeyFound = true;
        } else if (dataType == ge::DT_INT16) {
            tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_18);
            context->SetTilingKey(tilingKey);
            tilingKeyFound = true;
        }  else if (dataType == ge::DT_UINT16) {
            tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_19);
            context->SetTilingKey(tilingKey);
            tilingKeyFound = true;
        } else if (dataType == ge::DT_BF16) {
            tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_20);
            context->SetTilingKey(tilingKey);
            tilingKeyFound = true;
        } else if (dataType == ge::DT_INT8) {
            tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_21);
            context->SetTilingKey(tilingKey);
            tilingKeyFound = true;
        } else if (dataType == ge::DT_UINT8) {
            tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_22);
            context->SetTilingKey(tilingKey);
            tilingKeyFound = true;
        } else if (dataType == ge::DT_BOOL) {
            tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_23);
            context->SetTilingKey(tilingKey);
            tilingKeyFound = true;
        }else {
            tilingKeyFound = false;
        }
    }
    OP_CHECK_IF(!tilingKeyFound,
                OP_LOGE(context, "Unsupported dtype combo x:%d perm:%d", dataType, permType),
                return ge::GRAPH_FAILED);
    context->SetTilingKey(tilingKey); 
    return ge::GRAPH_SUCCESS;
}


static ge::graphStatus TilingParseForTransposev([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(Transposev).Tiling(TransposevTilingFunc).TilingParse<TransposevCompileInfo>(TilingParseForTransposev);
} // namespace optiling