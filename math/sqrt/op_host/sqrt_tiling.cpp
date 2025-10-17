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
 * \file sqrt_tiling.cpp
 * \brief
*/
#include "log/log.h"
#include "util/math_util.h"
#include "tiling_base/tiling_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "../op_kernel/sqrt_tiling_data.h"
#include "../op_kernel/sqrt_tiling_key.h"

namespace optiling {

using namespace Ops::Math::OpTiling;

// const uint32_t BLOCK_DIM = 8;
// const int64_t TILE_NUM = 8;
// const uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;
// const int32_t DIMS_LIMIT = 4;
// constexpr int32_t ATTRPOS0 = 0;
// constexpr uint32_t INDEXZERO = 0;
// constexpr uint32_t INDEXONE = 1;
// constexpr uint32_t INDEXTWO = 2;
// constexpr uint32_t INDEXTHREE = 3;



// 获取平台信息如ubSize, coreNum
// static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
// {
//     // 获取ubsize coreNum
//     fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
//     OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
//     auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
//     coreNum = ascendcPlatform.GetCoreNumAiv();
//     OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
//     ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
//     OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
//     return ge::GRAPH_SUCCESS;
// }

// // 获取属性，shape信息
// ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalIdx, ge::DataType& dataType)
// {
//     // 获取输入shape信息
//     auto inputX = context->GetInputShape(0);
//     OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
//     // 如果输入shape 是标量 转换为{1}，否则保持原 shape 不变
//     auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());
//     auto inputY = context->GetInputShape(1);
//     OP_CHECK_NULL_WITH_CONTEXT(context, inputY);
//     auto inputShapeY = EnsureNotScalar(inputY->GetStorageShape());
//     auto outZ = context->GetOutputShape(0);
//     OP_CHECK_NULL_WITH_CONTEXT(context, outZ);
//     auto outShapeZ = EnsureNotScalar(outZ->GetStorageShape());

//     // shape校验
//     OP_CHECK_IF(
//         inputShapeX.GetDimNum() != DIMS_LIMIT || inputShapeY.GetDimNum() != DIMS_LIMIT ||
//             outShapeZ.GetDimNum() != DIMS_LIMIT,
//         OP_LOGE(
//             context, "AddExample: inputx,inputy,outputz shape dim = %zu, %zu, %zu, should be equal 4",
//             inputShapeX.GetDimNum(), inputShapeY.GetDimNum(), outShapeZ.GetDimNum()),
//         return ge::GRAPH_FAILED);

//     // 获取shape dim值
//     auto nDim = inputShapeX.GetDim(INDEXZERO);
//     auto cDim = inputShapeX.GetDim(INDEXONE);
//     auto hDim = inputShapeX.GetDim(INDEXTWO);
//     auto wDim = inputShapeX.GetDim(INDEXTHREE);
//     totalIdx = nDim * cDim * hDim * wDim;
//     // dtype校验
//     const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_INT32};
//     auto inputDesc = context->GetInputDesc(0);
//     OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
//     dataType = inputDesc->GetDataType();
//     if (supportedDtype.count(dataType) == 0) {
//         OP_LOGE(context, "invalid dtype");
//         return ge::GRAPH_FAILED;
//     }
//     return ge::GRAPH_SUCCESS;
// }

// ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
// {
//     size_t* currentWorkspace = context->GetWorkspaceSizes(1);
//     OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
//     currentWorkspace[0] = WS_SYS_SIZE;
//     return ge::GRAPH_SUCCESS;
// }

// // tiling 分发入口
// static ge::graphStatus AddExampleTilingFunc(gert::TilingContext* context)
// {
//     // 1、获取平台运行信息
//     uint64_t ubSize;
//     int64_t coreNum;
//     OP_CHECK_IF(
//         GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
//         return ge::GRAPH_FAILED);
//     // 2、获取shape、属性信息
//     int64_t totalIdx;
//     ge::DataType dataType;

//     OP_CHECK_IF(
//         GetShapeAttrsInfo(context, totalIdx, dataType) != ge::GRAPH_SUCCESS,
//         OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);
//     // 3、获取WorkspaceSize信息
//     OP_CHECK_IF(
//         GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
//         return ge::GRAPH_FAILED);

//     // 4、设置tiling信息
//     AddExampleTilingData* tiling = context->GetTilingData<AddExampleTilingData>();
//     OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
//     OP_CHECK_IF(
//         memset_s(tiling, sizeof(AddExampleTilingData), 0, sizeof(AddExampleTilingData)) != EOK,
//         OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
//     tiling->totalLength = totalIdx;
//     tiling->tileNum = TILE_NUM;

//     context->SetBlockDim(BLOCK_DIM);
//     uint64_t tilingKey = 0;
//     // 区分dtype走不同得tiling key分支.
//     if (dataType == ge::DT_FLOAT) {
//         tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
//         context->SetTilingKey(tilingKey);
//     } else if (dataType == ge::DT_INT32) {
//         tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
//         context->SetTilingKey(tilingKey);
//     } else {
//         OP_LOGE(context, "get dtype error");
//         return ge::GRAPH_FAILED;
//     }
//     return ge::GRAPH_SUCCESS;
// }
const uint32_t BLOCK_SIZE = 32;
const uint32_t BUFFER_NUM = 2;

static ge::graphStatus TilingParseForSqrt([[maybe_unused]]gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SqrtTilingFunc(gert::TilingContext* context)
{
    // SqrtTilingData tiling;
    SqrtTilingData* tiling = context->GetTilingData<SqrtTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(SqrtTilingData), 0, sizeof(SqrtTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    uint64_t ubSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    auto coreNum = ascendcPlatform.GetCoreNum();
    auto socVersion = ascendcPlatform.GetSocVersion();
    if (socVersion != platform_ascendc::SocVersion::ASCEND910B && socVersion != platform_ascendc::SocVersion::ASCEND310B && context->GetInputDesc(0)->GetDataType() == ge::DT_BF16) {
        return ge::GRAPH_FAILED;
    }

    uint64_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    uint64_t inputLength = inputNum * typeLength;
    uint64_t inputBytes = inputLength / inputNum;

    uint64_t ubDataNumber = (context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT) ? 4 : 6;
    uint64_t tileBlockNum = (ubSize / BLOCK_SIZE ) / ubDataNumber;
    uint64_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / inputBytes;

    uint64_t inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);

    if(tileDataNum >= inputNum)
    {
        coreNum=1;
    }
    else
    {
        // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number of audits is the actual number of audits
        coreNum = (coreNum <  inputLengthAlgin32 / BLOCK_SIZE) ? coreNum : inputLengthAlgin32 / BLOCK_SIZE;
    }

    uint64_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;
    uint64_t tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum;
    
    uint64_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint64_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint64_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    uint64_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;
    
    everyCoreInputBlockNum += 1;
    uint64_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint64_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint64_t finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    uint64_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum; 
    
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    // tiling.set_smallCoreDataNum((uint32_t)smallCoreDataNum);
    // tiling.set_bigCoreDataNum((uint32_t)bigCoreDataNum);
    // tiling.set_tileDataNum((uint32_t)tileDataNum);
    // tiling.set_smallTailDataNum((uint32_t)smallTailDataNum);
    // tiling.set_bigTailDataNum((uint32_t)bigTailDataNum);
    // tiling.set_finalSmallTileNum((uint32_t)finalSmallTileNum);
    // tiling.set_finalBigTileNum((uint32_t)finalBigTileNum);
    // tiling.set_tailBlockNum((uint32_t)tailBlockNum);
    tiling->smallCoreDataNum = (uint32_t)smallCoreDataNum;
    tiling->bigCoreDataNum = (uint32_t)bigCoreDataNum;
    tiling->tileDataNum = (uint32_t)tileDataNum;
    tiling->smallTailDataNum = (uint32_t)smallTailDataNum;
    tiling->bigTailDataNum = (uint32_t)bigTailDataNum;
    tiling->finalSmallTileNum = (uint32_t)finalSmallTileNum;
    tiling->finalBigTileNum = (uint32_t)finalBigTileNum;
    tiling->tailBlockNum = (uint32_t)tailBlockNum;
    context->SetBlockDim(coreNum);
    // tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    // context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    uint32_t tilingKey = 0;
    // 区分dtype走不同得tiling key分支.
    tilingKey = GET_TPL_TILING_KEY(1);
    std::cout << "tilingKey: " << tilingKey << std::endl;
    context->SetTilingKey(tilingKey);
    

    return ge::GRAPH_SUCCESS;
}
struct SqrtCompileInfo {};
// tiling注册入口.
IMPL_OP_OPTILING(Sqrt).Tiling(SqrtTilingFunc).TilingParse<SqrtCompileInfo>(TilingParseForSqrt);
} // namespace optiling
