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

const uint32_t BLOCK_SIZE = 32;
const uint32_t BUFFER_NUM = 2;
struct SqrtCompileInfo {};

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
    //获取平台运行信息
    uint64_t ubSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    auto coreNum = ascendcPlatform.GetCoreNum();
    auto socVersion = ascendcPlatform.GetSocVersion();
    if (socVersion != platform_ascendc::SocVersion::ASCEND910B && socVersion != platform_ascendc::SocVersion::ASCEND310B && context->GetInputDesc(0)->GetDataType() == ge::DT_BF16) {
        return ge::GRAPH_FAILED;
    }

    //获取输入数据信息
    uint64_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    uint64_t inputLength = inputNum * typeLength;
    uint64_t inputBytes = inputLength / inputNum;

    uint64_t ubDataNumber = (context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT) ? 4 : 6;
    uint64_t tileBlockNum = (ubSize / BLOCK_SIZE ) / ubDataNumber;
    uint64_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / inputBytes;

    uint64_t inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);

    //计算coreNum
    if(tileDataNum >= inputNum)
    {
        coreNum=1;
    }
    else
    {
        // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number of audits is the actual number of audits
        coreNum = (coreNum <  inputLengthAlgin32 / BLOCK_SIZE) ? coreNum : inputLengthAlgin32 / BLOCK_SIZE;
    }

    //计算每个core处理的数据块数
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
    
    //计算workspace大小
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    //设置tiling数据
    tiling->smallCoreDataNum = (uint32_t)smallCoreDataNum;
    tiling->bigCoreDataNum = (uint32_t)bigCoreDataNum;
    tiling->tileDataNum = (uint32_t)tileDataNum;
    tiling->smallTailDataNum = (uint32_t)smallTailDataNum;
    tiling->bigTailDataNum = (uint32_t)bigTailDataNum;
    tiling->finalSmallTileNum = (uint32_t)finalSmallTileNum;
    tiling->finalBigTileNum = (uint32_t)finalBigTileNum;
    tiling->tailBlockNum = (uint32_t)tailBlockNum;
    context->SetBlockDim(coreNum);
    // 区分dtype走不同得tiling key分支.
    uint32_t tilingKey = 0;
    if(context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT)
    {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
    }
    else {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(Sqrt).Tiling(SqrtTilingFunc).TilingParse<SqrtCompileInfo>(TilingParseForSqrt);
} // namespace optiling
