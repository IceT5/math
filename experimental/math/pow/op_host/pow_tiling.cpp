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
  * \file pow_tiling.cpp
  * \brief
 */
#include "log/log.h"
#include "util/math_util.h"
#include "tiling_base/tiling_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "../op_kernel/pow_tiling_data.h"
#include "../op_kernel/pow_tiling_key.h"

namespace optiling {

    using namespace Ops::Math::OpTiling;

    const uint32_t BLOCK_SIZE = 256;
    const uint32_t BUFFER_NUM = 2;
    const uint32_t WS_SYS_SIZE = 0;
    struct PowCompileInfo {};

    static ge::graphStatus TilingParseForPow([[maybe_unused]] gert::TilingParseContext* context)
    {
        return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
    {
        // 获取ubsize coreNum
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        coreNum = ascendcPlatform.GetCoreNum();
        auto socVersion = ascendcPlatform.GetSocVersion();
        OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
        if (socVersion != platform_ascendc::SocVersion::ASCEND910B && socVersion != platform_ascendc::SocVersion::ASCEND310B && context->GetInputDesc(0)->GetDataType() == ge::DT_BF16) {
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

    ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, uint64_t ubSize, uint64_t& inputNum, uint64_t& inputBytes, uint64_t& tileBlockNum, uint64_t& tileDataNum, uint64_t& inputLengthAlgin32)
    {
        inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
        uint32_t typeLength = 0;
        ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
        uint64_t inputLength = inputNum * typeLength;
        if (inputNum == 0) {
            return ge::GRAPH_FAILED;
        }
        inputBytes = inputLength / inputNum;
        auto dataType = context->GetInputDesc(0)->GetDataType();
        uint64_t ubDataNumber = 6;
        switch (dataType)
        {
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            ubDataNumber = 6;
            break;
        case ge::DT_FLOAT16:
        case ge::DT_INT16:
        case ge::DT_BF16:
            ubDataNumber = 12;
            break;
        case ge::DT_UINT8:
        case ge::DT_INT8:
            ubDataNumber = 25;
            break;
        default:
            break;
        }
        tileBlockNum = (ubSize / BUFFER_NUM / BLOCK_SIZE) / ubDataNumber;
        if (inputBytes == 0) {
            return ge::GRAPH_FAILED;
        }
        tileDataNum = (tileBlockNum * BLOCK_SIZE) / inputBytes;
        inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CalculateCoreBlockNums(
    uint64_t inputLengthAlgin32,
    int64_t coreNum,
    uint64_t tileBlockNum,
    uint64_t inputBytes,
    uint64_t tileDataNum,
    uint64_t& smallCoreDataNum,
    uint64_t& bigCoreDataNum,
    uint64_t& smallTailDataNum,
    uint64_t& bigTailDataNum,
    uint64_t& finalSmallTileNum,
    uint64_t& finalBigTileNum,
    uint64_t& tailBlockNum) {
    if(0 == BLOCK_SIZE || 0 == coreNum || 0 == tileBlockNum || 0 == inputBytes) {
        return ge::GRAPH_FAILED;
    }
    uint64_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;
    tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum;
    smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint64_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    everyCoreInputBlockNum += 1;
    bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint64_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus PowTilingFunc(gert::TilingContext* context)
    {
        // PowTilingData tiling;
        PowTilingData* tiling = context->GetTilingData<PowTilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
        OP_CHECK_IF(
            memset_s(tiling, sizeof(PowTilingData), 0, sizeof(PowTilingData)) != EOK,
            OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
        //获取平台运行信息
        uint64_t ubSize;
        int64_t coreNum;
        ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        //获取输入数据信息
        uint64_t inputNum, inputBytes, tileBlockNum, tileDataNum, inputLengthAlgin32;
        ret = GetShapeAttrsInfo(context, ubSize, inputNum, inputBytes, tileBlockNum, tileDataNum, inputLengthAlgin32);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }

        //计算coreNum
        if (tileDataNum >= inputNum) {
            coreNum = 1;
        }
        else {
            // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number of audits is the actual number of audits
            coreNum = (static_cast<uint64_t>(coreNum) < inputLengthAlgin32 / BLOCK_SIZE) ? coreNum : inputLengthAlgin32 / BLOCK_SIZE;
        }
        //计算每个core处理的数据块数
        uint64_t smallCoreDataNum, bigCoreDataNum, smallTailDataNum, bigTailDataNum;
        uint64_t finalSmallTileNum, finalBigTileNum, tailBlockNum;
        ret = CalculateCoreBlockNums(inputLengthAlgin32, coreNum, tileBlockNum, inputBytes,tileDataNum, smallCoreDataNum, bigCoreDataNum, smallTailDataNum, bigTailDataNum, finalSmallTileNum, finalBigTileNum, tailBlockNum);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        //设置tiling数据
        tiling->smallCoreDataNum = (uint32_t)smallCoreDataNum;
        tiling->bigCoreDataNum = (uint32_t)bigCoreDataNum;
        tiling->tileDataNum = (uint32_t)tileDataNum;
        tiling->smallTailDataNum = (uint32_t)smallTailDataNum;
        tiling->bigTailDataNum = (uint32_t)bigTailDataNum;
        tiling->finalSmallTileNum = (uint32_t)finalSmallTileNum;
        tiling->finalBigTileNum = (uint32_t)finalBigTileNum;
        tiling->tailBlockNum = (uint32_t)tailBlockNum;
        //计算workspace大小
        OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);
        context->SetBlockDim(coreNum);
        // 设置tilingKey.
        uint32_t tilingKey = 0;
        if (context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT)
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
    IMPL_OP_OPTILING(Pow).Tiling(PowTilingFunc).TilingParse<PowCompileInfo>(TilingParseForPow);
} // namespace optiling