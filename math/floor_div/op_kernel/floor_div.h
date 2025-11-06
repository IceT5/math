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
 * \file floor_div.h
 * \brief
 * */
#ifndef FLOOR_DIV_H
#define FLOOR_DIV_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "floor_div_tiling_data.h"
#include "floor_div_tiling_key.h"

namespace NsFloorDiv {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

template <typename T>
class FloorDiv {
public:
    __aicore__ inline FloorDiv(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const FloorDivTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueY;
    TQue<QuePosition::TSCM, 1> tmp0, tmp1, tmp01, tmp11;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueZ;
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> inputGMY;
    GlobalTensor<T> outputGMZ;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

template <typename T>
__aicore__ inline void FloorDiv<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const FloorDivTilingData* tilingData)
{
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t coreNum = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = tilingData->bigCoreDataNum * AscendC::GetBlockIdx();
        this->tileDataNum = tilingData->tileDataNum;
        if (coreNum < tilingData->tailBlockNum) { 
          this->coreDataNum = tilingData->bigCoreDataNum;
          this->tileNum = tilingData->finalBigTileNum;
          this->tailDataNum = tilingData->bigTailDataNum;
        }
        else { 
          this->coreDataNum = tilingData->smallCoreDataNum;
          this->tileNum = tilingData->finalSmallTileNum;
          this->tailDataNum = tilingData->smallTailDataNum;
          globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) * (AscendC::GetBlockIdx() - tilingData->tailBlockNum);
        }
        inputGMX.SetGlobalBuffer((__gm__ T*)x + globalBufferIndex, this->coreDataNum);
        inputGMY.SetGlobalBuffer((__gm__ T*)y + globalBufferIndex, this->coreDataNum);
        outputGMZ.SetGlobalBuffer((__gm__ T*)z + globalBufferIndex, this->coreDataNum);
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(inputQueueY, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(outputQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(tmp0, 1, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmp1, 1, this->tileDataNum * sizeof(half));
        pipe.InitBuffer(tmp01, 1, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmp11, 1, this->tileDataNum * sizeof(half));
    }

template <typename T>
__aicore__ inline void FloorDiv<T>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.AllocTensor<T>();
    AscendC::DataCopy(xLocal, inputGMX[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(yLocal, inputGMY[progress * this->tileDataNum], this->processDataNum);
    inputQueueX.EnQue(xLocal);
    inputQueueY.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void FloorDiv<T>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
    AscendC::DataCopy(outputGMZ[progress * this->tileDataNum], zLocal, this->processDataNum);
    outputQueueZ.FreeTensor(zLocal);
}

template <typename T>
__aicore__ inline void FloorDiv<T>::Compute(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outputQueueZ.AllocTensor<T>();
    AscendC::LocalTensor<float> xFloat = tmp0.AllocTensor<float>();
    AscendC::LocalTensor<float> yFloat = tmp01.AllocTensor<float>();
    AscendC::LocalTensor<half> xHalf = tmp1.AllocTensor<half>();
    AscendC::LocalTensor<half> yHalf = tmp11.AllocTensor<half>();
    AscendC::DumpTensor(xLocal, 119, 10);
    AscendC::DumpTensor(yLocal, 120, 10);
    if constexpr (std::is_same_v<T, float>) {
        AscendC::Div(yLocal, xLocal, yLocal, this->processDataNum);
        AscendC::Floor(zLocal, yLocal, this->processDataNum);
    } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
        AscendC::Cast(xHalf, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
         AscendC::Cast(yHalf, yLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::DumpTensor(xHalf, 138, 10);
        AscendC::DumpTensor(yHalf, 139, 10);
                        // AscendC::Div(xHalf, xHalf, yHalf, this->processDataNum);
        // AscendC::Floor(yHalf, xHalf, this->processDataNum);
        // AscendC::Cast(zLocal, yHalf, AscendC::RoundMode::CAST_FLOOR, this->processDataNum);
    } else {
        AscendC::Cast(xFloat, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(yFloat, yLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::DumpTensor(xFloat, 139, 10);
        AscendC::DumpTensor(yFloat, 140, 10);
        AscendC::Div(yFloat, xFloat, yFloat, this->processDataNum);
        AscendC::DumpTensor(yFloat, 142, 10);
        AscendC::Floor(xFloat, yFloat, this->processDataNum);
        AscendC::DumpTensor(xFloat, 144, 10);
        AscendC::Cast(zLocal, xFloat, AscendC::RoundMode::CAST_FLOOR, this->processDataNum);
    }
    outputQueueZ.EnQue(zLocal);
    inputQueueX.FreeTensor(xLocal);
    inputQueueY.FreeTensor(yLocal);
    tmp0.FreeTensor(xFloat);
    tmp01.FreeTensor(yFloat);
    tmp1.FreeTensor(xHalf);
    tmp11.FreeTensor(yHalf);
}

template <typename T>
__aicore__ inline void FloorDiv<T>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    for (int32_t i = 0; i < loopCount; i++) {
        if (i == this->tileNum - 1) {
            this->processDataNum = this->tailDataNum;
        }
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

} // namespace NsFloorDiv
#endif // FloorDiv_H
