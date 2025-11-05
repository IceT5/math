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

constexpr int32_t BUFFER_NUM = 2;

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
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp0, tmp1, tmp2, tmp3, tmp4, tmp01, tmp11, tmp21, tmp31, tmp41;
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
        pipe.InitBuffer(tmp0, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmp3, this->tileDataNum * sizeof(half));
        pipe.InitBuffer(tmp4, this->tileDataNum * sizeof(half));
        pipe.InitBuffer(tmp01, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmp11, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmp21, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmp31, this->tileDataNum * sizeof(half));
        pipe.InitBuffer(tmp41, this->tileDataNum * sizeof(half));
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
    AscendC::DumpTensor(xLocal, 121, 10);
    AscendC::LocalTensor<float> xFloat;
    AscendC::LocalTensor<float> yFloat;
    AscendC::LocalTensor<float> zFloat;
    if(progress & 1) {
        xFloat = tmp0.Get<float>();
        yFloat = tmp1.Get<float>();
        zFloat = tmp2.Get<float>();
    } else {
        xFloat = tmp01.Get<float>();
        yFloat = tmp11.Get<float>();
        zFloat = tmp21.Get<float>();
    }

    if constexpr (std::is_same_v<T, float>) {
        AscendC::Div(yLocal, xLocal, yLocal, this->processDataNum);
        AscendC::Floor(zLocal, yLocal, this->processDataNum);
    } else {
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            AscendC::LocalTensor<half> xHalf, yHalf;
            if(progress & 1){
                xHalf = tmp3.Get<half>();
                yHalf = tmp4.Get<half>();
            } else {
                xHalf = tmp31.Get<half>();
                yHalf = tmp41.Get<half>();
            }
            AscendC::Cast(xHalf, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Cast(yHalf, yLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Cast(xFloat, xHalf, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Cast(yFloat, yHalf, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        } else {
            AscendC::Cast(xFloat, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Cast(yFloat, yLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        }
        AscendC::Div(zFloat, xFloat, yFloat, this->processDataNum);
        AscendC::Floor(xFloat, zFloat, this->processDataNum);
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            AscendC::LocalTensor<half> zHalf;
            if(progress & 1){
                zHalf = tmp3.Get<half>();
            } else {
                zHalf = tmp31.Get<half>();
            }
            AscendC::Cast(zHalf, xFloat, AscendC::RoundMode::CAST_FLOOR, this->processDataNum);
            AscendC::Cast(zLocal, zHalf, AscendC::RoundMode::CAST_FLOOR, this->processDataNum);
        } else {
            AscendC::Cast(zLocal, xFloat, AscendC::RoundMode::CAST_FLOOR, this->processDataNum);
        }
    }
    outputQueueZ.EnQue(zLocal);
    inputQueueX.FreeTensor(xLocal);
    inputQueueY.FreeTensor(yLocal);
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
