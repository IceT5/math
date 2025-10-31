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
 * \file tan_v2.h
 * \brief
*/
#ifndef TANV2_H
#define TANV2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tan_v2_tiling_data.h"
#include "tan_v2_tiling_key.h"

namespace NsTanV2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class TanV2 {
public:
    __aicore__ inline TanV2(){};
    __aicore__ inline void Init(GM_ADDR x,  GM_ADDR z, const TanV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueZ;
    TBuf<AscendC::TPosition::VECCALC> tmp;
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMZ;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

template <typename T>
__aicore__ inline void TanV2<T>::Init(GM_ADDR x, GM_ADDR z, const TanV2TilingData* tilingData)
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
        outputGMZ.SetGlobalBuffer((__gm__ T*)z + globalBufferIndex, this->coreDataNum);
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(outputQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(tmp, this->tileDataNum * sizeof(T)); 

    }

template <typename T>
__aicore__ inline void TanV2<T>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::DataCopy(xLocal, inputGMX[progress * this->tileDataNum], this->processDataNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void TanV2<T>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
    AscendC::DataCopy(outputGMZ[progress * this->tileDataNum], zLocal, this->processDataNum);
    outputQueueZ.FreeTensor(zLocal);
}

template <typename T>
__aicore__ inline void TanV2<T>::Compute(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outputQueueZ.AllocTensor<T>();
    AscendC::LocalTensor<T> sharedTmpBuffer = tmp.Get<T>();

    AscendC::Tan(zLocal, xLocal,this->processDataNum);
    
    outputQueueZ.EnQue<T>(zLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void TanV2<T>::Process()
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
} // namespace NsTanV2
#endif // TanV2_H