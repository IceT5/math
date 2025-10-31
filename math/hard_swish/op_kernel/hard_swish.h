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
 * \file hard_swish.h
 * \brief
*/
#ifndef HARD_SWISH_H
#define HARD_SWISH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "hard_swish_tiling_data.h"
#include "hard_swish_tiling_key.h"

namespace NsHardSwish {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class HardSwish
{
public:
    __aicore__ inline HardSwish(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, const HardSwishTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueZ;
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMZ;

    TBuf<QuePosition::VECCALC> tmpBuf0;
    int64_t blockLength_ = 0;
    int64_t tileNum_ = 0;
    uint32_t tileLength_ = 0;
};

template <typename T>
__aicore__ inline void HardSwish<T>::Init(GM_ADDR x, GM_ADDR z, const HardSwishTilingData* tilingData)
{
    blockLength_ = tilingData->totalLength / AscendC::GetBlockNum();
    tileNum_ = tilingData->tileNum;
    tileLength_ = blockLength_ / tileNum_ / BUFFER_NUM;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + blockLength_ * AscendC::GetBlockIdx(), blockLength_);
    outputGMZ.SetGlobalBuffer((__gm__ T*)z + blockLength_ * AscendC::GetBlockIdx(), blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, tileLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueZ, BUFFER_NUM, tileLength_ * sizeof(T));
    pipe.InitBuffer(tmpBuf0, tileLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void HardSwish<T>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::DataCopy(xLocal, inputGMX[progress * tileLength_], tileLength_);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void HardSwish<T>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
    AscendC::DataCopy(outputGMZ[progress * tileLength_], zLocal, tileLength_);
    outputQueueZ.FreeTensor(zLocal);
}

template <typename T>
__aicore__ inline void HardSwish<T>::Compute(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outputQueueZ.AllocTensor<T>();
    AscendC::LocalTensor<T> tmp0 = tmpBuf0.Get<T>();
    AscendC::Adds(zLocal, xLocal, (T)3.0f,tileLength_);
    AscendC::Maxs(zLocal, zLocal, (T)0.0f, tileLength_);
    AscendC::Mins(zLocal, zLocal, (T)6.0f,tileLength_);
    AscendC::Duplicate<T>(tmp0,(T)6.0, tileLength_);
    AscendC::Div(zLocal, zLocal,tmp0,tileLength_);
    AscendC::Mul(zLocal,xLocal, zLocal,tileLength_);
    outputQueueZ.EnQue<T>(zLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void HardSwish<T>::Process()
{
    int32_t loopCount = tileNum_ * BUFFER_NUM;
    for (int32_t i = 0; i < loopCount; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

} // namespace NsHardSwish
#endif // HARD_SWISH_H