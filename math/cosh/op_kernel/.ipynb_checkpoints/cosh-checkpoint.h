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
 * \file cosh.h
 * \brief
 */
#ifndef COSH_H
#define COSH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "cosh_tiling_data.h"
#include "cosh_tiling_key.h"

namespace NsCosh {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Cosh {
public:
    __aicore__ inline Cosh(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, const CoshTilingData* tilingData);
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

    int64_t blockLength_ = 0;
    int64_t tileNum_ = 0;
    uint32_t tileLength_ = 0;
};

template <typename T>
__aicore__ inline void Cosh<T>::Init(GM_ADDR x, GM_ADDR z, const CoshTilingData* tilingData)
{
    blockLength_ = tilingData->totalLength / AscendC::GetBlockNum();
    tileNum_ = tilingData->tileNum;
    tileLength_ = blockLength_ / tileNum_ / BUFFER_NUM;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + blockLength_ * AscendC::GetBlockIdx(), blockLength_);
    outputGMZ.SetGlobalBuffer((__gm__ T*)z + blockLength_ * AscendC::GetBlockIdx(), blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, tileLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueZ, BUFFER_NUM, tileLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Cosh<T>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::DataCopy(xLocal, inputGMX[progress * tileLength_], tileLength_);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Cosh<T>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
    AscendC::DataCopy(outputGMZ[progress * tileLength_], zLocal, tileLength_);
    outputQueueZ.FreeTensor(zLocal);
}

template <typename T>
__aicore__ inline void Cosh<T>::Compute(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outputQueueZ.AllocTensor<T>();
    
    // T scalar = static_cast<T>(0.5);
    // cosh(x) = (e^x + e^(-x)) / 2
    // AscendC::Exp(xLocal, xLocal, tileLength_);              // xLocal = e^x
    // AscendC::DataCopy(zLocal, xLocal, tileLength_);         // zLocal = e^x（保存副本）
    // AscendC::Reciprocal(xLocal, zLocal, tileLength_);       // xLocal = e^(-x) = 1/e^x
    // AscendC::Add(zLocal, zLocal, xLocal, tileLength_);      // zLocal = e^x + e^(-x)
    // AscendC::Muls(zLocal, zLocal, scalar, tileLength_);     // zLocal = (e^x + e^(-x)) / 2
    
    AscendC::Cosh(zLocal, xLocal, tileLength_);
    outputQueueZ.EnQue<T>(zLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Cosh<T>::Process()
{
    int32_t loopCount = tileNum_ * BUFFER_NUM;
    for (int32_t i = 0; i < loopCount; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

} // namespace NsCosh
#endif // COSH_H