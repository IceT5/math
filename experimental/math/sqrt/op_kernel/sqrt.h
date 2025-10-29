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
 * \file sqrt.h
 * \brief
 */
#ifndef SQRT_H
#define SQRT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sqrt_tiling_data.h"
#include "sqrt_tiling_key.h"

namespace MySqrt {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_X, typename TYPE_Y>
class KernelSqrt {
public:
    __aicore__ inline KernelSqrt(){};

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, uint32_t smallCoreDataNum, uint32_t bigCoreDataNum, uint32_t finalBigTileNum,
        uint32_t finalSmallTileNum, uint32_t tileDataNum, uint32_t smallTailDataNum, uint32_t bigTailDataNum,
        uint32_t tailBlockNuma);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp1;
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelSqrt<TYPE_X, TYPE_Y>::Init(
    GM_ADDR x, GM_ADDR y, uint32_t smallCoreDataNum, uint32_t bigCoreDataNum, uint32_t finalBigTileNum,
    uint32_t finalSmallTileNum, uint32_t tileDataNum, uint32_t smallTailDataNum, uint32_t bigTailDataNum,
    uint32_t tailBlockNum)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint32_t coreId = AscendC::GetBlockIdx();
    uint32_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
    this->tileDataNum = tileDataNum;
    if (coreId < tailBlockNum) {
        this->coreDataNum = bigCoreDataNum;
        this->tileNum = finalBigTileNum;
        this->tailDataNum = bigTailDataNum;
    } else {
        this->coreDataNum = smallCoreDataNum;
        this->tileNum = finalSmallTileNum;
        this->tailDataNum = smallTailDataNum;
        globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - tailBlockNum);
    }
    xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
    yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->coreDataNum);
    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
    if constexpr (!std::is_same_v<TYPE_X, float32_t>) {
        pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(float));
    }
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelSqrt<TYPE_X, TYPE_Y>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
    AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
    inQueueX.EnQue(xLocal);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelSqrt<TYPE_X, TYPE_Y>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
    AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
    outQueueY.FreeTensor(yLocal);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelSqrt<TYPE_X, TYPE_Y>::Compute(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
    AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();
    if constexpr (!std::is_same_v<TYPE_X, float32_t>) {
        AscendC::LocalTensor<float> p1 = tmp1.Get<float>();
        AscendC::Cast(p1, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sqrt(p1, p1, this->processDataNum);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(yLocal, p1, AscendC::RoundMode::CAST_CEIL, this->processDataNum);
    } else {
        AscendC::Sqrt(yLocal, xLocal, this->processDataNum);
    }
    outQueueY.EnQue<TYPE_Y>(yLocal);
    inQueueX.FreeTensor(xLocal);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelSqrt<TYPE_X, TYPE_Y>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    for (int32_t i = 0; i < loopCount - 1; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
    this->processDataNum = this->tailDataNum;
    CopyIn(loopCount - 1);
    Compute(loopCount - 1);
    CopyOut(loopCount - 1);
}

} // namespace MySqrt
#endif // SQRT_H
