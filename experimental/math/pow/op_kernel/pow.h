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
 * \file pow.h
 * \brief
*/
#ifndef POW_H
#define POW_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "pow_tiling_data.h"
#include "pow_tiling_key.h"

namespace MyPow {

/*
x > 0: use exp(y*ln(x))
    x < 0: use [-2*(|y|%2)+1]*exp(y*ln|x|)
    x = 0: 0^0=1 & 0^y=0
*/

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_X, typename TYPE_Y>
class KernelPow {
public:
    __aicore__ inline KernelPow(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR exponent, GM_ADDR y, uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum, uint32_t tileDataNum,
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum,
                                uint32_t tailBlockNuma);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);
    __aicore__ inline void PowCompute(LocalTensor<float> &xLocal, LocalTensor<float> &expLocal, LocalTensor<float> &yLocal, LocalTensor<uint8_t> &mask);
    __aicore__ inline void PowComputeUint8(LocalTensor<float> &xLocal, LocalTensor<float> &expLocal, LocalTensor<float> &yLocal, LocalTensor<uint8_t> &mask);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> absTmp, xCastTmp, expCastTmp, yCastTmp, tranCastTmp, maskTmp, zeroTmp;
    GlobalTensor<TYPE_X> xGm;
    GlobalTensor<TYPE_X> expGm;
    GlobalTensor<TYPE_Y> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelPow<TYPE_X, TYPE_Y>::Init(GM_ADDR x, GM_ADDR exponent, GM_ADDR y, uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum, uint32_t tileDataNum,
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum,
                                uint32_t tailBlockNum)
{
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    uint32_t coreNum = GetBlockIdx();
    uint32_t globalBufferIndex = bigCoreDataNum * GetBlockIdx();
    this->tileDataNum = tileDataNum;
    if (coreNum < tailBlockNum)
    {
        this->coreDataNum = bigCoreDataNum;
        this->tileNum = finalBigTileNum;
        this->tailDataNum = bigTailDataNum;
    }
    else
    {
        this->coreDataNum = smallCoreDataNum;
        this->tileNum = finalSmallTileNum;
        this->tailDataNum = smallTailDataNum;
        globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (GetBlockIdx() - tailBlockNum);
    }
    xGm.SetGlobalBuffer((__gm__ TYPE_X *)x + globalBufferIndex, this->coreDataNum);
    expGm.SetGlobalBuffer((__gm__ TYPE_X *)exponent + globalBufferIndex, this->coreDataNum);
    yGm.SetGlobalBuffer((__gm__ TYPE_Y *)y + globalBufferIndex, this->coreDataNum);
    pipe.InitBuffer(inQueue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X) * 2);
    pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
    pipe.InitBuffer(absTmp, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(zeroTmp, this->tileDataNum * sizeof(float));
    if constexpr (std::is_same_v<TYPE_X, float32_t> || std::is_same_v<TYPE_X, int32_t>){
        pipe.InitBuffer(maskTmp, this->tileDataNum * sizeof(uint8_t));
    }
    else
    {
        pipe.InitBuffer(xCastTmp, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(expCastTmp, this->tileDataNum * sizeof(float));

        if constexpr ( std::is_same_v<TYPE_X, uint8_t> || std::is_same_v<TYPE_X, int8_t>){
            pipe.InitBuffer(tranCastTmp, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(yCastTmp, this->tileDataNum * sizeof(float));
        }else{
            pipe.InitBuffer(maskTmp, this->tileDataNum * sizeof(uint8_t));
        }
    }

}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelPow<TYPE_X, TYPE_Y>::CopyIn(int32_t progress)
{
    LocalTensor<TYPE_X> xLocal = inQueue.AllocTensor<TYPE_X>();
    DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
    DataCopy(xLocal[this->processDataNum], expGm[progress * this->tileDataNum], this->processDataNum);
    inQueue.EnQue(xLocal);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelPow<TYPE_X, TYPE_Y>::CopyOut(int32_t progress)
{
    LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
    DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
    outQueueY.FreeTensor(yLocal);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelPow<TYPE_X, TYPE_Y>::PowCompute(LocalTensor<float> &xLocal, LocalTensor<float> &expLocal,
                                                            LocalTensor<float> &yLocal, LocalTensor<uint8_t> &mask){
    LocalTensor<float> p1 = absTmp.Get<float>();
    LocalTensor<float> zeros = zeroTmp.Get<float>();
    LocalTensor<float> &scalars = zeros;
    LocalTensor<float> &ones = expLocal;
    Duplicate(scalars, 2.0f, this->processDataNum);
    Abs(p1, xLocal, this->processDataNum);
    Ln(p1, p1, this->processDataNum);
    Mul(p1, p1, expLocal, this->processDataNum);
    Exp(yLocal, p1, this->processDataNum); // yLocal = exp(y*len(|x|))

    Abs(p1, expLocal, this->processDataNum);
    
    Div(zeros, p1, scalars, this->processDataNum);
    Cast(zeros, zeros, RoundMode::CAST_TRUNC, this->processDataNum);
    Mul(zeros, zeros, scalars, this->processDataNum);
    Sub(p1, p1, zeros, this->processDataNum);
    
    Muls(p1, p1, -2.0f, this->processDataNum);
    Adds(p1, p1, 1.0f, this->processDataNum);
    Mul(p1, p1, yLocal, this->processDataNum);

    Duplicate(zeros, 0.0f, this->processDataNum);

    Compare(mask, xLocal, zeros, CMPMODE::LT, this->processDataNum); // x小于0的部分
    Select(yLocal, mask, p1, yLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);

    Compare(mask, expLocal, zeros, CMPMODE::EQ, this->processDataNum); // exp等于0的部分
    Duplicate(ones, 1.0f, this->processDataNum);
    Select(p1, mask, ones, zeros, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);

    Compare(mask, xLocal, zeros, CMPMODE::EQ, this->processDataNum); // x等于0的部分
    Select(yLocal, mask, p1, yLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelPow<TYPE_X, TYPE_Y>::PowComputeUint8(LocalTensor<float> &xLocal, LocalTensor<float> &expLocal,
                                                                    LocalTensor<float> &yLocal, LocalTensor<uint8_t> &mask){
    LocalTensor<float> p1 = absTmp.Get<float>();
    LocalTensor<float> zeros = zeroTmp.Get<float>();
    LocalTensor<float> &ones = expLocal;
    Ln(p1, xLocal, this->processDataNum);
    Mul(p1, p1, expLocal, this->processDataNum);
    Exp(yLocal, p1, this->processDataNum); // p2 = exp(y*len(x))

    Duplicate(zeros, 0.0f, this->processDataNum);
    Compare(mask, expLocal, zeros, CMPMODE::EQ, this->processDataNum); // exp等于0的部分
    Duplicate(ones, 1.0f, this->processDataNum);
    Select(p1, mask, ones, zeros, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
    
    Compare(mask, xLocal, zeros, CMPMODE::EQ, this->processDataNum); // x等于0的部分
    Select(yLocal, mask, p1, yLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
    
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelPow<TYPE_X, TYPE_Y>::Compute(int32_t progress)
{
    LocalTensor<TYPE_X> xLocal = inQueue.DeQue<TYPE_X>();
    LocalTensor<TYPE_X> expLocal = xLocal[this->processDataNum];
    LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();
    if constexpr ( std::is_same_v<TYPE_X, half> || std::is_same_v<TYPE_X, bfloat16_t>
                     || std::is_same_v<TYPE_X, int16_t>)
    {
        LocalTensor<float> xLocalFp = xCastTmp.Get<float>();
        LocalTensor<float> expLocalFp = expCastTmp.Get<float>();
        LocalTensor<float> yLocalFp = xLocal.template ReinterpretCast<float>();
        LocalTensor<uint8_t> mask = maskTmp.Get<uint8_t>();
        Cast(xLocalFp, xLocal, RoundMode::CAST_NONE, this->processDataNum);
        Cast(expLocalFp, expLocal, RoundMode::CAST_NONE, this->processDataNum);
        PowCompute(xLocalFp, expLocalFp, yLocalFp, mask);
        Cast(yLocal, yLocalFp, RoundMode::CAST_RINT, this->processDataNum);
    }
    if constexpr (std::is_same_v<TYPE_X, int32_t>){
        LocalTensor<float> xLocalFp = xLocal.template ReinterpretCast<float>();
        LocalTensor<float> expLocalFp = expLocal.template ReinterpretCast<float>();
        LocalTensor<float> yLocalFp = yLocal.template ReinterpretCast<float>();
        LocalTensor<uint8_t> mask = maskTmp.Get<uint8_t>();
        Cast(xLocalFp, xLocal, RoundMode::CAST_NONE, this->processDataNum);
        Cast(expLocalFp, expLocal, RoundMode::CAST_NONE, this->processDataNum);
        PowCompute(xLocalFp, expLocalFp, yLocalFp, mask);
        Cast(yLocal, yLocalFp, RoundMode::CAST_RINT, this->processDataNum);
    }
    if constexpr ( std::is_same_v<TYPE_X, uint8_t>){
        LocalTensor<float> xLocalFp = xCastTmp.Get<float>();
        LocalTensor<float> expLocalFp = expCastTmp.Get<float>();
        LocalTensor<float> yLocalFp = yCastTmp.Get<float>();
        LocalTensor<half> tmpLocal = tranCastTmp.Get<half>();
        Cast(tmpLocal, xLocal, RoundMode::CAST_NONE, this->processDataNum);
        Cast(xLocalFp, tmpLocal, RoundMode::CAST_NONE, this->processDataNum);
        Cast(tmpLocal, expLocal, RoundMode::CAST_NONE, this->processDataNum);
        Cast(expLocalFp, tmpLocal, RoundMode::CAST_NONE, this->processDataNum);
        PowComputeUint8(xLocalFp, expLocalFp, yLocalFp, xLocal);
        Cast(tmpLocal, yLocalFp, RoundMode::CAST_RINT, this->processDataNum);
        Cast(yLocal, tmpLocal, RoundMode::CAST_RINT, this->processDataNum);
    }
    if constexpr ( std::is_same_v<TYPE_X, int8_t>){
        LocalTensor<float> xLocalFp = xCastTmp.Get<float>();
        LocalTensor<float> expLocalFp = expCastTmp.Get<float>();
        LocalTensor<float> yLocalFp = yCastTmp.Get<float>();
        LocalTensor<half> tmpLocal = tranCastTmp.Get<half>();
        LocalTensor<uint8_t> mask = xLocal.template ReinterpretCast<uint8_t>();
        Cast(tmpLocal, xLocal, RoundMode::CAST_NONE, this->processDataNum);
        Cast(xLocalFp, tmpLocal, RoundMode::CAST_NONE, this->processDataNum);
        Cast(tmpLocal, expLocal, RoundMode::CAST_NONE, this->processDataNum);
        Cast(expLocalFp, tmpLocal, RoundMode::CAST_NONE, this->processDataNum);
        PowCompute(xLocalFp, expLocalFp, yLocalFp, mask);
        Cast(tmpLocal, yLocalFp, RoundMode::CAST_RINT, this->processDataNum);
        Cast(yLocal, tmpLocal, RoundMode::CAST_RINT, this->processDataNum);
    }
    if constexpr ( std::is_same_v<TYPE_X, float32_t>)
    {
        LocalTensor<uint8_t> mask = maskTmp.Get<uint8_t>();
        PowCompute(xLocal, expLocal, yLocal, mask);
    }
    outQueueY.EnQue<TYPE_Y>(yLocal);
    inQueue.FreeTensor(xLocal);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelPow<TYPE_X, TYPE_Y>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    for (int32_t i = 0; i < loopCount-1; i++)
    {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
    this->processDataNum = this->tailDataNum;
    CopyIn(loopCount-1);
    Compute(loopCount-1);
    CopyOut(loopCount-1);
}

} // namespace KernelPow
#endif // POW_H