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
 * \file axpy_v3.h
 * \brief
 */
#ifndef AXPYV3_H
#define AXPYV3_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "axpy_v3_tiling_data.h"
#include "axpy_v3_tiling_key.h"

namespace NsAxpyV3 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class AxpyV3 {
public:
    __aicore__ inline AxpyV3(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const AxpyV3TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    AscendC::TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueY;
    AscendC::TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueZ;
    AscendC::GlobalTensor<T> inputGMX;
    AscendC::GlobalTensor<T> inputGMY;
    AscendC::GlobalTensor<T> outputGMZ;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    float a;
};

template <typename T>
__aicore__ inline void AxpyV3<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const AxpyV3TilingData* tilingData)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint32_t coreNum = AscendC::GetBlockIdx();
    uint32_t globalBufferIndex = tilingData->bigCoreDataNum * AscendC::GetBlockIdx();
    this->tileDataNum = tilingData->tileDataNum;
    this->a = tilingData->a;
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
}

template <typename T>
__aicore__ inline void AxpyV3<T>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.AllocTensor<T>();
    AscendC::DataCopy(xLocal, inputGMX[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(yLocal, inputGMY[progress * this->tileDataNum], this->processDataNum);
    inputQueueX.EnQue(xLocal);
    inputQueueY.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void AxpyV3<T>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
    AscendC::DataCopy(outputGMZ[progress * this->tileDataNum], zLocal, this->processDataNum);
    outputQueueZ.FreeTensor(zLocal);
}

template <typename T>
__aicore__ inline void AxpyV3<T>::Compute(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outputQueueZ.AllocTensor<T>();
    // float a = 5.0f; 
    AscendC::Muls(xLocal, xLocal, static_cast<T>(this->a), this->processDataNum);
    AscendC::Add(zLocal, xLocal, yLocal, this->processDataNum);
    outputQueueZ.EnQue<T>(zLocal);
    inputQueueX.FreeTensor(xLocal);
    inputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void AxpyV3<T>::Process()
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

} // namespace NsAxpyV3
#endif // AxpyV3_H



// #ifndef SUBV2_H
// #define SUBV2_H

// #include "kernel_operator.h"
// #include "kernel_tiling/kernel_tiling.h"
// #include "sub_v2_tiling_data.h"
// #include "sub_v2_tiling_key.h"

// namespace NsSubV2 {

// using namespace AscendC;

// constexpr int32_t BUFFER_NUM = 2;

// template <typename T>
// class SubV2 {
// public:
//     __aicore__ inline SubV2(){};

//     __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const SubV2TilingData* tilingData);
//     __aicore__ inline void Process();

// private:
//     __aicore__ inline void CopyIn(int32_t progress);
//     __aicore__ inline void CopyOut(int32_t progress);
//     __aicore__ inline void Compute(int32_t progress);

// private:
//     TPipe pipe;
//     TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
//     TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueY;
//     TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueZ;
//     GlobalTensor<T> inputGMX;
//     GlobalTensor<T> inputGMY;
//     GlobalTensor<T> outputGMZ;

//     uint32_t coreDataNum;
//     uint32_t tileNum;
//     uint32_t tileDataNum;
//     uint32_t tailDataNum;
//     uint32_t processDataNum;
// };

// template <typename T>
// __aicore__ inline void SubV2<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const SubV2TilingData* tilingData)
// {
//         ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
//         uint32_t coreNum = AscendC::GetBlockIdx();
//         uint32_t globalBufferIndex = tilingData->bigCoreDataNum * AscendC::GetBlockIdx();
//         this->tileDataNum = tilingData->tileDataNum;
//         if (coreNum < tilingData->tailBlockNum) { 
//           this->coreDataNum = tilingData->bigCoreDataNum;
//           this->tileNum = tilingData->finalBigTileNum;
//           this->tailDataNum = tilingData->bigTailDataNum;
//         }
//         else { 
//           this->coreDataNum = tilingData->smallCoreDataNum;
//           this->tileNum = tilingData->finalSmallTileNum;
//           this->tailDataNum = tilingData->smallTailDataNum;
//           globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) * (AscendC::GetBlockIdx() - tilingData->tailBlockNum);
//         }
//         inputGMX.SetGlobalBuffer((__gm__ T*)x + globalBufferIndex, this->coreDataNum);
//         inputGMY.SetGlobalBuffer((__gm__ T*)y + globalBufferIndex, this->coreDataNum);
//         outputGMZ.SetGlobalBuffer((__gm__ T*)z + globalBufferIndex, this->coreDataNum);
//         pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileDataNum * sizeof(T));
//         pipe.InitBuffer(inputQueueY, BUFFER_NUM, this->tileDataNum * sizeof(T));
//         pipe.InitBuffer(outputQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(T));

//     }

// template <typename T>
// __aicore__ inline void SubV2<T>::CopyIn(int32_t progress)
// {
//     AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
//     AscendC::LocalTensor<T> yLocal = inputQueueY.AllocTensor<T>();
//     AscendC::DataCopy(xLocal, inputGMX[progress * this->tileDataNum], this->processDataNum);
//     AscendC::DataCopy(yLocal, inputGMY[progress * this->tileDataNum], this->processDataNum);
//     inputQueueX.EnQue(xLocal);
//     inputQueueY.EnQue(yLocal);
// }

// template <typename T>
// __aicore__ inline void SubV2<T>::CopyOut(int32_t progress)
// {
//     AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
//     AscendC::DataCopy(outputGMZ[progress * this->tileDataNum], zLocal, this->processDataNum);
//     outputQueueZ.FreeTensor(zLocal);
// }

// template <typename T>
// __aicore__ inline void SubV2<T>::Compute(int32_t progress)
// {
//     AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
//     AscendC::LocalTensor<T> yLocal = inputQueueY.DeQue<T>();
//     AscendC::LocalTensor<T> zLocal = outputQueueZ.AllocTensor<T>();
//     AscendC::Sub(zLocal, xLocal, yLocal, this->processDataNum);
//     outputQueueZ.EnQue<T>(zLocal);
//     inputQueueX.FreeTensor(xLocal);
//     inputQueueY.FreeTensor(yLocal);
// }

// template <typename T>
// __aicore__ inline void SubV2<T>::Process()
// {
//     int32_t loopCount = this->tileNum;
//     this->processDataNum = this->tileDataNum;
//     for (int32_t i = 0; i < loopCount; i++) {
//         if (i == this->tileNum - 1) {
//             this->processDataNum = this->tailDataNum;
//         }
//         CopyIn(i);
//         Compute(i);
//         CopyOut(i);
//     }
// }

// } // namespace NsSubV2
// #endif // SubV2_H