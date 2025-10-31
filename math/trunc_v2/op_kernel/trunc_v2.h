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
 * \file trunc_v2.h
 * \brief
*/
#ifndef TRUNCV2_H
#define TRUNCV2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "trunc_v2_tiling_data.h"
#include "trunc_v2_tiling_key.h"

namespace NsTruncV2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class TruncV2 {
public:
    __aicore__ inline TruncV2(){};

    __aicore__ inline uint32_t AlignUp(uint32_t a, uint32_t b); 
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, const TruncV2TilingData &tilingData);
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

    int32_t core_id;
    uint32_t processDataNum;
    uint32_t tileLength;
    uint32_t aligned_tile_size;

    TruncV2TilingData tiling;
};

template <typename T>
__aicore__ inline uint32_t TruncV2<T>::AlignUp(uint32_t a, uint32_t b) 
{
    if (b == 0)
        return a;
    return ((a + 32 / b - 1) / (32 / b)) * (32 / b);
}

template <typename T>
__aicore__ inline void TruncV2<T>::Init(GM_ADDR x, GM_ADDR z, const TruncV2TilingData &tilingData)
{
    this->tiling = tilingData;
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    this->core_id = AscendC::GetBlockIdx();
    // 如果当前核没有分配到向量，直接返回
    if (tiling.core_element_count[this->core_id] <= 0) {
        return ;
    }

    uint32_t element_offest = (uint32_t)tiling.core_element_start[this->core_id];

    inputGMX.SetGlobalBuffer((__gm__ T *)x + element_offest, tiling.core_element_count[this->core_id]);
    outputGMZ.SetGlobalBuffer((__gm__ T *)z + element_offest, tiling.core_element_count[this->core_id]);

    this->aligned_tile_size = AlignUp((uint32_t)tiling.tile_element_num, sizeof(T));
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->aligned_tile_size * sizeof(T));
    pipe.InitBuffer(outputQueueZ, BUFFER_NUM, this->aligned_tile_size * sizeof(T));
}

template <typename T>
__aicore__ inline void TruncV2<T>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();

    AscendC::DataCopyExtParams copyParams{
        1,
        static_cast<uint32_t>(this->tileLength * sizeof(T)),
        0, 0, 0
    };
        
    AscendC::DataCopyPadExtParams<T> padParams{
        true, 
        0,
        static_cast<uint8_t>(this->processDataNum - this->tileLength),
        0
    };
        
    // 使用相对偏移，因为inputGMX的base已经是当前core的起始位置
    AscendC::DataCopyPad(xLocal, inputGMX[progress * tiling.tile_element_num], copyParams, padParams);

    // 如果需要对齐，填充剩余部分为0
    if (this->processDataNum > this->tileLength) {
        xLocal[this->tileLength].SetValue(this->processDataNum - this->tileLength, static_cast<T>(0));
    }

    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void TruncV2<T>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();

    // 输出时使用 DataCopyPad 只复制有效数据
    AscendC::DataCopyExtParams copyParams{
        1,  // blockCount
        static_cast<uint32_t>(this->tileLength * sizeof(T)),  // blockLen: 只复制有效数据
        0,  // srcStride
        0,  // dstStride
        0   // rsv
    };
        
    AscendC::DataCopyPad(outputGMZ[progress * tiling.tile_element_num], 
                        zLocal, 
                        copyParams);
    outputQueueZ.FreeTensor(zLocal);
}

template <typename T>
__aicore__ inline void TruncV2<T>::Compute(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outputQueueZ.AllocTensor<T>();

    AscendC::Trunc(zLocal, xLocal, this->processDataNum);

    outputQueueZ.EnQue<T>(zLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void TruncV2<T>::Process()
{
    int32_t loopCount = tiling.core_loop_times[this->core_id];
        
    // 处理完整的循环
    for (int32_t i = 0; i < loopCount; i++) 
    {
        // 判断是否是最后一次循环
        if (i == loopCount - 1) {
            // 最后一次循环，处理tail
            this->tileLength = (uint32_t)tiling.core_tail_elements[this->core_id];
            this->processDataNum = AlignUp((uint32_t)tiling.core_tail_elements[this->core_id], sizeof(T));
        } else {
        // 正常循环
        this->tileLength = (uint32_t)tiling.tile_element_num;
            this->processDataNum = this->aligned_tile_size;
        }
            
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

} // namespace NsTruncV2
#endif // TRUNCV2_H