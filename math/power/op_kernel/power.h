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
 * \file power.h
 * \brief
*/
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "power_tiling_data.h"
#include "power_tiling_key.h"


namespace NsPower {
constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
using namespace AscendC;


template <typename TYPE_X, typename TYPE_Y, typename TYPE_Z>
class Power {
public:
    __aicore__ inline Power() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                const PowerTilingData* tilingData
                                ) 
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        uint64_t coreNum = GetBlockIdx();
        this->globalBufferIndex = tilingData->bigCoreDataNum * coreNum;
        this->tileDataNum = tilingData->tileDataNum;
        this->is_input0_scalar = tilingData->is_input0_scalar;
        this->is_input1_scalar =  tilingData->is_input1_scalar;

        this->x1Dim = tilingData->x1Dim;
        this->x2Dim = tilingData->x2Dim;
        this->yDim  = tilingData->yDim;
        const int64_t* x1ShapeSrc = tilingData->x1Shape;
        const int64_t* x2ShapeSrc = tilingData->x2Shape;
        const int64_t* yShapeSrc = tilingData->yShape;
        for (int i = 0; i < tilingData->x1Dim; ++i) this->x1Shape[i] = x1ShapeSrc[i];
        for (int i = 0; i < tilingData->x2Dim; ++i) this->x2Shape[i] = x2ShapeSrc[i];
        for (int i = 0; i < tilingData->yDim; ++i) this->yShape[i] = yShapeSrc[i];
        const int64_t* strideX1Src = tilingData->strideX1;
        const int64_t* strideX2Src = tilingData->strideX2;
        const int64_t* strideYSrc  = tilingData->strideY;
        for (int i = 0; i < 10; ++i) {
            this->strideX1[i] = strideX1Src[i];
            this->strideX2[i] = strideX2Src[i];
            this->strideY[i]  = strideYSrc[i];
        }
        this->isSameX1  = tilingData->isSameX1;
        this->isSameX2  = tilingData->isSameX2;


        if (coreNum <  tilingData->tailBlockNum) {
            this->coreDataNum =  tilingData->bigCoreDataNum;
            this->tileNum =  tilingData->finalBigTileNum;
            this->tailDataNum =  tilingData->bigTailDataNum;
        } else {
            this->coreDataNum =  tilingData->smallCoreDataNum;
            this->tileNum =  tilingData->finalSmallTileNum;
            this->tailDataNum =  tilingData->smallTailDataNum;
            globalBufferIndex -= ( tilingData->bigCoreDataNum -  tilingData->smallCoreDataNum) * (coreNum -  tilingData->tailBlockNum);
        }
        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, coreDataNum);
        zGm.SetGlobalBuffer((__gm__ TYPE_Z*)z + globalBufferIndex, coreDataNum);
        PRINTF("xglobalBufferIndex=%llu, yglobalBufferIndex=%llu, zglobalBufferIndex=%llu\n",
                 (uint64_t)((__gm__ TYPE_X*)x + globalBufferIndex), (uint64_t)((__gm__ TYPE_Y*)y + globalBufferIndex),
                 (uint64_t)((__gm__ TYPE_Z*)z + globalBufferIndex) );   
        PRINTF("core=%u, globalBufferIndex=%llu, tileDataNum=%u, sizeof(TYPE_X)=%u\n",
                    GetBlockIdx(), this->globalBufferIndex, this->tileDataNum, sizeof(TYPE_X));


        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileDataNum * sizeof(TYPE_Y));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileDataNum * sizeof(TYPE_Z));

        // ===== Determine compute dtype =====
        using ComputeType = typename std::conditional<
            std::is_same<TYPE_Z, int8_t>::value ||
            std::is_same<TYPE_Z, uint8_t>::value ||
            std::is_same<TYPE_Z, int16_t>::value,
            half,
            typename std::conditional<
                std::is_same<TYPE_Z, bfloat16_t>::value,
                float,
                TYPE_Z>::type>::type;

         // === Init Tbuf ===
        pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(ComputeType));
        pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(ComputeType));
        pipe.InitBuffer(tmp3, this->tileDataNum * sizeof(ComputeType));

        //=== Init Tbuf === 用于cast二次中转
        pipe.InitBuffer(tmp4, this->tileDataNum * sizeof(half));
        pipe.InitBuffer(tmp5, this->tileDataNum * sizeof(float));
        this->computeTypeSize = sizeof(ComputeType);

        PRINTF("Init: core=%lu, tileNum=%u, tileDataNum=%u\n",
                 coreNum, this->tileNum, this->tileDataNum);
        PRINTF("sizeof(TYPE_X)=%u, sizeof(TYPE_Y)=%u, sizeof(ComputeType)=%u\n",
         sizeof(TYPE_X), sizeof(TYPE_Y), sizeof(ComputeType));
    }

    __aicore__ inline void Process() {
        PRINTF("Process start: tileNum=%u  ", this->tileNum);
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount; i++) {
            if (i == this->tileNum - 1) {
              this->processDataNum = this->tailDataNum;
            }
            PRINTF("processDataNum=%u, totalBytes_X=%u, totalBytes_Y=%u\n",
                processDataNum,
                processDataNum * sizeof(TYPE_X),
                processDataNum * sizeof(TYPE_Y));
            if ((processDataNum * sizeof(TYPE_X)) % 32 != 0)
                PRINTF("[WARN] X DMA not 32B aligned\n");
            if (i == 0) PRINTF("Tile[0] dataNum=%u", this->processDataNum);
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        PRINTF("Process done \n ");
    }

private:
    // ===============================================
    //  广播索引计算函数
    //  将输出索引映射到输入索引
    // ===============================================
        __aicore__ inline uint32_t GetBroadcastIndexEff( uint32_t linearIdx,
                                                const uint32_t *effStride  // 预计算后的有效stride
                                                )
    {
        // 利用预计算stride快速定位
        // linearIdx 按输出 shape 展开
        uint32_t offset = 0;
        uint32_t tmp = linearIdx;
        for (int i = 0; i < yDim; ++i) {
            uint32_t stride_y = this->strideY[i];
            if (stride_y == 0) {
                continue;
            }
            uint32_t coord = tmp / stride_y;
            tmp %= stride_y;
            // 如果该维度是广播维（shape[i] == 1），则host已经将effStride[i]设为0
            offset += coord * effStride[i];
        }
        return offset;
    }

    __aicore__ inline void CopyIn(int32_t progress) {
        LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        LocalTensor<TYPE_Y> yLocal = inQueueY.AllocTensor<TYPE_Y>();
        if (this->is_input0_scalar && !this->is_input1_scalar) {
            TYPE_X scalarValX;
            scalarValX = xGm.GetValue(0);//duplicate不支持int8和uint8
            for (int i = 0; i < this->processDataNum; ++i) {
                xLocal.SetValue(i, scalarValX);
            }
            DataCopy(yLocal, yGm[progress * this->tileDataNum], this->processDataNum);
        }
        else if (!this->is_input0_scalar && this->is_input1_scalar) {
            DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
            TYPE_Y scalarValY;
            scalarValY = yGm.GetValue(0);
             for (int i = 0; i < this->processDataNum; ++i) {
                yLocal.SetValue(i, scalarValY);
            }
        }
        else{
            // ====== 判断是否完全匹配 ======
            if (isSameX1 && isSameX2 ) {
                // 无广播，直接DMA搬运
                DataCopy(xLocal, xGm[progress * tileDataNum], processDataNum);
                DataCopy(yLocal, yGm[progress * tileDataNum], processDataNum);
            } else if ( !isSameX1 && isSameX2 ){//x广播yDMA搬运
                 for (uint32_t i = 0; i < processDataNum; ++i) {
                    uint32_t globalIdx = globalBufferIndex + i;
                    uint32_t idxX = GetBroadcastIndexEff(globalIdx, this->strideX1);
                    xLocal.SetValue(i, xGm.GetValue(idxX));
                }
                DataCopy(yLocal, yGm[progress * tileDataNum], processDataNum);  
            }else if ( isSameX1 && !isSameX2 ){//xDMA搬运y广播
                DataCopy(xLocal, xGm[progress * tileDataNum], processDataNum);
                 for (uint32_t i = 0; i < processDataNum; ++i) {
                    uint32_t globalIdx = globalBufferIndex + i;
                    uint32_t idxY = GetBroadcastIndexEff(globalIdx, this->strideX2);
                    yLocal.SetValue(i, yGm.GetValue(idxY));
                }
            }else {//全广播
                 for (uint32_t i = 0; i < processDataNum; ++i) {
                    uint32_t globalIdx = globalBufferIndex + i;
                    uint32_t idxX = GetBroadcastIndexEff(globalIdx, this->strideX1);
                    uint32_t idxY = GetBroadcastIndexEff(globalIdx, this->strideX2);
                    xLocal.SetValue(i, xGm.GetValue(idxX));
                    yLocal.SetValue(i, yGm.GetValue(idxY));
                    
                }
            }          
        }
        
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
        if (progress == 0) PRINTF("CopyIn ok (tile0)\n");
    }

    __aicore__ inline void Compute(int32_t progress) {
        LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        LocalTensor<TYPE_Y> yLocal = inQueueY.DeQue<TYPE_Y>();
        LocalTensor<TYPE_Z> zLocal = outQueueZ.AllocTensor<TYPE_Z>();

        using ComputeType = typename std::conditional<
            std::is_same<TYPE_Z, int8_t>::value ||
            std::is_same<TYPE_Z, uint8_t>::value ||
            std::is_same<TYPE_Z, int16_t>::value,
            half,
            typename std::conditional<
                std::is_same<TYPE_Z, bfloat16_t>::value,
                float,
                TYPE_Z>::type>::type;

         // 根据  决定 cast + power 流程
         if constexpr(std::is_same<ComputeType, int32_t>::value) {//TYPE_Z == DT_INT32
            LocalTensor<int32_t> xCast = tmp1.Get<int32_t>();
            LocalTensor<int32_t> yCast = tmp2.Get<int32_t>();
            LocalTensor<int32_t> zCast = tmp3.Get<int32_t>();

            if constexpr (std::is_same<TYPE_X, int16_t>::value){//DT_INT16转成half再转成int32可能有精度损失，转成folat再转int32
                LocalTensor<float> xComputeCast = tmp5.Get<float>();
                Cast(xComputeCast, xLocal, RoundMode::CAST_NONE, this->processDataNum);
                Cast(xCast, xComputeCast, RoundMode::CAST_RINT, this->processDataNum);
                }
            else if constexpr(std::is_same<TYPE_X, int8_t>::value || std::is_same<TYPE_X, uint8_t>::value){//DT_INT8,DT_UINT8先转为half再转为int32
                LocalTensor<half> xComputeCast = tmp4.Get<half>();
                Cast(xComputeCast, xLocal, RoundMode::CAST_NONE, this->processDataNum);
                Cast(xCast, xComputeCast, RoundMode::CAST_RINT, this->processDataNum);
            }
            else
                xCast = xLocal.template ReinterpretCast<int32_t>();
            //Y处理照搬X
            if constexpr (std::is_same<TYPE_Y, int16_t>::value){//DT_INT16转成half再转成int32可能有精度损失，转成folat再转int32
                LocalTensor<float> yComputeCast = tmp5.Get<float>();
                Cast(yComputeCast, yLocal, RoundMode::CAST_NONE, this->processDataNum);
                Cast(yCast, yComputeCast, RoundMode::CAST_RINT, this->processDataNum);
            }
            else if constexpr(std::is_same<TYPE_Y, int8_t>::value || std::is_same<TYPE_Y, uint8_t>::value){//DT_INT8,DT_UINT8先转为half再转为int32
                LocalTensor<half> yComputeCast = tmp4.Get<half>();
                Cast(yComputeCast, yLocal, RoundMode::CAST_NONE, this->processDataNum);
                Cast(yCast, yComputeCast, RoundMode::CAST_RINT, this->processDataNum);
            }
            else
                yCast = yLocal.template ReinterpretCast<int32_t>();
            //运算
            AscendC::Power(zCast, xCast, yCast);
            //处理Z
            //ComputeType == TYPE_Z
            zLocal = zCast.template ReinterpretCast<TYPE_Z>();
        } else if constexpr(std::is_same<ComputeType, half>::value){//TYPE_Z == DT_FLOAT16  或  int8_t,uint8_t,int16_t
            LocalTensor<half> xCast = tmp1.Get<half>();
            LocalTensor<half> yCast = tmp2.Get<half>();
            LocalTensor<half> zCast = tmp3.Get<half>();
            if constexpr (std::is_same<TYPE_X, int16_t>::value){//DT_INT16转成halfn有直接CAST_NONE但是按理说会出现精度误差，最好别用,转成float再转half
                LocalTensor<float> xComputeCast = tmp5.Get<float>();
                Cast(xComputeCast, xLocal, RoundMode::CAST_NONE, this->processDataNum);
                Cast(xCast, xComputeCast, RoundMode::CAST_RINT, this->processDataNum);
                }
            else if constexpr(std::is_same<TYPE_X, int8_t>::value || std::is_same<TYPE_X, uint8_t>::value){//DT_INT8,DT_UINT8直接转half
                Cast(xCast, xLocal, RoundMode::CAST_NONE, this->processDataNum);
            }
            else if constexpr(std::is_same<TYPE_X, int32_t>::value){//DT_INT32转half可能会有损失，这里先这么些
                //int32_t->half时roundMode不生效，与SetDeqScale(half scale)接口配合使用
                half scale = 1.0;
                SetDeqScale(scale);
                Cast(xCast, xLocal, RoundMode::CAST_NONE, this->processDataNum);
            }
            else
                xCast = xLocal.template ReinterpretCast<half>();
            //Y处理照搬X
            if constexpr (std::is_same<TYPE_Y, int16_t>::value){//DT_INT16转成half再转成int32可能有精度损失，转成folat再转int32
                LocalTensor<float> yComputeCast = tmp5.Get<float>();
                Cast(yComputeCast, yLocal, RoundMode::CAST_NONE, this->processDataNum);
                Cast(yCast, yComputeCast, RoundMode::CAST_RINT, this->processDataNum);
            }
            else if constexpr(std::is_same<TYPE_Y, int8_t>::value || std::is_same<TYPE_Y, uint8_t>::value){//DT_INT8,DT_UINT8先转为half再转为int32
                Cast(yCast, yLocal, RoundMode::CAST_NONE, this->processDataNum);
            }
            else if constexpr(std::is_same<TYPE_Y, int32_t>::value){//DT_INT32转half可能会有损失，这里先这么些
                //int32_t->half时roundMode不生效，与SetDeqScale(half scale)接口配合使用
                half scale = 1.0;
                SetDeqScale(scale);
                Cast(yCast, yLocal, RoundMode::CAST_NONE, this->processDataNum);
            }
            else
                yCast = yLocal.template ReinterpretCast<half>();
            //运算
            AscendC::Power(zCast, xCast, yCast);
            //处理Z
            if constexpr (!std::is_same<TYPE_Z, ComputeType>::value)//int8、uint8、int16
                if constexpr(std::is_same<TYPE_Z, int16_t>::value)//half转int16不支持:CAST_NONE
                    Cast(zLocal, zCast, RoundMode::CAST_RINT, this->processDataNum);//但是转过去不会损失精度因为他们就是从int8_t,uint8_t,int16_t转来的不会溢出
                else
                    Cast(zLocal, zCast, RoundMode::CAST_NONE, this->processDataNum);
            else
                zLocal = zCast.template ReinterpretCast<TYPE_Z>();
        } else if constexpr(std::is_same<ComputeType, float>::value){//TYPE_Z == DT_FLOAT
            LocalTensor<float> xCast = tmp1.Get<float>();
            LocalTensor<float> yCast = tmp2.Get<float>();
            LocalTensor<float> zCast = tmp3.Get<float>();
            //TYPE_X可能时七种数据类型中的任何一种
            if constexpr (std::is_same<TYPE_X, int8_t>::value || std::is_same<TYPE_X, uint8_t>::value){//DT_INT8、DT_UINT8转成half再转float
                LocalTensor<half> xComputeCast = tmp4.Get<half>();
                Cast(xComputeCast, xLocal, RoundMode::CAST_NONE, this->processDataNum);
                Cast(xCast, xComputeCast, RoundMode::CAST_NONE, this->processDataNum);
            }
            else if constexpr(std::is_same<TYPE_X, int16_t>::value || std::is_same<TYPE_X, int32_t>::value
                                    || std::is_same<TYPE_X, bfloat16_t>::value ||std::is_same<TYPE_X, half>::value){//DT_INT16,DT_UINT32,DT_BF16直接转float
                Cast(xCast, xLocal, RoundMode::CAST_NONE, this->processDataNum);
            }
            else
                xCast = xLocal.template ReinterpretCast<float>();
            //Y搬X
            if constexpr (std::is_same<TYPE_Y, int8_t>::value || std::is_same<TYPE_Y, uint8_t>::value){//DT_INT8、DT_UINT8转成half再转float
                LocalTensor<half> yComputeCast = tmp4.Get<half>();
                Cast(yComputeCast, yLocal, RoundMode::CAST_NONE, this->processDataNum);
                Cast(yCast, yComputeCast, RoundMode::CAST_NONE, this->processDataNum);
            }
            else if constexpr(std::is_same<TYPE_Y, int16_t>::value || std::is_same<TYPE_Y, int32_t>::value
                                    || std::is_same<TYPE_Y, bfloat16_t>::value ||std::is_same<TYPE_X, half>::value){//DT_INT16,DT_UINT32,DT_BF16直接转float
                Cast(yCast, yLocal, RoundMode::CAST_NONE, this->processDataNum);
            }
            else
                yCast = yLocal.template ReinterpretCast<float>();
            //运算
            AscendC::Power(zCast, xCast, yCast);
            //处理Z
            //ComputeType == TYPE_Z
            if constexpr (!std::is_same<TYPE_Z, ComputeType>::value)//TYPE_Z=bf16
                Cast(zLocal, zCast, RoundMode::CAST_NONE, this->processDataNum);
            else//只是为了将输出送回zlocal统一格式
                zLocal = zCast.template ReinterpretCast<TYPE_Z>();
        }
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
        if (progress == 0) PRINTF("Compute done (tile0)");
    }

    __aicore__ inline void CopyOut(uint32_t progress) {
        if (progress == 0) PRINTF("CopyOut start (tile0)\n");
        LocalTensor<TYPE_Z> zLocal = outQueueZ.DeQue<TYPE_Z>();  
        DataCopy(zGm[progress * this->tileDataNum], zLocal, this->processDataNum);
        outQueueZ.FreeTensor(zLocal);
        if (progress == 0) PRINTF("CopyOut ok (tile0)");
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    TBuf<QuePosition::VECCALC> tmp1, tmp2, tmp3;
    TBuf<QuePosition::VECCALC> tmp4, tmp5;
    GlobalTensor<TYPE_X> xGm;
    GlobalTensor<TYPE_Y> yGm;
    GlobalTensor<TYPE_Z> zGm;

    //实际运算
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;

    uint64_t globalBufferIndex;
    uint32_t computeTypeSize;
    //标量判断
    uint32_t is_input0_scalar;
    uint32_t is_input1_scalar;
    //形状广播
    uint32_t x1Shape[10];
    uint32_t x2Shape[10];
    uint32_t yShape[10];
    uint32_t x1Dim;
    uint32_t x2Dim;
    uint32_t yDim;
    uint32_t isSameX1;
    uint32_t isSameX2;
    uint32_t strideX1[10];
    uint32_t strideX2[10];
    uint32_t strideY[10];

};
}
                
