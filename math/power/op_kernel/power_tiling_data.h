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
 * \file power_tiling_data.h
 * \brief tiling data struct
*/
#ifndef POWER_CUSTOM_TILING_H
#define POWER_CUSTOM_TILING_H

struct PowerTilingData {
    int64_t smallCoreDataNum;
    int64_t bigCoreDataNum;
    int64_t finalBigTileNum;
    int64_t finalSmallTileNum;
    int64_t tileDataNum;
    int64_t smallTailDataNum;
    int64_t bigTailDataNum;
    int64_t tailBlockNum;
    int64_t x_dtype;
    int64_t y_dtype;
    int64_t z_dtype;
    int64_t is_input0_scalar;
    int64_t is_input1_scalar;
    int64_t x1Shape[10];
    int64_t x2Shape[10];
    int64_t yShape[10];
    int64_t x1Dim;
    int64_t x2Dim;
    int64_t yDim;
    int64_t isSameX1;
    int64_t isSameX2;
    int64_t strideX1[10];
    int64_t strideX2[10];
    int64_t strideY[10];
};

#endif // POWER_CUSTOM_TILING_H