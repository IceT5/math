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
 * \file power.cpp
 * \brief
*/

#include "power.h"

// 枚举类定义：每个类型组合对应唯一的调度键
enum class PowerTilingKey : uint32_t
{
    // INT8 行
    TILING_KEY_INT8_INT8 = 0,
    TILING_KEY_INT8_UINT8 = 1,
    TILING_KEY_INT8_INT16 = 2,
    TILING_KEY_INT8_INT32 = 3,
    TILING_KEY_INT8_FLOAT16 = 4,
    TILING_KEY_INT8_BF16 = 5,
    TILING_KEY_INT8_FLOAT32 = 6,

    // UINT8 行
    TILING_KEY_UINT8_INT8 = 7,
    TILING_KEY_UINT8_UINT8 = 8,
    TILING_KEY_UINT8_INT16 = 9,
    TILING_KEY_UINT8_INT32 = 10,
    TILING_KEY_UINT8_FLOAT16 = 11,
    TILING_KEY_UINT8_BF16 = 12,
    TILING_KEY_UINT8_FLOAT32 = 13,

    // INT16 行
    TILING_KEY_INT16_INT8 = 14,
    TILING_KEY_INT16_UINT8 = 15,
    TILING_KEY_INT16_INT16 = 16,
    TILING_KEY_INT16_INT32 = 17,
    TILING_KEY_INT16_FLOAT16 = 18,
    TILING_KEY_INT16_BF16 = 19,
    TILING_KEY_INT16_FLOAT32 = 20,

    // INT32 行
    TILING_KEY_INT32_INT8 = 21,
    TILING_KEY_INT32_UINT8 = 22,
    TILING_KEY_INT32_INT16 = 23,
    TILING_KEY_INT32_INT32 = 24,
    TILING_KEY_INT32_FLOAT16 = 25,
    TILING_KEY_INT32_BF16 = 26,
    TILING_KEY_INT32_FLOAT32 = 27,

    // FLOAT16 行
    TILING_KEY_FLOAT16_INT8 = 28,
    TILING_KEY_FLOAT16_UINT8 = 29,
    TILING_KEY_FLOAT16_INT16 = 30,
    TILING_KEY_FLOAT16_INT32 = 31,
    TILING_KEY_FLOAT16_FLOAT16 = 32,
    TILING_KEY_FLOAT16_BF16 = 33,
    TILING_KEY_FLOAT16_FLOAT32 = 34,

    // BF16 行
    TILING_KEY_BF16_INT8 = 35,
    TILING_KEY_BF16_UINT8 = 36,
    TILING_KEY_BF16_INT16 = 37,
    TILING_KEY_BF16_INT32 = 38,
    TILING_KEY_BF16_FLOAT16 = 39,
    TILING_KEY_BF16_BF16 = 40,
    TILING_KEY_BF16_FLOAT32 = 41,

    // FLOAT32 行
    TILING_KEY_FLOAT32_INT8 = 42,
    TILING_KEY_FLOAT32_UINT8 = 43,
    TILING_KEY_FLOAT32_INT16 = 44,
    TILING_KEY_FLOAT32_INT32 = 45,
    TILING_KEY_FLOAT32_FLOAT16 = 46,
    TILING_KEY_FLOAT32_BF16 = 47,
    TILING_KEY_FLOAT32_FLOAT32 = 48,


};


template <uint32_t schMode>
__global__ __aicore__ void power(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    // 注册并获取Tiling数据
    REGISTER_TILING_DEFAULT(PowerTilingData);
    GET_TILING_DATA_WITH_STRUCT(PowerTilingData, tilingData, tiling);

    
    if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_INT8_INT8)) {
        NsPower::Power<int8_t, int8_t, int8_t> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_INT8_UINT8)) {
        NsPower::Power<int8_t, uint8_t, int16_t> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_INT8_INT16)) {
        NsPower::Power<int8_t, int16_t, int16_t> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_INT8_INT32)) {
        NsPower::Power<int8_t, int32_t, int32_t> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_INT8_FLOAT16)) {
        NsPower::Power<int8_t, half, half> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_INT8_BF16)) {
        NsPower::Power<int8_t, bfloat16_t, bfloat16_t> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_INT8_FLOAT32)) {
        NsPower::Power<int8_t, float, float> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_UINT8_INT8)) {
        NsPower::Power<uint8_t, int8_t, int16_t> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_UINT8_UINT8)) {
        NsPower::Power<uint8_t, uint8_t, uint8_t> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_UINT8_INT16)) {
        NsPower::Power<uint8_t, int16_t, int16_t> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_UINT8_INT32)) {
        NsPower::Power<uint8_t, int32_t, int32_t> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_UINT8_FLOAT16)) {
        NsPower::Power<uint8_t, half, half> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_UINT8_BF16)) {
        NsPower::Power<uint8_t, bfloat16_t, bfloat16_t> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_UINT8_FLOAT32)) {
        NsPower::Power<uint8_t, float, float> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_INT16_INT8)) {
        NsPower::Power<int16_t, int8_t, int16_t> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_INT16_UINT8)) {
        NsPower::Power<int16_t, uint8_t, int16_t> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_INT16_INT16)) {
        NsPower::Power<int16_t, int16_t, int16_t> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_INT16_INT32)) {
        NsPower::Power<int16_t, int32_t, int32_t> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_INT16_FLOAT16)) {
        NsPower::Power<int16_t, half, half> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_INT16_BF16)) {
        NsPower::Power<int16_t, bfloat16_t, bfloat16_t> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_INT16_FLOAT32)) {
        NsPower::Power<int16_t, float, float> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_INT32_INT8)) {
        NsPower::Power<int32_t, int8_t, int32_t> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_INT32_UINT8)) {
        NsPower::Power<int32_t, uint8_t, int32_t> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_INT32_INT16)) {
        NsPower::Power<int32_t, int16_t, int32_t> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_INT32_INT32)) {
        NsPower::Power<int32_t, int32_t, int32_t> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_INT32_FLOAT16)) {
        NsPower::Power<int32_t, half, half> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_INT32_BF16)) {
        NsPower::Power<int32_t, bfloat16_t, bfloat16_t> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_INT32_FLOAT32)) {
        NsPower::Power<int32_t, float, float> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_FLOAT16_INT8)) {
        NsPower::Power<half, int8_t, half> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_FLOAT16_UINT8)) {
        NsPower::Power<half, uint8_t, half> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_FLOAT16_INT16)) {
        NsPower::Power<half, int16_t, half> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_FLOAT16_INT32)) {
        NsPower::Power<half, int32_t, half> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_FLOAT16_FLOAT16)) {
        NsPower::Power<half, half, half> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_FLOAT16_BF16)) {
        NsPower::Power<half, bfloat16_t, float> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_FLOAT16_FLOAT32)) {
        NsPower::Power<half, float, float> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_BF16_INT8)) {
        NsPower::Power<bfloat16_t, int8_t, bfloat16_t> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_BF16_UINT8)) {
        NsPower::Power<bfloat16_t, uint8_t, bfloat16_t> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_BF16_INT16)) {
        NsPower::Power<bfloat16_t, int16_t, bfloat16_t> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_BF16_INT32)) {
        NsPower::Power<bfloat16_t, int32_t, bfloat16_t> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_BF16_FLOAT16)) {
        NsPower::Power<bfloat16_t, half, float> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_BF16_BF16)) {
        NsPower::Power<bfloat16_t, bfloat16_t, bfloat16_t> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_BF16_FLOAT32)) {
        NsPower::Power<bfloat16_t, float, float> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_FLOAT32_INT8)) {
        NsPower::Power<float, int8_t, float> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_FLOAT32_UINT8)) {
        NsPower::Power<float, uint8_t, float> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_FLOAT32_INT16)) {
        NsPower::Power<float, int16_t, float> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_FLOAT32_INT32)) {
        NsPower::Power<float, int32_t, float> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_FLOAT32_FLOAT16)) {
        NsPower::Power<float, half, float> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_FLOAT32_BF16)) {
        NsPower::Power<float, bfloat16_t, float> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    else if constexpr (schMode == static_cast<uint32_t>(PowerTilingKey::TILING_KEY_FLOAT32_FLOAT32)) {
        NsPower::Power<float, float, float> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
    
}