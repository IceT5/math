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
 * \file floor_div.cpp
 * \brief
*/
#include "floor_div.h"

enum class FloorDivTilingKey : uint32_t
{
    TILING_KEY_EXAMPLE_FLOAT = 0,
    TILING_KEY_EXAMPLE_INT32 = 1,
    TILING_KEY_EXAMPLE_INT16 = 2,
    TILING_KEY_EXAMPLE_HALF = 3,
};

template <uint32_t schMode>
__global__ __aicore__ void floor_div(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(FloorDivTilingData);
    GET_TILING_DATA_WITH_STRUCT(FloorDivTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(FloorDivTilingKey::TILING_KEY_EXAMPLE_FLOAT)) {
        NsFloorDiv::FloorDiv<float> op; // 算子kernel实例获取
        op.Init(x, y, z, &tilingData);      // 算子kernel实例初始化
        op.Process();                       // 算子kernel实例执行
    }
    else if constexpr (schMode == static_cast<uint32_t>(FloorDivTilingKey::TILING_KEY_EXAMPLE_INT32)) {
        NsFloorDiv::FloorDiv<int32_t> op; // 算子kernel实例获取
        op.Init(x, y, z, &tilingData);        // 算子kernel实例初始化
        op.Process();                         // 算子kernel实例执行
    }
    else if constexpr (schMode == static_cast<uint32_t>(FloorDivTilingKey::TILING_KEY_EXAMPLE_INT16)) {
        NsFloorDiv::FloorDiv<int16_t> op; // 算子kernel实例获取
        op.Init(x, y, z, &tilingData);      // 算子kernel实例初始化
        op.Process();                       // 算子kernel实例执行
    }
    else if constexpr (schMode == static_cast<uint32_t>(FloorDivTilingKey::TILING_KEY_EXAMPLE_HALF)) {
        NsFloorDiv::FloorDiv<half> op; // 算子kernel实例获取
        op.Init(x, y, z, &tilingData);        // 算子kernel实例初始化
        op.Process();                         // 算子kernel实例执行
    }
}
