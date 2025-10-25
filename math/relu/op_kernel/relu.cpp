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
 * \file relu.cpp
 * \brief
 */

#include "relu.h"

enum class ReluTilingKey : uint32_t
{
    TILING_KEY_FLOAT = 0,
    TILING_KEY_FLOAT16 = 1,
    TILING_KEY_INT32 = 2,
    TILING_KEY_INT16 = 3,
    TILING_KEY_INT8 = 4,
};

template <uint32_t schMode>
__global__ __aicore__ void relu(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReluTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReluTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(ReluTilingKey::TILING_KEY_FLOAT)) {
        NsRelu::Relu<float> op; // 算子kernel实例获取
        op.Init(x, z, tilingData);      // 算子kernel实例初始化
        op.Process();                       // 算子kernel实例执行
    }
    if constexpr (schMode == static_cast<uint32_t>(ReluTilingKey::TILING_KEY_FLOAT16)) {
        NsRelu::Relu<half> op; // 算子kernel实例获取
        op.Init(x, z, tilingData);        // 算子kernel实例初始化
        op.Process();                         // 算子kernel实例执行
    }
    if constexpr (schMode == static_cast<uint32_t>(ReluTilingKey::TILING_KEY_INT32)) {
        NsRelu::Relu<int32_t> op; // 算子kernel实例获取
        op.Init(x, z, tilingData);        // 算子kernel实例初始化
        op.Process();                         // 算子kernel实例执行
    }
    if constexpr (schMode == static_cast<uint32_t>(ReluTilingKey::TILING_KEY_INT16)) {
        NsRelu::Relu<int16_t> op; // 算子kernel实例获取
        op.Init(x, z, tilingData);        // 算子kernel实例初始化
        op.Process();                         // 算子kernel实例执行
    }
    if constexpr (schMode == static_cast<uint32_t>(ReluTilingKey::TILING_KEY_INT8)) {
        NsRelu::Relu<int8_t> op; // 算子kernel实例获取
        op.Init(x, z, tilingData);        // 算子kernel实例初始化
        op.Process();                         // 算子kernel实例执行
    }
}
