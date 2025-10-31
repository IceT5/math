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
 * \file soft_plus_v2.cpp
 * \brief
 */
#include "soft_plus_v2.h"

extern "C" __global__ __aicore__ void soft_plus_v2(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SoftPlusV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(SoftPlusV2TilingData, tilingData, tiling);
    NsSoftPlusV2::SoftPlusV2 op;
    op.Init(x, z, &tilingData);
    op.Process();
}