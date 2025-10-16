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

#ifndef _GE_COSH_REGBASE_TILING_H_
#define _GE_COSH_REGBASE_TILING_H_

#include <cstdint>
#include <cstring>
#include "kernel_tiling/kernel_tiling.h"

#define __CCE_UT_TEST__

#pragma pack(1)

struct EleBaseTilingData {
    int64_t totalLength;
    int32_t coreNum;

    // 核间划分
    int32_t core_element_start[MAX_USE_CORE_NUM];
    int32_t core_element_end[MAX_USE_CORE_NUM];
    int32_t core_element_count[MAX_USE_CORE_NUM];

    // 核内划分
    int32_t tile_element_num;
    int32_t core_loop_times[MAX_USE_CORE_NUM];
    int32_t core_tail_elements[MAX_USE_CORE_NUM];
};

struct CoshRegbaseTilingData {
    EleBaseTilingData baseTiling;
};

#pragma pack()

inline void InitTilingData(uint8_t* tiling, CoshRegbaseTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(CoshRegbaseTilingData));
}

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData(tiling_arg, &tiling_data)

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    CoshRegbaseTilingData tiling_data;       \
    InitTilingData(tiling_arg, &tiling_data)
#endif // _GE_COSH_REGBASE_TILING_H_