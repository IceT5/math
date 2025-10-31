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
 * \file ceil_v2_tiling_data.h
 * \brief tiling data struct
*/

#ifndef _ROTARY_POSITION_EMBEDDING_GRAD_TILING_DATA_H_
#define _ROTARY_POSITION_EMBEDDING_GRAD_TILING_DATA_H_

constexpr int32_t MAX_USE_CORE_NUM = 32;  // 设置合理的最大核数

struct CeilV2TilingData {
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
#endif