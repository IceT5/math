/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Liang Yanglin <@liang-yanglin>
 * - Liu Jun <@kbryantttt>
 * - Zhou Jianhua <@LePenseur>
 * - Tu Yuanhang <@TuYHAAAAAA>
 * - Li Xing <@li-xingHIT>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file transposev.cpp
 * \brief
 */

#include "transposev.h"

enum class TransposevTilingKey : uint32_t
{
    TILING_KEY_EXAMPLE_FLOAT    = 0,
    TILING_KEY_EXAMPLE_INT32    = 1,
    TILING_KEY_EXAMPLE_UINT32   = 2,
    TILING_KEY_EXAMPLE_INT64    = 3,
    TILING_KEY_EXAMPLE_UINT64   = 4,
    TILING_KEY_EXAMPLE_FLOAT16  = 5,
    TILING_KEY_EXAMPLE_INT16    = 6,
    TILING_KEY_EXAMPLE_UINT16   = 7,
    TILING_KEY_EXAMPLE_BF16     = 8,
    TILING_KEY_EXAMPLE_INT8     = 9,
    TILING_KEY_EXAMPLE_UINT8    = 10,
    TILING_KEY_EXAMPLE_BOOL     = 11,

    TILING_KEY_EXAMPLE2_FLOAT    = 12,
    TILING_KEY_EXAMPLE2_INT32    = 13,
    TILING_KEY_EXAMPLE2_UINT32   = 14,
    TILING_KEY_EXAMPLE2_INT64    = 15,
    TILING_KEY_EXAMPLE2_UINT64   = 16,
    TILING_KEY_EXAMPLE2_FLOAT16  = 17,
    TILING_KEY_EXAMPLE2_INT16    = 18,
    TILING_KEY_EXAMPLE2_UINT16   = 19,
    TILING_KEY_EXAMPLE2_BF16     = 20,
    TILING_KEY_EXAMPLE2_INT8     = 21,
    TILING_KEY_EXAMPLE2_UINT8    = 22,
    TILING_KEY_EXAMPLE2_BOOL     = 23,
};


template <uint32_t schMode>
__global__ __aicore__ void transposev(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TransposevTilingData);
    GET_TILING_DATA_WITH_STRUCT(TransposevTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(TransposevTilingKey::TILING_KEY_EXAMPLE_FLOAT)) {
        NsTransposev::Transposev<float, int32_t> op; 
        op.Init(x, y, z, &tilingData);      
        op.Process();                       
    }else if constexpr (schMode == static_cast<uint32_t>(TransposevTilingKey::TILING_KEY_EXAMPLE_INT32)) {
        NsTransposev::Transposev<int32_t, int32_t> op; 
        op.Init(x, y, z, &tilingData);        
        op.Process();
    }else if constexpr (schMode == static_cast<uint32_t>(TransposevTilingKey::TILING_KEY_EXAMPLE_UINT32)) {
        NsTransposev::Transposev<uint32_t, int32_t> op; 
        op.Init(x, y, z, &tilingData);        
        op.Process();                         
    }else if constexpr (schMode == static_cast<uint32_t>(TransposevTilingKey::TILING_KEY_EXAMPLE_INT64)) {
        NsTransposev::Transposev<int64_t, int32_t> op; 
        op.Init(x, y, z, &tilingData);        
        op.Process();                         
    }else if constexpr (schMode == static_cast<uint32_t>(TransposevTilingKey::TILING_KEY_EXAMPLE_UINT64)) {
        NsTransposev::Transposev<uint64_t, int32_t> op; 
        op.Init(x, y, z, &tilingData);        
        op.Process();                         
    }else if constexpr (schMode == static_cast<uint32_t>(TransposevTilingKey::TILING_KEY_EXAMPLE_FLOAT16)) {
        NsTransposev::Transposev<half, int32_t> op; 
        op.Init(x, y, z, &tilingData);        
        op.Process();                         
    }else if constexpr (schMode == static_cast<uint32_t>(TransposevTilingKey::TILING_KEY_EXAMPLE_INT16)) {
        NsTransposev::Transposev<int16_t, int32_t> op; 
        op.Init(x, y, z, &tilingData);        
        op.Process();                         
    }else if constexpr (schMode == static_cast<uint32_t>(TransposevTilingKey::TILING_KEY_EXAMPLE_UINT16)) {
        NsTransposev::Transposev<uint16_t, int32_t> op; 
        op.Init(x, y, z, &tilingData);        
        op.Process(); 
    }else if constexpr (schMode == static_cast<uint32_t>(TransposevTilingKey::TILING_KEY_EXAMPLE_BF16)) {
        NsTransposev::Transposev<half, int32_t> op; 
        op.Init(x, y, z, &tilingData);        
        op.Process();                         
    }else if constexpr (schMode == static_cast<uint32_t>(TransposevTilingKey::TILING_KEY_EXAMPLE_INT8)) {
        NsTransposev::Transposev<int8_t, int32_t> op; 
        op.Init(x, y, z, &tilingData);        
        op.Process();                         
    }else if constexpr (schMode == static_cast<uint32_t>(TransposevTilingKey::TILING_KEY_EXAMPLE_UINT8)) {
        NsTransposev::Transposev<uint8_t, int32_t> op; 
        op.Init(x, y, z, &tilingData);        
        op.Process();                         
    }else if constexpr (schMode == static_cast<uint32_t>(TransposevTilingKey::TILING_KEY_EXAMPLE_BOOL)) {
        NsTransposev::Transposev<bool, int32_t> op;    
        op.Init(x, y, z, &tilingData);        
        op.Process();                         
    }
    // x2,int64
    else if constexpr (schMode == static_cast<uint32_t>(TransposevTilingKey::TILING_KEY_EXAMPLE2_FLOAT)) {
        NsTransposev::Transposev<float, int64_t> op; 
        op.Init(x, y, z, &tilingData);      
        op.Process();                       
    }else if constexpr (schMode == static_cast<uint32_t>(TransposevTilingKey::TILING_KEY_EXAMPLE2_INT32)) {
        NsTransposev::Transposev<int32_t, int64_t> op; 
        op.Init(x, y, z, &tilingData);        
        op.Process();
    }else if constexpr (schMode == static_cast<uint32_t>(TransposevTilingKey::TILING_KEY_EXAMPLE2_UINT32)) {
        NsTransposev::Transposev<uint32_t, int64_t> op; 
        op.Init(x, y, z, &tilingData);        
        op.Process();                         
    }else if constexpr (schMode == static_cast<uint32_t>(TransposevTilingKey::TILING_KEY_EXAMPLE2_INT64)) {
        NsTransposev::Transposev<int64_t, int64_t> op; 
        op.Init(x, y, z, &tilingData);        
        op.Process();                         
    }else if constexpr (schMode == static_cast<uint32_t>(TransposevTilingKey::TILING_KEY_EXAMPLE2_UINT64)) {
        NsTransposev::Transposev<uint64_t, int64_t> op; 
        op.Init(x, y, z, &tilingData);        
        op.Process();                         
    }else if constexpr (schMode == static_cast<uint32_t>(TransposevTilingKey::TILING_KEY_EXAMPLE2_FLOAT16)) {
        NsTransposev::Transposev<half, int64_t> op; 
        op.Init(x, y, z, &tilingData);        
        op.Process();                         
    }else if constexpr (schMode == static_cast<uint32_t>(TransposevTilingKey::TILING_KEY_EXAMPLE2_INT16)) {
        NsTransposev::Transposev<int16_t, int64_t> op; 
        op.Init(x, y, z, &tilingData);        
        op.Process();                         
    }else if constexpr (schMode == static_cast<uint32_t>(TransposevTilingKey::TILING_KEY_EXAMPLE2_UINT16)) {
        NsTransposev::Transposev<uint16_t, int64_t> op; 
        op.Init(x, y, z, &tilingData);        
        op.Process(); 
    }else if constexpr (schMode == static_cast<uint32_t>(TransposevTilingKey::TILING_KEY_EXAMPLE2_BF16)) {
        NsTransposev::Transposev<half, int64_t> op; 
        op.Init(x, y, z, &tilingData);        
        op.Process();                         
    }else if constexpr (schMode == static_cast<uint32_t>(TransposevTilingKey::TILING_KEY_EXAMPLE2_INT8)) {
        NsTransposev::Transposev<int8_t, int64_t> op; 
        op.Init(x, y, z, &tilingData);        
        op.Process();                         
    }else if constexpr (schMode == static_cast<uint32_t>(TransposevTilingKey::TILING_KEY_EXAMPLE2_UINT8)) {
        NsTransposev::Transposev<uint8_t, int64_t> op; 
        op.Init(x, y, z, &tilingData);        
        op.Process();                         
    }else if constexpr (schMode == static_cast<uint32_t>(TransposevTilingKey::TILING_KEY_EXAMPLE2_BOOL)) {
        NsTransposev::Transposev<bool, int64_t> op;    
        op.Init(x, y, z, &tilingData);        
        op.Process();                         
    }
}