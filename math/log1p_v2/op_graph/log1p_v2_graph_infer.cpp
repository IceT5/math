/**
 * Copyright(c) Huawei Technologies Co., Ltd.2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License");
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file log1p_v2_graph_infer.cpp
 * \brief log1p_v2 operater graph infer resource
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
using namespace ge;

static constexpr int64_t IDX_0 = 0;

static ge::graphStatus InferDataTypeLog1pV2(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeLog1pV2");

    // 设置输出的dtype Log1pV2算子逻辑是两个数相除，因此输出dataType与输入dataType一致
    ge::DataType sizeDtype = context->GetInputDataType(IDX_0);
    context->SetOutputDataType(IDX_0, sizeDtype);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeLog1pV2");
    return GRAPH_SUCCESS;
}

IMPL_OP(Log1pV2).InferDataType(InferDataTypeLog1pV2);

}; // namespace ops
