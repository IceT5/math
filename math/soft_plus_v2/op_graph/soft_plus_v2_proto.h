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
 * \file soft_plus_v2_proto.h
 * \brief Declaration of the SoftPlusV2 operator prototype, which defines the interface and attributes of the SoftPlusV2 operator.
 * 
 * SoftPlusV2 is a smooth approximation of the ReLU activation function, 
 * commonly used in neural networks to introduce non-linearity while avoiding 
 * the dying ReLU problem.
 */
#ifndef OPS_OP_PROTO_INC_SOFTPLUSV2_H_
#define OPS_OP_PROTO_INC_SOFTPLUSV2_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Computes the SoftPlus activation function: y = ln(1 + e^x)
* 
* SoftPlusV2 applies the softplus function element-wise to the input tensor.
* The function is a smooth alternative to the rectified linear unit (ReLU),
* with the mathematical property that its derivative is the sigmoid function.
* 
*@par Mathematical Formula:
* y = log(1 + exp(x))
* Where exp(x) is the exponential function, and log is the natural logarithm.
* 
*@par Inputs:
*One required input:
* @li x: A tensor of type float32 or float16, with any valid shape (e.g., NCHW or NHWC format for image data). 
*        Represents the input features to be activated.
*
*@par Outputs:
*y: A tensor with the same shape and data type as input 'x'. 
*   Each element is the result of applying the softplus function to the corresponding element in 'x'.
*
*@par Third-party framework compatibility
*Compatible with TensorFlow's Softplus operator (consistent in mathematical behavior).
*
*@par Constraints:
* - Input and output tensors must have the same data type (float32 or float16).
* - For large positive values of x, the output approximates x (since exp(x) dominates 1, so log(exp(x)) ≈ x).
* - For large negative values of x, the output approximates 0 (since exp(x) becomes negligible, so log(1) ≈ 0).
*/
REG_OP(SoftPlusV2)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))  // Input tensor for softplus activation
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16})) // Output tensor after applying softplus
    .OP_END_FACTORY_REG(SoftPlusV2)

} // namespace ge

#endif // OPS_OP_PROTO_INC_SOFTPLUSV2_H_