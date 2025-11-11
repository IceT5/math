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
 * \file cummax.cc
 * \brief
 */
#include "cummax_aicpu.h"
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include <algorithm>
#include <iostream>

using namespace std;
namespace {
const char *const kCumMax = "Cummax";
const uint32_t kInputNum = 1;
const uint32_t kOutputNum = 2;

#define CUMMAX_COMPUTE_CASE(DTYPE, TYPE, ITYPE, CTX)                                 \
  case ((DTYPE)): {                                                                  \
    uint32_t result = KERNEL_STATUS_OK;                                              \
    if ((ITYPE) == DT_INT32) {                                                       \
      result = CumMaxCompute<TYPE, int32_t>((CTX));                                  \
    } else if ((ITYPE) == DT_INT64) {                                                \
      result = CumMaxCompute<TYPE, int64_t>((CTX));                                  \
    } else {                                                                         \
      KERNEL_LOG_WARN("Cummax kernel indices data type [%u] not support.", (ITYPE)); \
      return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);                     \
    }                                                                                \
    if (result != static_cast<uint32_t>(KERNEL_STATUS_OK)) {                         \
      KERNEL_LOG_ERROR("CumMax kernel compute failed.");                             \
      return result;                                                                 \
    }                                                                                \
    break;                                                                           \
  }
} // namespace

namespace aicpu {
uint32_t CumMaxCpuKernel::Compute(CpuKernelContext &ctx) {
  std::vector<std::string> attr_names = {"dim"};
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum, attr_names),
                      "Cummax check input and output number failed.");
  Tensor *x = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(x->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get input[0] failed.")
  Tensor *y = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(y->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get input[0] failed.")
  Tensor *indices = ctx.Output(1);
  KERNEL_CHECK_NULLPTR(indices->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get input[1] failed.")

  DataType x_type = x->GetDataType();
  DataType ind_type = indices->GetDataType();
  // DT_FLOAT,DT_DOUBLE,DT_INT32,DT_UINT8,DT_INT16,DT_INT8,DT_INT64,DT_UINT16,DT_FLOAT16,DT_UINT32,DT_UINT64,DT_BFLOAT16
  switch (x_type) {
    CUMMAX_COMPUTE_CASE(DT_INT8, int8_t, ind_type, ctx)
    CUMMAX_COMPUTE_CASE(DT_INT16, int16_t, ind_type, ctx)
    CUMMAX_COMPUTE_CASE(DT_INT32, int32_t, ind_type, ctx)
    CUMMAX_COMPUTE_CASE(DT_INT64, int64_t, ind_type, ctx)
    CUMMAX_COMPUTE_CASE(DT_UINT8, uint8_t, ind_type, ctx)
    CUMMAX_COMPUTE_CASE(DT_UINT16, uint16_t, ind_type, ctx)
    CUMMAX_COMPUTE_CASE(DT_UINT32, uint32_t, ind_type, ctx)
    CUMMAX_COMPUTE_CASE(DT_UINT64, uint64_t, ind_type, ctx)
    CUMMAX_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ind_type, ctx)
    CUMMAX_COMPUTE_CASE(DT_FLOAT, float, ind_type, ctx)
    CUMMAX_COMPUTE_CASE(DT_DOUBLE, double, ind_type, ctx)
    CUMMAX_COMPUTE_CASE(DT_BFLOAT16, Eigen::bfloat16, ind_type, ctx)
  default:
    KERNEL_LOG_WARN("Cummax kernel x data type [%s] not support.", DTypeStr(x_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

template <typename T, typename D>
uint32_t CumMaxCpuKernel::CumMaxCompute(const CpuKernelContext &ctx) {
  Tensor *x = ctx.Input(0);
  Tensor *y = ctx.Output(0);
  Tensor *indices = ctx.Output(1);

  AttrValue *dim = ctx.GetAttr("dim");

  D *indices_data = reinterpret_cast<D*>(indices->GetData());
  T *x_data = reinterpret_cast<T*>(x->GetData());
  T *y_data = reinterpret_cast<T*>(y->GetData());

  int64_t dim_data = dim->GetInt();
  int64_t dims = x->GetTensorShape()->GetDims();
  KERNEL_CHECK_FALSE(dim_data >= -dims && dim_data < dims,
                     KERNEL_STATUS_PARAM_INVALID,
                     "dim should be in the range of [-x_dim, x_dim)");
  if (dim_data == -1) dim_data = dims - 1;
  vector<int64_t> d = x->GetTensorShape()->GetDimSizes();
  int64_t paralled_data_size = 512 * 1024;
  int64_t data_size = x->NumElements() * static_cast<int64_t>(sizeof(T));
  int64_t outer, inner, depth;
  outer = inner = depth = 1;
  for (int64_t i = 0; i < dims; i++) {
    if (i < dim_data) inner *= d[i];
    else if (i > dim_data) outer *= d[i];
    else depth = d[i];
  }
  if (data_size <= paralled_data_size) { // data_size <= paralled_data_size
    for (int64_t outer_index = 0; outer_index < outer; ++outer_index) {
      for (int64_t inner_index = 0; inner_index < inner; inner_index++) {
        T maxv;
        D idx = -1;
        for (int64_t depth_index = 0; depth_index < depth; depth_index++) {
          int64_t index = outer_index;
          index += inner_index * depth * outer;
          index += depth_index * outer;
          if (depth_index == 0) {
            maxv = x_data[index];
            idx = 0;
          } else if ((maxv <= x_data[index]) || (isnan(x_data[index]))) {
            maxv = x_data[index];
            idx = static_cast<D>(depth_index);
          }
          indices_data[index] = idx;
          y_data[index] = maxv;
        }
      }
    }
  } else {
    auto shard_cummax = [&](int64_t start, int64_t end) {
      for (int64_t outer_index = start; outer_index < end; ++outer_index) {
        for (int64_t inner_index = 0; inner_index < inner; inner_index++) {
          T maxv;
          D idx = -1;
          for (int64_t depth_index = 0; depth_index < depth; depth_index++) {
            int64_t index = outer_index;
            index += inner_index * depth * outer;
            index += depth_index * outer;
            if (depth_index == 0) {
              maxv = x_data[index];
              idx = 0;
            } else if ((maxv <= x_data[index]) || (isnan(x_data[index]))) {
              maxv = x_data[index];
              idx = static_cast<D>(depth_index);
            }
            indices_data[index] = idx;
            y_data[index] = maxv;
          }
        }
      }
    };
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(
        min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (max_core_num > outer) {
      max_core_num = outer;
    }
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(
        ctx, outer, outer / max_core_num, shard_cummax),
        "CumMax Compute failed.")
  }
  return static_cast<uint32_t>(KERNEL_STATUS_OK);
}
REGISTER_CPU_KERNEL(kCumMax, CumMaxCpuKernel);
} // namespace aicpu
