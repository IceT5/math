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
 * \file cummin.cc
 * \brief
 */
#include "cummin_aicpu.h"
#include <algorithm>
#include <iostream>
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

using namespace std;
namespace {
const char *kCumMin = "Cummin";
const uint32_t kInputNum = 1;
const uint32_t kOutputNum = 2;

#define CUMMIN_COMPUTE_CASE(DTYPE, TYPE, ITYPE, CTX)                       \
  case ((DTYPE)): {                                                        \
    uint32_t result = KERNEL_STATUS_OK;                                    \
    if ((ITYPE) == DT_INT32) {                                             \
      result = CumMinCompute<TYPE, int32_t>((CTX));                        \
    } else if ((ITYPE) == DT_INT64) {                                      \
      result = CumMinCompute<TYPE, int64_t>((CTX));                        \
    } else {                                                               \
      KERNEL_LOG_WARN("Cummin kernel indices data type [%u] not support.", \
                      (ITYPE));                                            \
      return KERNEL_STATUS_PARAM_INVALID;                                  \
    }                                                                      \
    if (result != KERNEL_STATUS_OK) {                                      \
      KERNEL_LOG_ERROR("CumMin kernel compute failed.");                   \
      return result;                                                       \
    }                                                                      \
    break;                                                                 \
  }
}  // namespace

namespace aicpu {
uint32_t CumMinCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "Cummin check input and output number failed.");
  DataType x_type = ctx.Input(0)->GetDataType();
  DataType ind_type = ctx.Output(1)->GetDataType();
  // DT_FLOAT,DT_DOUBLE,DT_INT32,DT_UINT8,DT_INT16,DT_INT8,DT_INT64,DT_UINT16,DT_FLOAT16,DT_UINT32,DT_UINT64
  switch (x_type) {
    CUMMIN_COMPUTE_CASE(DT_INT8, int8_t, ind_type, ctx)
    CUMMIN_COMPUTE_CASE(DT_INT16, int16_t, ind_type, ctx)
    CUMMIN_COMPUTE_CASE(DT_INT32, int32_t, ind_type, ctx)
    CUMMIN_COMPUTE_CASE(DT_INT64, int64_t, ind_type, ctx)
    CUMMIN_COMPUTE_CASE(DT_UINT8, uint8_t, ind_type, ctx)
    CUMMIN_COMPUTE_CASE(DT_UINT16, uint16_t, ind_type, ctx)
    CUMMIN_COMPUTE_CASE(DT_UINT32, uint32_t, ind_type, ctx)
    CUMMIN_COMPUTE_CASE(DT_UINT64, uint64_t, ind_type, ctx)
    CUMMIN_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ind_type, ctx)
    CUMMIN_COMPUTE_CASE(DT_FLOAT, float, ind_type, ctx)
    CUMMIN_COMPUTE_CASE(DT_DOUBLE, double, ind_type, ctx)
    default:
      KERNEL_LOG_ERROR("Cummin kernel x data type [%s] not support.",
                       DTypeStr(x_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T, typename D>
uint32_t SingleCoreCal(CpuKernelContext &ctx, int64_t outer, int64_t inner,
                       int64_t depth) {
  Tensor *x = ctx.Input(0);
  Tensor *y = ctx.Output(0);
  Tensor *indices = ctx.Output(1);
  D *indices_data = reinterpret_cast<D *>(indices->GetData());
  T *x_data = reinterpret_cast<T *>(x->GetData());
  T *y_data = reinterpret_cast<T *>(y->GetData());
  for (int64_t outer_index = 0; outer_index < outer; ++outer_index) {
    for (int64_t inner_index = 0; inner_index < inner; inner_index++) {
      T minv;
      D idx = -1;
      for (int64_t depth_index = 0; depth_index < depth; depth_index++) {
        int64_t index = outer_index;
        index += inner_index * depth * outer;
        index += depth_index * outer;
        if (depth_index == 0) {
          minv = x_data[index];
          idx = 0;
        } else if ((minv >= x_data[index]) || (isnan(x_data[index]))) {
          minv = x_data[index];
          idx = depth_index;
        }
        indices_data[index] = idx;
        y_data[index] = minv;
      }
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T, typename D>
uint32_t MutilCoreCal(CpuKernelContext &ctx, int64_t outer, int64_t inner,
                      int64_t depth) {
  Tensor *x = ctx.Input(0);
  Tensor *y = ctx.Output(0);
  Tensor *indices = ctx.Output(1);
  D *indices_data = reinterpret_cast<D *>(indices->GetData());
  T *x_data = reinterpret_cast<T *>(x->GetData());
  T *y_data = reinterpret_cast<T *>(y->GetData());
  auto shard_cummin = [&](int64_t start, int64_t end) {
    for (int64_t outer_index = start; outer_index < end; ++outer_index) {
      for (int64_t inner_index = 0; inner_index < inner; inner_index++) {
        T minv;
        D idx = -1;
        for (int64_t depth_index = 0; depth_index < depth; depth_index++) {
          int64_t index = outer_index;
          index += inner_index * depth * outer;
          index += depth_index * outer;
          if (depth_index == 0) {
            minv = x_data[index];
            idx = 0;
          } else if ((minv >= x_data[index]) || (isnan(x_data[index]))) {
            minv = x_data[index];
            idx = depth_index;
          }
          indices_data[index] = idx;
          y_data[index] = minv;
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
  KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, outer, outer / max_core_num, shard_cummin),
                      "CumMin Compute failed.");
  return KERNEL_STATUS_OK;
}

template <typename T, typename D>
uint32_t CumMinCpuKernel::CumMinCompute(CpuKernelContext &ctx) {
  Tensor *x = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(ctx.GetAttr("axis"), KERNEL_STATUS_PARAM_INVALID,
                       "Cummin get axis failed.");
  AttrValue *dim = ctx.GetAttr("axis");
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
    if (i < dim_data)
      inner *= d[i];
    else if (i > dim_data)
      outer *= d[i];
    else
      depth = d[i];
  }
  if (data_size <= paralled_data_size) {
    return SingleCoreCal<T, D>(ctx, outer, inner, depth);
  } else {
    return MutilCoreCal<T, D>(ctx, outer, inner, depth);
  }
}
REGISTER_CPU_KERNEL(kCumMin, CumMinCpuKernel);
}  // namespace aicpu
