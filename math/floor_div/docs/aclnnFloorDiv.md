# aclnnFloorDiv

## 产品支持情况

| 产品  | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品 / Atlas 800I A2 推理产品 / A200I A2 Box 异构组件 | √   |

## 功能说明

- **算子功能**  
  对输入张量 `x1` 和输入张量 `x2` 相除得到`y`，并对`y`中每一个元素执行向下取整（不大于该值的最大整数）操作，并输出结果张量 `out`。
  
- **计算公式**
  
  $$
  out=⌊\frac{x1}{x2}⌋
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/context/%E4%B8%A4%E6%AE%B5%E5%BC%8F%E6%8E%A5%E5%8F%A3.md)，必须先调用“aclnnAbsGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAbs”接口执行计算。

```Cpp
aclnnStatus aclnnFloorDivGetWorkspaceSize(
  const aclTensor *self,
  aclTensor       *out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnFloorDiv(
  void              *workspace,
  uint64_t           workspaceSize,
  aclOpExecutor     *executor,
  const aclrtStream  stream)
```

## aclnnFloorDivGetWorkspaceSize

### 参数说明

| 参数名 | 输入/输出 | 描述  | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| x1 | 输入  | 待进行 `floor_div` 计算的输入张量。公式中的 `x1`。 | 无   | FLOAT32、FLOAT16、INT32、INT8、UINT8、BFLOAT16 | ND  | 0-8 | √   |
| x2 | 输入  | 待进行 `floor_div` 计算的输入张量。公式中的 `x2`。 | shape与`x1`相同   | FLOAT32、FLOAT16、INT32、INT8、UINT8、BFLOAT16 | ND  | 0-8 | √   |
| out | 输出  | `floor_div` 计算的输出张量，公式中的 `out`。 | shape 与 `x1` 相同 | FLOAT32、FLOAT16、INT32、INT8、UINT8、BFLOAT16 | ND  | 0-8 | √   |
| workspaceSize | 输出  | 返回 device 侧执行该算子时所需的 workspace 大小。 | -   | -   | -   | -   | -   |
| executor | 输出  | 返回 op 执行器，包含算子计算流程。 | -   | -   | -   | -   | -   |

---

### 返回值

`aclnnStatus`：返回状态码，具体参见 [aclnn返回码](https://poe.com/docs/context/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81.md)。

**第一段接口** 会完成入参检查，如以下场景时报错：

| 返回码 | 错误码 | 描述  |
| --- | --- | --- |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 传入的 tensor 是空指针。 |
| ACLNN_ERR_PARAM_INVALID | 161002 | self 数据类型或格式不在支持范围内。 |
|     |     | self 数据维度超过 8 维。 |
|     |     | self 与 out 的数据形状不一致。 |

## aclnnFloorDiv

### 参数说明

| 参数名 | 输入/输出 | 描述  |
| --- | --- | --- |
| workspace | 输入  | 在 Device 侧申请的 workspace 内存地址。 |
| workspaceSize | 输入  | workspace 的大小，由第一段接口 `aclnnFloorDivGetWorkspaceSize` 获取。 |
| executor | 输入  | op 执行器，包含算子计算流程。 |
| stream | 输入  | 指定执行任务的 Stream。 |

### 返回值

`aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81.md)。

## 约束说明

- 当前支持 **FLOAT32、FLOAT16、INT32、INT8、UINT8、BFLOAT16** 三种数据类型

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/%E7%BC%96%E8%AF%91%E4%B8%8E%E8%BF%90%E8%A1%8C%E6%A0%B7%E4%BE%8B.md)。

```Cpp
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

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_floor_div.h"
// 修改测试数据类型
using DataType = int32_t;
#define ACL_TYPE aclDataType::ACL_INT32
#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<DataType> resultData(size, 0);
    auto ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
         LOG_PRINT("mean result[%ld] is: ", i);       // float
         std::cout << resultData[i] << std::endl;
        //LOG_PRINT("mean result[%ld] is: %d\n", i, resultData[i]);       // int
    }
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 2. 申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 3. 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. 调用acl进行device/stream初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    aclTensor* selfX = nullptr;
    void* selfXDeviceAddr = nullptr;
    std::vector<int64_t> selfXShape = {1, 1, 3, 4};
    std::vector<DataType> selfXHostData(12);
    for(int i = 0; i < selfXHostData.size(); i++) {
        selfXHostData[i] = (DataType)(i - (int)selfXHostData.size() / 2);
    }
    ret = CreateAclTensor(selfXHostData, selfXShape, &selfXDeviceAddr, ACL_TYPE, &selfX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* selfY = nullptr;
    void* selfYDeviceAddr = nullptr;
    std::vector<int64_t> selfYShape = {1, 1, 3, 4};
    std::vector<DataType> selfYHostData(12, 2.0);
    ret = CreateAclTensor(selfYHostData, selfYShape, &selfYDeviceAddr, ACL_TYPE, &selfY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* out = nullptr;
    void* outDeviceAddr = nullptr;
    std::vector<int64_t> outShape = {1, 1, 3, 4};
    std::vector<DataType> outHostData(12, 300.0);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, ACL_TYPE, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    LOG_PRINT("Before GetWorkspaceSize: selfX=%p, selfY=%p, out=%p\n", (void*)selfX, (void*)selfY, (void*)out);
    LOG_PRINT("Before GetWorkspaceSize: selfXDeviceAddr=%p, selfYDeviceAddr=%p, outDeviceAddr=%p\n",
          selfXDeviceAddr, selfYDeviceAddr, outDeviceAddr);
    // 4. 调用aclnnAddExample第一段接口
    ret = aclnnFloorDivGetWorkspaceSize(selfX, selfY, out, &workspaceSize, &executor);
    LOG_PRINT("aclnnFloorDivGetWorkspaceSize returned %d, workspaceSize=%llu, executor=%p\n",
          ret, (unsigned long long)workspaceSize, (void*)executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFloorDivExampleGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 5. 调用aclnnAddExample第二段接口
    ret = aclnnFloorDiv(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulExample failed. ERROR: %d\n", ret); return ret);

    // 6. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    std::vector<int64_t> outShape1 = {12};
    PrintOutResult(outShape1, &outDeviceAddr);

    // 7. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(selfX);
    aclDestroyTensor(selfY);
    aclDestroyTensor(out);

    // 8. 释放device资源
    aclrtFree(selfXDeviceAddr);
    aclrtFree(selfYDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);

    // 9. acl去初始化
    aclFinalize();

    return 0;
}
```
