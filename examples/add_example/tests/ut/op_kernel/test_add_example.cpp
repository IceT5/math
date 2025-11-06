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

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "tiling_case_executor.h"
#include "test_add_example_tiling.h"

using namespace std;

extern "C" __global__ __aicore__ void add_example(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling);

class add_example_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "add_example_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "add_example_test TearDown\n" << endl;
    }
};

TEST_F(add_example_test, test_case_0)
{
    size_t xByteSize = 1 * 2 * 8 * 16 * sizeof(float);
    size_t yByteSize = 1 * 2 * 8 * 16 * sizeof(float);
    size_t zByteSize = 1 * 2 * 8 * 16 * sizeof(float);
    size_t tiling_data_size = sizeof(AddExampleTilingData);
    uint32_t blockDim = 8;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(zByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AddExampleTilingData* tilingDatafromBin = reinterpret_cast<AddExampleTilingData*>(tiling);

    tilingDatafromBin->totalLength = 1 * 2 * 8 * 16;
    tilingDatafromBin->tileNum = 8;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_example,
        blockDim,
        x,
        y,
        z,
        workspace,
        (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
