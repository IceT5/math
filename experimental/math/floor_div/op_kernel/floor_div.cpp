/*! \file floor_div.cpp */
#include "floor_div.h"
#include "register/tilingdata_base.h"

using namespace AscendC;

#define KEY_DTYPE_FP32  1UL
#define KEY_DTYPE_FP16  2UL
#define KEY_DTYPE_BF16  3UL
#define KEY_DTYPE_INT32 4UL
#define KEY_DTYPE_INT16 5UL
#define KEY_DTYPE_INT8  6UL
#define KEY_DTYPE_UINT8 7UL

struct FloorDivTilingData {
    uint32_t totalLength;
    uint64_t dtypeKey;
};

extern "C" __global__ __aicore__
void floor_div(GM_ADDR x, GM_ADDR y, GM_ADDR out, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    uint32_t totalLength = tilingData.totalLength;

    if (TILING_KEY_IS(KEY_DTYPE_FP32)) {
        FloorDivN::Kernel<float> op;
        op.Init(x, y, out, totalLength);
        op.Process();
    } else if (TILING_KEY_IS(KEY_DTYPE_FP16)) {
        FloorDivN::Kernel<half> op;
        op.Init(x, y, out, totalLength);
        op.Process();
    } else if (TILING_KEY_IS(KEY_DTYPE_BF16)) {
        FloorDivN::Kernel<bfloat16> op;
        op.Init(x, y, out, totalLength);
        op.Process();
    } else if (TILING_KEY_IS(KEY_DTYPE_INT32)) {
        FloorDivN::Kernel<int32_t> op;
        op.Init(x, y, out, totalLength);
        op.Process();
    } else if (TILING_KEY_IS(KEY_DTYPE_INT16)) {
        FloorDivN::Kernel<int16_t> op;
        op.Init(x, y, out, totalLength);
        op.Process();
    } else if (TILING_KEY_IS(KEY_DTYPE_INT8)) {
        FloorDivN::Kernel<int8_t> op;
        op.Init(x, y, out, totalLength);
        op.Process();
    } else if (TILING_KEY_IS(KEY_DTYPE_UINT8)) {
        FloorDivN::Kernel<uint8_t> op;
        op.Init(x, y, out, totalLength);
        op.Process();
    }
}