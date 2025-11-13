/*! \file floor_div_tiling.cpp */
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "util/math_util.h"
#include "log/log.h"
#include "floor_div_tiling.h"

namespace optiling {

constexpr uint64_t KEY_FP32  = 1UL;
constexpr uint64_t KEY_FP16  = 2UL;
constexpr uint64_t KEY_BF16  = 3UL;
constexpr uint64_t KEY_INT32 = 4UL;
constexpr uint64_t KEY_INT16 = 5UL;
constexpr uint64_t KEY_INT8  = 6UL;
constexpr uint64_t KEY_UINT8 = 7UL;

static inline uint64_t MakeDtypeKey(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT:   return KEY_FP32;
        case ge::DT_FLOAT16: return KEY_FP16;
        case ge::DT_BF16:    return KEY_BF16;
        case ge::DT_INT32:   return KEY_INT32;
        case ge::DT_INT16:   return KEY_INT16;
        case ge::DT_INT8:    return KEY_INT8;
        case ge::DT_UINT8:   return KEY_UINT8;
        default:             return KEY_FP32;
    }
}

static ge::GraphErrCodeStatus FloorDivTilingFunc(gert::TilingContext* context)
{
    const ge::TensorDesc* xDesc = context->GetInputDesc(0);
    const ge::TensorDesc* yDesc = context->GetInputDesc(1);
    (void)yDesc;

    ge::Shape outShape = context->GetOutputDesc(0)->GetShape();
    uint64_t totalLen64 = 1;
    for (auto d : outShape.GetDims()) {
        totalLen64 *= static_cast<uint64_t>(d <= 0 ? 1 : d);
    }
    uint32_t totalLen = static_cast<uint32_t>(totalLen64);

    FloorDivTilingData* td = context->GetTilingData<FloorDivTilingData>();
    td->set_totalLength(totalLen);
    td->set_dtypeKey(MakeDtypeKey(xDesc->GetDataType()));
    td->set_alignNum(16);
    td->set_blockNum(0);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {
REG_OP(FloorDiv)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(y, ge::TensorType::ALL())
    .OUTPUT(out, ge::TensorType::ALL());

optiling::TilingFuncRegistration FloorDivTilingReg(
    "FloorDiv", optiling::FloorDivTilingFunc, optiling::FloorDivTilingData::GetTilingDataInfo);
} // namespace ge