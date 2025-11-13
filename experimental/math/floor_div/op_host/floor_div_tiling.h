/*! \file floor_div_tiling.h */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_FLOOR_DIV_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_FLOOR_DIV_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(FloorDivTilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalLength);
TILING_DATA_FIELD_DEF(uint64_t, dtypeKey);
TILING_DATA_FIELD_DEF(uint32_t, alignNum);
TILING_DATA_FIELD_DEF(uint32_t, blockNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FloorDiv, FloorDivTilingData)

} // namespace optiling
#endif