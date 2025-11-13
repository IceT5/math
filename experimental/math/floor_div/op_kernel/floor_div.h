/*! \file floor_div.h */
#pragma once
#include "kernel_operator.h"
#include "floor_div_core.h"

using namespace AscendC;

namespace FloorDivN {

template <typename T>
class Kernel {
public:
    __aicore__ inline Kernel() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR out, uint32_t totalLen)
    {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));
        totalLength_ = totalLen;
    }

    __aicore__ inline void Process()
    {
        uint32_t coreId = GetBlockIdx();
        uint32_t coreNum = GetBlockNum();

        uint32_t lenPerCore = (totalLength_ + coreNum - 1) / coreNum;
        uint32_t start = coreId * lenPerCore;
        uint32_t end   = totalLength_ < (start + lenPerCore) ? totalLength_ : (start + lenPerCore);
        if (start >= end) { return; }

        constexpr uint32_t TILE = 1024; // tune per UB
        TPipe pipe;
        TQue<QuePosition::VECIN,  1> inQx;
        TQue<QuePosition::VECIN,  1> inQy;
        TQue<QuePosition::VECOUT, 1> outQ;

        pipe.InitBuffer(inQx, TILE * sizeof(T));
        pipe.InitBuffer(inQy, TILE * sizeof(T));
        pipe.InitBuffer(outQ, TILE * sizeof(T));

        for (uint32_t off = start; off < end; off += TILE) {
            uint32_t cur = (end - off) < TILE ? (end - off) : TILE;

            // GM -> UB
            {
                LocalTensor<T> xUb = inQx.AllocTensor<T>();
                LocalTensor<T> yUb = inQy.AllocTensor<T>();
                DataCopy(xUb, xGm_[off], cur * sizeof(T));
                DataCopy(yUb, yGm_[off], cur * sizeof(T));
                inQx.EnQue(xUb);
                inQy.EnQue(yUb);
            }

            // compute
            LocalTensor<T> xUb = inQx.DeQue<T>();
            LocalTensor<T> yUb = inQy.DeQue<T>();
            LocalTensor<T> oUb = outQ.AllocTensor<T>();

            T* xPtr = reinterpret_cast<T*>(xUb.GetVirAddr());
            T* yPtr = reinterpret_cast<T*>(yUb.GetVirAddr());
            T* oPtr = reinterpret_cast<T*>(oUb.GetVirAddr());
            FloorDivCore::FloorDivArray<T>(xPtr, yPtr, oPtr, cur);

            inQx.FreeTensor(xUb);
            inQy.FreeTensor(yUb);

            // UB -> GM
            DataCopy(outGm_[off], oUb, cur * sizeof(T));
            outQ.FreeTensor(oUb);
        }
    }

private:
    GmTensor<T> xGm_;
    GmTensor<T> yGm_;
    GmTensor<T> outGm_;
    uint32_t totalLength_ {0};
};

} // namespace FloorDivN