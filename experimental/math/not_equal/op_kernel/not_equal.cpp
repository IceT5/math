#include "kernel_operator.h"

using namespace AscendC;

template<typename T, typename... Ts>
struct is_one_of : std::false_type {};

template<typename T, typename U, typename... Ts>
struct is_one_of<T, U, Ts...> : std::conditional_t<std::is_same_v<T, U>, std::true_type, is_one_of<T, Ts...>> {};

template<typename T, typename... Ts>
constexpr bool is_one_of_v = is_one_of<T, Ts...>::value;

template<typename T>
__aicore__ inline constexpr T ceil_div(T x, T y)
{
    return (x - 1) / y + 1;
}

template<typename T>
__aicore__ inline constexpr T ceil_round(T x, T y)
{
    return ceil_div(x, y) * y;
}

template<typename T, typename U>
__aicore__ inline std::enable_if_t<std::is_same_v<T, half>> NotEqual(LocalTensor<U> y, LocalTensor<U> x1, LocalTensor<U> x2, int _)
{
    LocalTensor<uint8_t> y_uint8 = y.template ReinterpretCast<uint8_t>();
    Compare(y_uint8, x1, x2, CMPMODE::NE, ceil_round(_, static_cast<int>(256 / sizeof(U))));
    Duplicate(x1, static_cast<U>(1), _);
    Duplicate(x2, static_cast<U>(0), _);
    Select(x1, y_uint8, x1, x2, SELMODE::VSEL_TENSOR_TENSOR_MODE, _);
    Cast(y_uint8, x1, RoundMode::CAST_RINT, _);
}

template<typename T, typename U>
__aicore__ inline std::enable_if_t<std::is_same_v<T, float>> NotEqual(LocalTensor<U> y, LocalTensor<U> x1, LocalTensor<U> x2, int _)
{
    LocalTensor<uint8_t> y_uint8 = y.template ReinterpretCast<uint8_t>();
    LocalTensor<half> x1_half = x1.template ReinterpretCast<half>();
    LocalTensor<half> x2_half = x2.template ReinterpretCast<half>();
    Compare(y_uint8, x1, x2, CMPMODE::NE, ceil_round(_, static_cast<int>(256 / sizeof(U))));
    Duplicate(x1_half, static_cast<half>(1), _);
    Duplicate(x2_half, static_cast<half>(0), _);
    Select(x1_half, y_uint8, x1_half, x2_half, SELMODE::VSEL_TENSOR_TENSOR_MODE, _);
    Cast(y_uint8, x1_half, RoundMode::CAST_RINT, _);
}

template<typename T, typename U>
__aicore__ inline std::enable_if_t<std::is_same_v<T, int>> NotEqual(LocalTensor<U> y, LocalTensor<U> x1, LocalTensor<U> x2, int _)
{
    LocalTensor<uint8_t> y_uint8 = y.template ReinterpretCast<uint8_t>();
    LocalTensor<half> x1_half = x1.template ReinterpretCast<half>();
    LocalTensor<half> x2_half = x2.template ReinterpretCast<half>();
    Compare(y_uint8, x1, x2, CMPMODE::EQ, ceil_round(_, static_cast<int>(256 / sizeof(U))));
    Duplicate(x1_half, static_cast<half>(0), _);
    Duplicate(x2_half, static_cast<half>(1), _);
    Select(x1_half, y_uint8, x1_half, x2_half, SELMODE::VSEL_TENSOR_TENSOR_MODE, _);
    Cast(y_uint8, x1_half, RoundMode::CAST_RINT, _);
}

template<typename T, typename U>
__aicore__ inline std::enable_if_t<is_one_of_v<T, int8_t, uint8_t>> NotEqual(LocalTensor<U> y, LocalTensor<U> x1, LocalTensor<U> x2, int _)
{
    LocalTensor<uint8_t> y_uint8 = y.template ReinterpretCast<uint8_t>();
    Cast(y, x1.template ReinterpretCast<T>(), RoundMode::CAST_NONE, _);
    Cast(x1, x2.template ReinterpretCast<T>(), RoundMode::CAST_NONE, _);
    Compare(y_uint8, y, x1, CMPMODE::NE, ceil_round(_, static_cast<int>(256 / sizeof(U))));
    Duplicate(x1, static_cast<U>(1), _);
    Duplicate(x2, static_cast<U>(0), _);
    Select(x1, y_uint8, x1, x2, SELMODE::VSEL_TENSOR_TENSOR_MODE, _);
    Cast(y_uint8, x1, RoundMode::CAST_RINT, _);
}

template<typename T, typename U>
__aicore__ inline std::enable_if_t<std::is_same_v<T, bfloat16_t>> NotEqual(LocalTensor<U> y, LocalTensor<U> x1, LocalTensor<U> x2, int _)
{
    LocalTensor<uint8_t> y_uint8 = y.template ReinterpretCast<uint8_t>();
    LocalTensor<half> x1_half = x1.template ReinterpretCast<half>();
    LocalTensor<half> x2_half = x2.template ReinterpretCast<half>();
    Cast(y, x1.template ReinterpretCast<T>(), RoundMode::CAST_NONE, _);
    Cast(x1, x2.template ReinterpretCast<T>(), RoundMode::CAST_NONE, _);
    Compare(y_uint8, y, x1, CMPMODE::NE, ceil_round(_, static_cast<int>(256 / sizeof(U))));
    Duplicate(x1_half, static_cast<half>(1), _);
    Duplicate(x2_half, static_cast<half>(0), _);
    Select(x1_half, y_uint8, x1_half, x2_half, SELMODE::VSEL_TENSOR_TENSOR_MODE, _);
    Cast(y_uint8, x1_half, RoundMode::CAST_RINT, _);
}

template<typename T, typename U>
__aicore__ inline void not_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, NotEqualTilingData &tiling)
{
    int block_index = GetBlockIdx();
    int block_dim = GetBlockNum();
    constexpr int DATA_BLOCK_SIZE = 512 / sizeof(T);
    int compute_blocks = ceil_div(tiling.size, DATA_BLOCK_SIZE);
    int compute_start = compute_blocks * block_index / block_dim * DATA_BLOCK_SIZE;
    int compute_end = compute_blocks * (block_index + 1) / block_dim * DATA_BLOCK_SIZE;

    GlobalTensor<T> x1_global_tensor, x2_global_tensor, y_global_tensor;
    x1_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x1));
    x2_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x2));
    y_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y));

    TPipe t_pipe;
    TQue<TPosition::VECIN, 1> x1_t_que, x2_t_que;
    TQue<TPosition::VECOUT, 1> y_t_que;

    constexpr int MAX_TILE_SIZE = (30 << 10) / sizeof(U);
    t_pipe.InitBuffer(x1_t_que, 2, MAX_TILE_SIZE * sizeof(U));
    t_pipe.InitBuffer(x2_t_que, 2, MAX_TILE_SIZE * sizeof(U));
    t_pipe.InitBuffer(y_t_que, 2, MAX_TILE_SIZE * sizeof(U));

    for (int i = compute_start; i < compute_end; i += MAX_TILE_SIZE)
    {
        int _ = min(compute_end - i, MAX_TILE_SIZE);
        //
        {
            LocalTensor<T> x1 = x1_t_que.AllocTensor<T>();
            LocalTensor<T> x2 = x2_t_que.AllocTensor<T>();
            DataCopy(x1, x1_global_tensor[i], _);
            DataCopy(x2, x2_global_tensor[i], _);
            x1_t_que.EnQue(x1);
            x2_t_que.EnQue(x2);
        }
        //
        {
            LocalTensor<U> x1 = x1_t_que.DeQue<U>();
            LocalTensor<U> x2 = x2_t_que.DeQue<U>();
            LocalTensor<U> y = y_t_que.AllocTensor<U>();
            NotEqual<T, U>(y, x1, x2, _);
            x1_t_que.FreeTensor(x1);
            x2_t_que.FreeTensor(x2);
            y_t_que.EnQue(y);
        }
        //
        {
            LocalTensor<T> y = y_t_que.DeQue<T>();
            DataCopy(y_global_tensor[i], y, _);
            y_t_que.FreeTensor(y);
        }
    }
}

template<typename T>
__aicore__ inline void not_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, NotEqualTilingData &tiling)
{
    if constexpr (is_one_of_v<T, half, float, int>)
        not_equal<T, T>(x1, x2, y, tiling);
    else if constexpr (is_one_of_v<T, int8_t, uint8_t>)
        not_equal<T, half>(x1, x2, y, tiling);
    else if constexpr (std::is_same_v<T, bool>)
        not_equal<uint8_t, half>(x1, x2, y, tiling);
    else if constexpr (std::is_same_v<T, bfloat16_t>)
        not_equal<T, float>(x1, x2, y, tiling);
}

extern "C" __global__ __aicore__ void not_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling_data, tiling);
    not_equal<DTYPE_X1>(x1, x2, y, tiling_data);
}
