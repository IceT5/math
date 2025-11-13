/**
 * FloorDiv core math, shared by AiCore and optional AICPU.
 */
#pragma once
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace FloorDivCore {

template <typename T>
struct is_integer_type
    : std::integral_constant<bool,
                             std::is_integral<T>::value &&
                             !std::is_same<T, bool>::value> {};

template <typename T>
__attribute__((always_inline)) inline T FloorDivScalarIntImpl(T x, T y)
{
    T q = x / y;        // trunc toward 0
    T r = x - q * y;
    if (r != 0) {
        bool rPos = (r > 0);
        bool yPos = (y > 0);
        if (rPos != yPos) { q -= static_cast<T>(1); }
    }
    return q;
}

template <typename T>
__attribute__((always_inline)) inline T FloorDivScalarUnsignedImpl(T x, T y)
{
    return (y == 0) ? 0 : static_cast<T>(x / y);
}

template <typename T>
__attribute__((always_inline)) inline T FloorDivScalarFloatImpl(T x, T y)
{
    double xd = static_cast<double>(x);
    double yd = static_cast<double>(y);
    double q  = std::floor(xd / yd);
    return static_cast<T>(q);
}

template <typename T>
__attribute__((always_inline)) inline T FloorDivScalar(T x, T y)
{
    if constexpr (std::is_floating_point<T>::value) {
        return FloorDivScalarFloatImpl<T>(x, y);
    } else if constexpr (std::is_unsigned<T>::value) {
        return FloorDivScalarUnsignedImpl<T>(x, y);
    } else if constexpr (is_integer_type<T>::value) {
        return FloorDivScalarIntImpl<T>(x, y);
    } else {
        return static_cast<T>(0);
    }
}

template <typename T>
inline void FloorDivArray(const T* __restrict x,
                          const T* __restrict y,
                          T* __restrict out,
                          uint32_t n)
{
    for (uint32_t i = 0; i < n; ++i) {
        out[i] = FloorDivScalar<T>(x[i], y[i]);
    }
}

} // namespace FloorDivCore