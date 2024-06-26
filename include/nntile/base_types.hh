/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/base_types.hh
 * Base integer and floating point types.
 *
 * @version 1.0.0
 * */

#pragma once

#include <cstdint>
#include <assert.h>
#include <iostream>

namespace nntile
{

//! Enumeration of base supported data types inside tiles and tensors
enum class DTypeEnum
{
    INT64=101,
    INT32,
    INT16,
    INT8,
    BOOL,
    FP64,
    FP32,
    FP32_FAST_TF32,
    TF32=FP32_FAST_TF32,
    FP16,
    BF16,
    FP8_E4M3,
    FP8_E5M2
};

using scal_t = float;

template<typename T, typename Y>
inline constexpr Y T2Y(const T &other)
{
    return static_cast<Y>(other);
}

template<typename T, size_t nbytes, enum DTypeEnum dtype_enum_,
    typename T_compat, T compat2T(const T_compat &)=T2Y<T_compat, T>,
    T_compat T2compat(const T &)=T2Y<T, T_compat>>
class DTypeBase
{
public:
    T value;
    using compat_t = T_compat;
    static constexpr enum DTypeEnum dtype_enum = dtype_enum_;
    constexpr DTypeBase(const T_compat &other):
        value(compat2T(other))
    {
        static_assert(sizeof(T) == nbytes);
        static_assert(sizeof(*this) == nbytes);
    }
    constexpr DTypeBase &operator=(const T_compat &other)
    {
        value = comapt2T(other);
        return *this;
    }
    constexpr DTypeBase &operator=(const DTypeBase &other)
    {
        value = other.value;
        return *this;
    }
//    constexpr bool operator==(const scal_t &other) const
//    {
//        return (value == scal2T(other));
//    }
//    constexpr bool operator==(const DTypeBase &other) const
//    {
//        return (value == other.value);
//    }
//    constexpr bool operator!=(const scal_t &other) const
//    {
//        return (value != scal2T(other));
//    }
//    constexpr bool operator!=(const DTypeBase &other) const
//    {
//        return (value != other.value);
//    }
    constexpr T_compat get() const
    {
        return T2compat(value);
    }
    constexpr DTypeBase &set(const T_compat &other)
    {
        value = compat2T(other);
        return *this;
    }
};

using fp64_t = DTypeBase<double, 8, DTypeEnum::FP64, double>;

using fp32_t = DTypeBase<float, 4, DTypeEnum::FP32, float>;

using fp32_fast_tf32_t = DTypeBase<float, 4, DTypeEnum::FP32_FAST_TF32, float>;

inline constexpr int16_t float2fp16(const float &other)
{
    // Fake FP32 to FP16 convertor
    int16_t res = 0;
    return res;
}

inline constexpr float fp162float(const int16_t &other)
{
    // Fake FP16 to FP32 convertor
    float res = 0;
    return res;
}

using fp16_t = DTypeBase<int16_t, 2, DTypeEnum::FP16, float, float2fp16,
      fp162float>;

//! Large enough signed integer for indexing purposes
using Index = int64_t;

// Boolean type for mask
using bool_t = bool;

// Overload for printing fp64_t
static std::ostream &operator<<(std::ostream &cout, fp64_t val)
{
    // Convert value to scal_t and print it
    cout << val.get();
    return cout;
}

// Overload for printing fp32_t
static std::ostream &operator<<(std::ostream &cout, fp32_t val)
{
    // Convert value to scal_t and print it
    cout << val.get();
    return cout;
}

// Overload for printing fp32_fast_tf32_t
static std::ostream &operator<<(std::ostream &cout, fp32_fast_tf32_t val)
{
    // Convert value to scal_t and print it
    cout << val.get();
    return cout;
}

// Overload for printing fp16_t
static std::ostream &operator<<(std::ostream &cout, fp16_t val)
{
    // Convert value to scal_t and print it
    cout << "FP16 is not yet printable";
    return cout;
}

} // namespace nntile
