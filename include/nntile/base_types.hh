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

template<typename T, enum DTypeEnum dtype_enum_, T scal2T(const scal_t &),
    scal_t T2scal(const T &), size_t nbytes>
class DTypeBase
{
public:
    T value;
    static constexpr enum DTypeEnum dtype_enum = dtype_enum_;
    constexpr DTypeBase(const scal_t &other):
        value(scal2T(other))
    {
        static_assert(sizeof(T) == nbytes);
        static_assert(sizeof(*this) == nbytes);
    }
    constexpr DTypeBase &operator=(const scal_t &other)
    {
        value = scal2T(other);
        return *this;
    }
    constexpr DTypeBase &operator=(const DTypeBase &other)
    {
        value = other.value;
        return *this;
    }
    constexpr bool operator==(const scal_t &other) const
    {
        return (value == scal2T(other));
    }
    constexpr bool operator==(const DTypeBase &other) const
    {
        return (value == other.value);
    }
    constexpr bool operator!=(const scal_t &other) const
    {
        return (value != scal2T(other));
    }
    constexpr bool operator!=(const DTypeBase &other) const
    {
        return (value != other.value);
    }
    constexpr scal_t get() const
    {
        return T2scal(value);
    }
};

template<typename T, typename Y>
inline constexpr Y T2Y(const T &other)
{
    return static_cast<Y>(other);
}

using fp64_t = DTypeBase<double, DTypeEnum::FP64, T2Y<scal_t, double>,
      T2Y<double, scal_t>, 8>;

using fp32_t = DTypeBase<float, DTypeEnum::FP32, T2Y<scal_t, float>,
      T2Y<float, scal_t>, 4>;

using fp32_fast_tf32_t = DTypeBase<float, DTypeEnum::FP32_FAST_TF32,
      T2Y<scal_t, float>, T2Y<float, scal_t>, 4>;

inline constexpr int16_t scal2fp16(const scal_t &other)
{
    // Fake FP32 to FP16 convertor
    return *reinterpret_cast<const int16_t *>(&other);
}

inline constexpr scal_t fp162scal(const int16_t &other)
{
    // Fake FP16 to FP32 convertor
    scal_t res = 0;
    return res;
}

using fp16_t = DTypeBase<int16_t, DTypeEnum::FP16, scal2fp16, fp162scal, 2>;

//! Large enough signed integer for indexing purposes
using Index = int64_t;

// Boolean type for mask
using bool_t = bool;


// Add more types like fp16_t, bf16_t and tf32_t in the future

} // namespace nntile
