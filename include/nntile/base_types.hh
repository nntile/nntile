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
//enum class DTypeEnum: int
//{
//    NOT_INITIALIZED=101,
//    INT64,
//    INT32,
//    INT16,
//    INT8,
//    BOOL,
//    FP64,
//    FP32,
//    FP32_FAST_TF32,
//    FP32_FAST_FP16,
//    FP32_FAST_BF16,
//    TF32,
//    FP16,
//    BF16,
//    FP8_E4M3,
//    FP8_E5M2
//};
//    constexpr char *name()
//    {
//        switch(value)
//        {
//            case INT64:
//                return "int64";
//                break;
//            case INT32:
//                return "int32";
//                break;
//            case INT16:
//                return "int16";
//                break;
//            case INT8:
//                return "int8";
//                break;
//            case BOOL:
//                return "bool";
//                break;
//            case FP64:
//                return "fp64";
//                break;
//            case FP32:
//                return "fp32";
//                break;
//            case FP32_FAST_TF32:
//                return "fp32_fast_tf32";
//                break;
//            case FP32_FAST_FP16:
//                return "fp32_fast_fp16";
//                break;
//            case FP32_FAST_BF16:
//                return "fp32_fast_bf16";
//                break;
//            case TF32:
//                return "tf32";
//                break;
//            case FP16:
//                return "fp16";
//                break;
//            case BF16:
//                return "bf16";
//                break;
//            case FP8_E4M3:
//                return "fp8_e4m3";
//                break;
//            case FP8_E5M2:
//                return "fp8_e5m2";
//                break;
//            default:
//                return "error";
//        }
//    }
//};

//! Integer type for scalar values and indices outside StarPU buffers
using Index = std::int64_t;

//! This type is meant for scalar values outside StarPU buffers
using scal_t = float;

//! NNTile type for std::int64_t inside StarPU buffers, Tile and Tensor
class int64_t
{
public:
    using internal_t = std::int64_t;
    using compat_t = std::int64_t;
    static constexpr bool trivial_copy_from_compat = true;
    internal_t value;
    explicit constexpr int64_t(const compat_t &other):
        value(other)
    {
    }
    constexpr int64_t &operator=(const compat_t &other)
    {
        value = other;
        return *this;
    }
    constexpr int64_t &operator=(const int64_t &other)
    {
        value = other.value;
        return *this;
    }
    explicit constexpr operator compat_t() const
    {
        return value;
    }
};

//! Print function for nntile::int64_t
inline std::ostream &operator<<(std::ostream &os, const int64_t &value)
{
    os << static_cast<typename int64_t::compat_t>(value);
    return os;
}

//! NNTile type for std::int32_t inside StarPU buffers, Tile and Tensor
class int32_t
{
public:
    using internal_t = std::int32_t;
    using compat_t = std::int32_t;
    static constexpr bool trivial_copy_from_compat = true;
    internal_t value;
    explicit constexpr int32_t(const compat_t &other):
        value(other)
    {
    }
    constexpr int32_t &operator=(const compat_t &other)
    {
        value = other;
        return *this;
    }
    constexpr int32_t &operator=(const int32_t &other)
    {
        value = other.value;
        return *this;
    }
    explicit constexpr operator compat_t() const
    {
        return value;
    }
};

//! Print function for nntile::int32_t
inline std::ostream &operator<<(std::ostream &os, const int32_t &value)
{
    os << static_cast<typename int32_t::compat_t>(value);
    return os;
}

//! NNTile type for std::int16_t inside StarPU buffers, Tile and Tensor
class int16_t
{
public:
    using internal_t = std::int16_t;
    using compat_t = std::int16_t;
    static constexpr bool trivial_copy_from_compat = true;
    internal_t value;
    explicit constexpr int16_t(const compat_t &other):
        value(other)
    {
    }
    constexpr int16_t &operator=(const compat_t &other)
    {
        value = other;
        return *this;
    }
    constexpr int16_t &operator=(const int16_t &other)
    {
        value = other.value;
        return *this;
    }
    explicit constexpr operator compat_t() const
    {
        return value;
    }
};

//! Print function for nntile::int16_t
inline std::ostream &operator<<(std::ostream &os, const int16_t &value)
{
    os << static_cast<typename int16_t::compat_t>(value);
    return os;
}

//! NNTile type for std::int8_t inside StarPU buffers, Tile and Tensor
class int8_t
{
public:
    using internal_t = std::int8_t;
    using compat_t = std::int8_t;
    static constexpr bool trivial_copy_from_compat = true;
    internal_t value;
    explicit constexpr int8_t(const compat_t &other):
        value(other)
    {
    }
    constexpr int8_t &operator=(const compat_t &other)
    {
        value = other;
        return *this;
    }
    constexpr int8_t &operator=(const int8_t &other)
    {
        value = other.value;
        return *this;
    }
    explicit constexpr operator compat_t() const
    {
        return value;
    }
};

//! Print function for nntile::int8_t
inline std::ostream &operator<<(std::ostream &os, const int8_t &value)
{
    os << static_cast<typename int8_t::compat_t>(value);
    return os;
}

//! NNTile type for bool inside StarPU buffers, Tile and Tensor
class bool_t
{
public:
    using internal_t = bool;
    using compat_t = bool;
    static constexpr bool trivial_copy_from_compat = true;
    internal_t value;
    explicit constexpr bool_t(const compat_t &other):
        value(other)
    {
    }
    constexpr bool_t &operator=(const compat_t &other)
    {
        value = other;
        return *this;
    }
    constexpr bool_t &operator=(const bool_t &other)
    {
        value = other.value;
        return *this;
    }
    explicit constexpr operator compat_t() const
    {
        return value;
    }
};

//! Print function for nntile::bool_t
inline std::ostream &operator<<(std::ostream &os, const bool_t &value)
{
    os << static_cast<typename bool_t::compat_t>(value);
    return os;
}

//! NNTile type for double inside StarPU buffers, Tile and Tensor
class fp64_t
{
public:
    using internal_t = double;
    using compat_t = double;
    static constexpr bool trivial_copy_from_compat = true;
    internal_t value;
    explicit constexpr fp64_t(const compat_t &other):
        value(other)
    {
    }
    constexpr fp64_t &operator=(const compat_t &other)
    {
        value = other;
        return *this;
    }
    constexpr fp64_t &operator=(const fp64_t &other)
    {
        value = other.value;
        return *this;
    }
    explicit constexpr operator compat_t() const
    {
        return value;
    }
};

//! Print function for nntile::fp64_t
inline std::ostream &operator<<(std::ostream &os, const fp64_t &value)
{
    os << static_cast<typename fp64_t::compat_t>(value);
    return os;
}

//! NNTile type for float inside StarPU buffers, Tile and Tensor
class fp32_t
{
public:
    using internal_t = float;
    using compat_t = float;
    static constexpr bool trivial_copy_from_compat = true;
    internal_t value;
    explicit constexpr fp32_t(const compat_t &other):
        value(other)
    {
    }
    constexpr fp32_t &operator=(const compat_t &other)
    {
        value = other;
        return *this;
    }
    constexpr fp32_t &operator=(const fp32_t &other)
    {
        value = other.value;
        return *this;
    }
    explicit constexpr operator compat_t() const
    {
        return value;
    }
};

//! Print function for nntile::fp32_t
inline std::ostream &operator<<(std::ostream &os, const fp32_t &value)
{
    os << static_cast<typename fp32_t::compat_t>(value);
    return os;
}

/*! NNTile type for TensorFloat32-accelerated float type
 * 
 * All memory-bound operations are performed in `float` precision, while
 * all compute-bound operations are performed in `TensorFloat32` type.
 */
class fp32_fast_tf32_t
{
public:
    using internal_t = float;
    using compat_t = float;
    static constexpr bool trivial_copy_from_compat = true;
    internal_t value;
    explicit constexpr fp32_fast_tf32_t(const compat_t &other):
        value(other)
    {
    }
    constexpr fp32_fast_tf32_t &operator=(const compat_t &other)
    {
        value = other;
        return *this;
    }
    constexpr fp32_fast_tf32_t &operator=(const fp32_fast_tf32_t &other)
    {
        value = other.value;
        return *this;
    }
    explicit constexpr operator compat_t() const
    {
        return value;
    }
};

//! Print function for nntile::fp32_fast_tf32_t
inline std::ostream &operator<<(std::ostream &os,
        const fp32_fast_tf32_t &value)
{
    os << static_cast<typename fp32_fast_tf32_t::compat_t>(value);
    return os;
}

} // namespace nntile
