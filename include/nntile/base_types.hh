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
 * @version 1.1.0
 * */

#pragma once

#include <cstdint>
#include <assert.h>
#include <iostream>
#include <limits>

// TODO: add conversions aside from CUDA
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#   include <cuda_bf16.h>
#endif // NNTILE_USE_CUDA

// Copy definition of HOST_DEVICE from NVIDIA/cutlass
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
#   define NNTILE_HOST_DEVICE __host__ __device__
#else
#   define NNTILE_HOST_DEVICE
#endif // NNTILE_HOST_DEVICE

namespace nntile
{

//! Integer type for sizes and indices outside NNTile tensors
using Index = std::int64_t;

//! Floating point type for scalar values outside NNTile tensors
using Scalar = float;

//! Default conversion function from StorageT to ReprT
template<typename StorageT, typename ReprT>
inline NNTILE_HOST_DEVICE ReprT convert(const StorageT &value)
{
    return static_cast<ReprT>(value);
}

//! Base type for all NNTile types inside NNTile tensors
/*! It is used to avoid code duplication for all NNTile types. Each type only
 * allows construction and assignment from a compatible standard type.
 * Arithmetic operations are intentionally disabled.
 *
 * @tparam StorageT: memory layout type
 * @tparam ReprT: standard type for representation
 * @tparam to_repr: conversion function from StorageT to ReprT
 * @tparam to_storage: conversion function from ReprT to StorageT
 */
template<
    typename StorageT,
    typename ReprT = StorageT,
    ReprT (*to_repr)(const StorageT &) = convert<StorageT, ReprT>,
    StorageT (*to_storage)(const ReprT &) = convert<ReprT, StorageT>
>
class BaseType
{
public:
    //! Storage type of the integer value
    using storage_t = StorageT;

    //! Representation type of the integer value
    using repr_t = ReprT;

    //! Flag if copy from repr_t does not require conversion
    static constexpr bool trivial_copy_from_compat = trivial_copy_from_repr;

    //! String to represent this type
    static constexpr const char *type_repr = "BaseInt";

    //! Internal value of this type to hold actual data
    storage_t value;

    //! Constructor
    NNTILE_HOST_DEVICE BaseInt() = default;

    //! Constructor from another value of this type
    NNTILE_HOST_DEVICE BaseInt(const BaseInt &other) = default;

    //! Constructor from a repr_t value
    NNTILE_HOST_DEVICE explicit BaseInt(const repr_t &other):
        value(to_storage(other))
    {
    }
    //! Assignment from another value of this type
    NNTILE_HOST_DEVICE BaseInt &operator=(const BaseInt &other) = default;

    //! Assignment from a repr_t value
    NNTILE_HOST_DEVICE BaseInt &operator=(const repr_t &other)
    {
        value = other;
        return *this;
    }

    //! Conversion to repr_t value
    NNTILE_HOST_DEVICE explicit operator repr_t() const
    {
        return to_repr(value);
    }
};

//! NNTile wrapper type for 64-bit signed integers inside NNTile tensors
class int64_t: public BaseInt<std::int64_t, std::int64_t>
{
public:
    //! String to represent this type
    static constexpr const char *type_repr = "nntile::int64_t";
};

//! Print function for nntile::int64_t
inline std::ostream &operator<<(std::ostream &os, const int64_t &value)
{
    os << static_cast<typename int64_t::repr_t>(value);
    return os;
}

//! NNTile wrapper type for bool values inside NNTile tensors
class bool_t
{
public:
    //! Basic type that must have the same size, as this type
    using storage_t = bool;
    //! Basic type that must cover all possible values of this type
    using repr_t = bool;
    //! Flag if copy from repr_t does not require conversion
    static const bool trivial_copy_from_compat = true;
    //! String to represent this type
    static constexpr const char *type_repr = "bool_t";
    //! Internal value of this type to hold actual data
    storage_t value;
    //! Constructor
    NNTILE_HOST_DEVICE bool_t() = default;
    //! Constructor from another value of this type
    NNTILE_HOST_DEVICE bool_t(const bool_t &other) = default;
    //! Constructor from a repr_t value
    NNTILE_HOST_DEVICE explicit bool_t(const repr_t &other):
        value(other)
    {
    }
    //! Assignment from another value of this type
    NNTILE_HOST_DEVICE bool_t &operator=(const bool_t &other) = default;
    //! Assignment from a repr_t value
    NNTILE_HOST_DEVICE bool_t &operator=(const repr_t &other)
    {
        value = other;
        return *this;
    }
    //! Conversion to repr_t value
    NNTILE_HOST_DEVICE explicit operator repr_t() const
    {
        return value;
    }
};

//! Print function for nntile::bool_t
inline std::ostream &operator<<(std::ostream &os, const bool_t &value)
{
    os << static_cast<typename bool_t::repr_t>(value);
    return os;
}

//! NNTile wrapper type for double inside NNTile tensors
class fp64_t: public BaseFloat<double, double>
{
public:
    // Inherit all constructors
    using BaseFloat<double, double>::BaseFloat;

    // Inherit all assignment operators
    using BaseFloat<double, double>::operator=;

    //! String to represent this type
    static constexpr const char *type_repr = "nntile::fp64_t";
};

// class fp64_t
// {
// public:
//     //! Basic type that must have the same size, as this type
//     using storage_t = double;
//     //! Basic type that must cover all possible values of this type
//     using repr_t = double;
//     //! Flag if copy from repr_t does not require conversion
//     static const bool trivial_copy_from_compat = true;
//     //! String to represent this type
//     static constexpr const char *type_repr = "fp64_t";
//     //! Internal value of this type to hold actual data
//     storage_t value;
//     //! Constructor
//     NNTILE_HOST_DEVICE fp64_t() = default;
//     //! Constructor from another value of this type
//     NNTILE_HOST_DEVICE fp64_t(const fp64_t &other) = default;
//     //! Constructor from a repr_t value
//     NNTILE_HOST_DEVICE explicit fp64_t(const repr_t &other):
//         value(other)
//     {
//     }
//     //! Assignment from another value of this type
//     NNTILE_HOST_DEVICE fp64_t &operator=(const fp64_t &other) = default;
//     //! Assignment from a repr_t value
//     NNTILE_HOST_DEVICE fp64_t &operator=(const repr_t &other)
//     {
//         value = other;
//         return *this;
//     }
//     //! Conversion to repr_t value
//     NNTILE_HOST_DEVICE explicit operator repr_t() const
//     {
//         return value;
//     }
//     //! Machine precision of this type
//     static repr_t epsilon()
//     {
//         // Just use std::numeric_limits
//         return std::numeric_limits<double>::epsilon();
//     }
// };

//! Print function for nntile::fp64_t
inline std::ostream &operator<<(std::ostream &os, const fp64_t &value)
{
    os << static_cast<typename fp64_t::repr_t>(value);
    return os;
}

//! NNTile wrapper type for float inside NNTile tensors
class fp32_t
{
public:
    //! Basic type that must have the same size, as this type
    using storage_t = float;
    //! Basic type that must cover all possible values of this type
    using repr_t = float;
    //! Flag if copy from repr_t does not require conversion
    static const bool trivial_copy_from_compat = true;
    //! String to represent this type
    static constexpr const char *type_repr = "fp32_t";
    //! Internal value of this type to hold actual data
    storage_t value;
    //! Constructor
    NNTILE_HOST_DEVICE fp32_t() = default;
    //! Constructor from another value of this type
    NNTILE_HOST_DEVICE fp32_t(const fp32_t &other) = default;
    //! Constructor from a repr_t value
    NNTILE_HOST_DEVICE explicit fp32_t(const repr_t &other):
        value(other)
    {
    }
    //! Assignment from another value of this type
    NNTILE_HOST_DEVICE fp32_t &operator=(const fp32_t &other) = default;
    //! Assignment from a repr_t value
    NNTILE_HOST_DEVICE fp32_t &operator=(const repr_t &other)
    {
        value = other;
        return *this;
    }
    //! Conversion to repr_t value
    NNTILE_HOST_DEVICE explicit operator repr_t() const
    {
        return value;
    }
    //! Machine precision of this type
    static repr_t epsilon()
    {
        // Just use std::numeric_limits
        return std::numeric_limits<float>::epsilon();
    }
};

//! Print function for nntile::fp32_t
inline std::ostream &operator<<(std::ostream &os, const fp32_t &value)
{
    os << static_cast<typename fp32_t::repr_t>(value);
    return os;
}

/*! NNTile wrapper type for TensorFloat32-accelerated float type inside tensors
 *
 * All memory-bound operations are performed in `float` precision, while
 * all compute-bound operations are performed in `TensorFloat32` type.
 */
class fp32_fast_tf32_t
{
public:
    //! Basic type that must have the same size, as this type
    using storage_t = float;
    //! Basic type that must cover all possible values of this type
    using repr_t = float;
    //! Flag if copy from repr_t does not require conversion
    static const bool trivial_copy_from_compat = true;
    //! String to represent this type
    static constexpr const char *type_repr = "fp32_fast_tf32_t";
    //! Internal value of this type to hold actual data
    storage_t value;
    //! Constructor
    NNTILE_HOST_DEVICE fp32_fast_tf32_t() = default;
    //! Constructor from another value of this type
    NNTILE_HOST_DEVICE fp32_fast_tf32_t(const fp32_fast_tf32_t &other) = default;
    //! Constructor from a repr_t value
    NNTILE_HOST_DEVICE explicit fp32_fast_tf32_t(const repr_t &other):
        value(other)
    {
    }
    //! Assignment from another value of this type
    NNTILE_HOST_DEVICE fp32_fast_tf32_t &operator=(const fp32_fast_tf32_t &other) = default;
    //! Assignment from a repr_t value
    NNTILE_HOST_DEVICE fp32_fast_tf32_t &operator=(const repr_t &other)
    {
        value = other;
        return *this;
    }
    //! Conversion to repr_t value
    NNTILE_HOST_DEVICE explicit operator repr_t() const
    {
        return value;
    }
    //! Machine precision of this type
    static repr_t epsilon()
    {
        // Init 1.0 and 1.0+eps identically
        fp32_fast_tf32_t one{1.0}, one_p_eps{1.0};
        auto uintptr = reinterpret_cast<std::uint32_t *>(&one_p_eps);
        // Add a bit into mantissa of 1+eps to get actual value of 1+eps
        *uintptr += 0x2000;
        // Output difference of 1+eps and 1
        return static_cast<repr_t>(one_p_eps) - static_cast<repr_t>(one);
    }
};

//! Print function for nntile::fp32_fast_tf32_t
inline std::ostream &operator<<(std::ostream &os,
        const fp32_fast_tf32_t &value)
{
    os << static_cast<typename fp32_fast_tf32_t::repr_t>(value);
    return os;
}

class fp32_fast_fp16_t
{
public:
    //! Basic type that must have the same size, as this type
    using storage_t = float;
    //! Basic type that must cover all possible values of this type
    using repr_t = float;
    //! Flag if copy from repr_t does not require conversion
    static const bool trivial_copy_from_compat = true;
    //! String to represent this type
    static constexpr const char *type_repr = "fp32_fast_fp16_t";
    //! Internal value of this type to hold actual data
    storage_t value;
    //! Constructor
    NNTILE_HOST_DEVICE fp32_fast_fp16_t() = default;
    //! Constructor from another value of this type
    NNTILE_HOST_DEVICE fp32_fast_fp16_t(const fp32_fast_fp16_t &other) = default;
    //! Constructor from a repr_t value
    NNTILE_HOST_DEVICE explicit fp32_fast_fp16_t(const repr_t &other):
        value(other)
    {
    }
    //! Assignment from another value of this type
    NNTILE_HOST_DEVICE fp32_fast_fp16_t &operator=(const fp32_fast_fp16_t &other) = default;
    //! Assignment from a repr_t value
    NNTILE_HOST_DEVICE fp32_fast_fp16_t &operator=(const repr_t &other)
    {
        value = other;
        return *this;
    }
    //! Conversion to repr_t value
    NNTILE_HOST_DEVICE explicit operator repr_t() const
    {
        return value;
    }
    //! Machine precision of this type
    static repr_t epsilon()
    {
        // Init 1.0 and 1.0+eps identically
        fp32_fast_fp16_t one{1.0}, one_p_eps{1.0};
        auto uintptr = reinterpret_cast<std::uint32_t *>(&one_p_eps);
        // Add a bit into mantissa of 1+eps to get actual value of 1+eps
        *uintptr += 0x10000;
        // Output difference of 1+eps and 1
        return static_cast<repr_t>(one_p_eps) - static_cast<repr_t>(one);
    }
};

//! Print function for nntile::fp32_fast_fp16_t
inline std::ostream &operator<<(std::ostream &os,
        const fp32_fast_fp16_t &value)
{
    os << static_cast<typename fp32_fast_fp16_t::repr_t>(value);
    return os;
}

class fp32_fast_bf16_t
{
public:
    //! Basic type that must have the same size, as this type
    using storage_t = float;
    //! Basic type that must cover all possible values of this type
    using repr_t = float;
    //! Flag if copy from repr_t does not require conversion
    static const bool trivial_copy_from_compat = true;
    //! String to represent this type
    static constexpr const char *type_repr = "fp32_fast_bf16_t";
    //! Internal value of this type to hold actual data
    storage_t value;
    //! Constructor
    NNTILE_HOST_DEVICE fp32_fast_bf16_t() = default;
    //! Constructor from another value of this type
    NNTILE_HOST_DEVICE fp32_fast_bf16_t(const fp32_fast_bf16_t &other) = default;
    //! Constructor from a repr_t value
    NNTILE_HOST_DEVICE explicit fp32_fast_bf16_t(const repr_t &other):
        value(other)
    {
    }
    //! Assignment from another value of this type
    NNTILE_HOST_DEVICE fp32_fast_bf16_t &operator=(const fp32_fast_bf16_t &other) = default;
    //! Assignment from a repr_t value
    NNTILE_HOST_DEVICE fp32_fast_bf16_t &operator=(const repr_t &other)
    {
        value = other;
        return *this;
    }
    //! Conversion to repr_t value
    NNTILE_HOST_DEVICE explicit operator repr_t() const
    {
        return value;
    }
    //! Machine precision of this type
    static repr_t epsilon()
    {
        // Init 1.0 and 1.0+eps identically
        fp32_fast_bf16_t one{1.0}, one_p_eps{1.0};
        auto uintptr = reinterpret_cast<std::uint32_t *>(&one_p_eps);
        // Add a bit into mantissa of 1+eps to get actual value of 1+eps
        *uintptr += 0x10000;
        // Output difference of 1+eps and 1
        return static_cast<repr_t>(one_p_eps) - static_cast<repr_t>(one);
    }
};

//! Print function for nntile::fp32_fast_bf16_t
inline std::ostream &operator<<(std::ostream &os,
        const fp32_fast_bf16_t &value)
{
    os << static_cast<typename fp32_fast_bf16_t::repr_t>(value);
    return os;
}

//! NNTile wrapper type BrainFloat16 type inside tensors
class bf16_t
{
public:
    //! Basic type that must have the same size, as this type
    using storage_t = std::uint16_t;
    //! Basic type that must cover all possible values of this type
    using repr_t = float;
    //! Flag if copy from repr_t does not require conversion
    static const bool trivial_copy_from_compat = false;
    //! String to represent this type
    static constexpr const char *type_repr = "bf16_t";
    //! Internal value of this type to hold actual data
    storage_t value;
    //! Constructor
    NNTILE_HOST_DEVICE bf16_t() = default;
    //! Constructor from another value of this type
    NNTILE_HOST_DEVICE bf16_t(const bf16_t &other) = default;
    //! Constructor from a repr_t value
    NNTILE_HOST_DEVICE explicit bf16_t(const repr_t &other)
    {
#ifdef NNTILE_USE_CUDA
        auto val = __float2bfloat16(other);
        value = *reinterpret_cast<storage_t *>(&val);
#else
        auto raw_uint32 = reinterpret_cast<const std::uint32_t *>(&other);
        value = static_cast<std::uint16_t>(*raw_uint32 >> 16);
#endif
    }
    //! Assignment from another value of this type
    NNTILE_HOST_DEVICE bf16_t &operator=(const bf16_t &other) = default;
    //! Assignment from a repr_t value
    NNTILE_HOST_DEVICE bf16_t &operator=(const repr_t &other)
    {
        return *this = bf16_t(other);
    }
    //! Conversion to repr_t value
    NNTILE_HOST_DEVICE explicit operator repr_t() const
    {
#ifdef NNTILE_USE_CUDA
        auto val = reinterpret_cast<const __nv_bfloat16 *>(&value);
        return __bfloat162float(*val);
#else
        auto raw_uint16 = reinterpret_cast<const std::uint16_t *>(&value);
        auto raw_uint32 = static_cast<std::uint32_t>(*raw_uint16);
        return *reinterpret_cast<repr_t *>(raw_uint32 << 16);
#endif
    }
    //! Machine precision of this type
    static repr_t epsilon()
    {
        // Init 1.0 and 1.0+eps identically
        bf16_t one{1.0}, one_p_eps{1.0};
        auto uintptr = reinterpret_cast<std::uint16_t *>(&one_p_eps);
        // Add a bit into mantissa of 1+eps to get actual value of 1+eps
        *uintptr += 1;
        // Output difference of 1+eps and 1
        return static_cast<repr_t>(one_p_eps) - static_cast<repr_t>(one);
    }
};

//! Print function for nntile::bf16_t
inline std::ostream &operator<<(std::ostream &os, const bf16_t &value)
{
    os << static_cast<typename bf16_t::repr_t>(value);
    return os;
}

} // namespace nntile
