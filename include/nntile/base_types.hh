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
#include <limits>

namespace nntile
{

//! Integer type for sizes and indices outside StarPU buffers
using Index = std::int64_t;

//! Floating point type for scalar values outside StarPU buffers
using Scalar = float;

//! NNTile wrapper type for 64-bit signed integers inside NNTile tensors
class int64_t
{
public:
    //! Basic type that must have the same size, as this type
    using internal_t = std::int64_t;
    //! Basic type that must cover all possible values of this type
    using compat_t = std::int64_t;
    //! Flag if copy from compat_t does not require conversion
    static const bool trivial_copy_from_compat = true;
    //! String to represent this type
    static constexpr const char *type_repr = "int64_t";
    //! Internal value of this type to hold actual data
    internal_t value;
    //! Constructor
    int64_t() = default;
    //! Constructor from another value of this type
    explicit int64_t(const int64_t &other) = default;
    //! Constructor from a compat_t value
    explicit int64_t(const compat_t &other):
        value(other)
    {
    }
    //! Assignment from another value of this type
    int64_t &operator=(const int64_t &other) = default;
    //! Assignment from a compat_t value
    int64_t &operator=(const compat_t &other)
    {
        value = other;
        return *this;
    }
    //! Conversion to compat_t value
    explicit operator compat_t() const
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

//! NNTile wrapper type for bool values inside NNTile tensors
class bool_t
{
public:
    //! Basic type that must have the same size, as this type
    using internal_t = bool;
    //! Basic type that must cover all possible values of this type
    using compat_t = bool;
    //! Flag if copy from compat_t does not require conversion
    static const bool trivial_copy_from_compat = true;
    //! String to represent this type
    static constexpr const char *type_repr = "bool_t";
    //! Internal value of this type to hold actual data
    internal_t value;
    //! Constructor
    bool_t() = default;
    //! Constructor from another value of this type
    explicit bool_t(const bool_t &other) = default;
    //! Constructor from a compat_t value
    explicit bool_t(const compat_t &other):
        value(other)
    {
    }
    //! Assignment from another value of this type
    bool_t &operator=(const bool_t &other) = default;
    //! Assignment from a compat_t value
    bool_t &operator=(const compat_t &other)
    {
        value = other;
        return *this;
    }
    //! Conversion to compat_t value
    explicit operator compat_t() const
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

//! NNTile wrapper type for double inside NNTile tensors
class fp64_t
{
public:
    //! Basic type that must have the same size, as this type
    using internal_t = double;
    //! Basic type that must cover all possible values of this type
    using compat_t = double;
    //! Flag if copy from compat_t does not require conversion
    static const bool trivial_copy_from_compat = true;
    //! String to represent this type
    static constexpr const char *type_repr = "fp64_t";
    //! Internal value of this type to hold actual data
    internal_t value;
    //! Constructor
    fp64_t() = default;
    //! Constructor from another value of this type
    explicit fp64_t(const fp64_t &other) = default;
    //! Constructor from a compat_t value
    explicit fp64_t(const compat_t &other):
        value(other)
    {
    }
    //! Assignment from another value of this type
    fp64_t &operator=(const fp64_t &other) = default;
    //! Assignment from a compat_t value
    fp64_t &operator=(const compat_t &other)
    {
        value = other;
        return *this;
    }
    //! Conversion to compat_t value
    explicit operator compat_t() const
    {
        return value;
    }
    //! Machine precision of this type
    static compat_t epsilon()
    {
        // Just use std::numeric_limits
        return std::numeric_limits<double>::epsilon();
    }
};

//! Print function for nntile::fp64_t
inline std::ostream &operator<<(std::ostream &os, const fp64_t &value)
{
    os << static_cast<typename fp64_t::compat_t>(value);
    return os;
}

//! NNTile wrapper type for float inside NNTile tensors
class fp32_t
{
public:
    //! Basic type that must have the same size, as this type
    using internal_t = float;
    //! Basic type that must cover all possible values of this type
    using compat_t = float;
    //! Flag if copy from compat_t does not require conversion
    static const bool trivial_copy_from_compat = true;
    //! String to represent this type
    static constexpr const char *type_repr = "fp32_t";
    //! Internal value of this type to hold actual data
    internal_t value;
    //! Constructor
    fp32_t() = default;
    //! Constructor from another value of this type
    explicit fp32_t(const fp32_t &other) = default;
    //! Constructor from a compat_t value
    explicit fp32_t(const compat_t &other):
        value(other)
    {
    }
    //! Assignment from another value of this type
    fp32_t &operator=(const fp32_t &other) = default;
    //! Assignment from a compat_t value
    fp32_t &operator=(const compat_t &other)
    {
        value = other;
        return *this;
    }
    //! Conversion to compat_t value
    explicit operator compat_t() const
    {
        return value;
    }
    //! Machine precision of this type
    static compat_t epsilon()
    {
        // Just use std::numeric_limits
        return std::numeric_limits<float>::epsilon();
    }
};

//! Print function for nntile::fp32_t
inline std::ostream &operator<<(std::ostream &os, const fp32_t &value)
{
    os << static_cast<typename fp32_t::compat_t>(value);
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
    using internal_t = float;
    //! Basic type that must cover all possible values of this type
    using compat_t = float;
    //! Flag if copy from compat_t does not require conversion
    static const bool trivial_copy_from_compat = true;
    //! String to represent this type
    static constexpr const char *type_repr = "fp32_fast_tf32_t";
    //! Internal value of this type to hold actual data
    internal_t value;
    //! Constructor
    fp32_fast_tf32_t() = default;
    //! Constructor from another value of this type
    explicit fp32_fast_tf32_t(const fp32_fast_tf32_t &other) = default;
    //! Constructor from a compat_t value
    explicit fp32_fast_tf32_t(const compat_t &other):
        value(other)
    {
    }
    //! Assignment from another value of this type
    fp32_fast_tf32_t &operator=(const fp32_fast_tf32_t &other) = default;
    //! Assignment from a compat_t value
    fp32_fast_tf32_t &operator=(const compat_t &other)
    {
        value = other;
        return *this;
    }
    //! Conversion to compat_t value
    explicit operator compat_t() const
    {
        return value;
    }
    //! Machine precision of this type
    static compat_t epsilon()
    {
        // Init 1.0 and 1.0+eps identically
        fp32_fast_tf32_t one{1.0}, one_p_eps{1.0};
        auto uintptr = reinterpret_cast<std::uint32_t *>(&one_p_eps);
        // Add a bit into mantissa of 1+eps to get actual value of 1+eps
        *uintptr += 0x2000;
        // Output difference of 1+eps and 1
        return static_cast<compat_t>(one_p_eps) - static_cast<compat_t>(one);
    }
};

//! Print function for nntile::fp32_fast_tf32_t
inline std::ostream &operator<<(std::ostream &os,
        const fp32_fast_tf32_t &value)
{
    os << static_cast<typename fp32_fast_tf32_t::compat_t>(value);
    return os;
}

//! NNTile wrapper type BrainFloat16 type inside tensors
class bf16_t
{
public:
    //! Basic type that must have the same size, as this type
    using internal_t = std::uint16_t;
    //! Basic type that must cover all possible values of this type
    using compat_t = float;
    //! Flag if copy from compat_t does not require conversion
    static const bool trivial_copy_from_compat = false;
    //! String to represent this type
    static constexpr const char *type_repr = "bf16_t";
    //! Internal value of this type to hold actual data
    internal_t value;
    //! Constructor
    bf16_t() = default;
    //! Constructor from another value of this type
    explicit bf16_t(const bf16_t &other) = default;
    //! Constructor from a compat_t value
    explicit bf16_t(const compat_t &other);
    //! Assignment from another value of this type
    bf16_t &operator=(const bf16_t &other) = default;
    //! Assignment from a compat_t value
    bf16_t &operator=(const compat_t &other);
    //! Conversion to compat_t value
    explicit operator compat_t() const;
    //! Machine precision of this type
    static compat_t epsilon()
    {
        // Init 1.0 and 1.0+eps identically
        bf16_t one{1.0}, one_p_eps{1.0};
        auto uintptr = reinterpret_cast<std::uint16_t *>(&one_p_eps);
        // Add a bit into mantissa of 1+eps to get actual value of 1+eps
        *uintptr += 1;
        // Output difference of 1+eps and 1
        return static_cast<compat_t>(one_p_eps) - static_cast<compat_t>(one);
    }
};

//! Print function for nntile::bf16_t
inline std::ostream &operator<<(std::ostream &os, const bf16_t &value)
{
    os << static_cast<typename bf16_t::compat_t>(value);
    return os;
}

} // namespace nntile
