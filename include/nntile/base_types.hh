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

// Compile-time definitions
#include <nntile/defs.h>

// Standard library headers
#include <cstdint>
#include <assert.h>
#include <iostream>
#include <limits>
#include <string>
#include <cstring>

// Third-party headers
#ifdef NNTILE_USE_CUDA
#   include <cuda_bf16.h>
#   include <cuda_fp16.h>
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

//! Type postfix template function
template<typename T, typename... Ts>
inline std::string type_postfix()
{
    std::string result(T::short_name);
    if constexpr (sizeof...(Ts) > 0)
    {
        result += "_" + type_postfix<Ts...>();
    }
    return result;
}

//! NNTile wrapper type for 64-bit signed integers inside NNTile tensors
class int64_t
{
public:
    //! Storage type of the value
    using storage_t = std::int64_t;

    //! Representation type of the value
    using repr_t = std::int64_t;

    //! Internal value of storage_t type to hold actual data
    storage_t value;

    //! Default constructor with no arguments
    NNTILE_HOST_DEVICE int64_t() = default;

    //! Default copy constructor
    NNTILE_HOST_DEVICE int64_t(const int64_t &other) = default;

    //! Constructor from a compatible standard type value
    NNTILE_HOST_DEVICE int64_t(const repr_t &other):
        value(other)
    {
    }

    //! Default assignment from another value of this type
    NNTILE_HOST_DEVICE int64_t &operator=(const int64_t &other) = default;

    //! Assignment from a compatible standard type value
    NNTILE_HOST_DEVICE int64_t &operator=(const repr_t &other)
    {
        value = other;
        return *this;
    }

    //! Conversion to compatible standard type value
    NNTILE_HOST_DEVICE explicit operator repr_t() const
    {
        return value;
    }

    // Short type name
    static constexpr const char *short_name = "i64";

    // Long type name
    static constexpr const char *long_name = "int64_t";

    // Function to check if a type is a floating point type
    static constexpr bool is_floating_point_type = false;
};

//! Print function for nntile::int64_t
inline std::ostream &operator<<(std::ostream &os, const int64_t &value)
{
    os << value.value;
    return os;
}

//! NNTile wrapper type for bool values inside NNTile tensors
class bool_t
{
public:
    //! Storage type of the value
    using storage_t = bool;

    //! Representation type of the value
    using repr_t = bool;

    //! Internal value of storage_t type to hold actual data
    storage_t value;

    //! Default constructor with no arguments
    NNTILE_HOST_DEVICE bool_t() = default;

    //! Default copy constructor
    NNTILE_HOST_DEVICE bool_t(const bool_t &other) = default;

    //! Constructor from a compatible standard type value
    NNTILE_HOST_DEVICE bool_t(const repr_t &other):
        value(other)
    {
    }

    //! Default assignment from another value of this type
    NNTILE_HOST_DEVICE bool_t &operator=(const bool_t &other) = default;

    //! Assignment from a compatible standard type value
    NNTILE_HOST_DEVICE bool_t &operator=(const repr_t &other)
    {
        value = other;
        return *this;
    }

    //! Conversion to compatible standard type value
    NNTILE_HOST_DEVICE explicit operator repr_t() const
    {
        return value;
    }

    // Short type name
    // "b8" stands for "boolean 8-bit", as it occupies entire byte
    static constexpr const char *short_name = "b8";

    // Long type name
    static constexpr const char *long_name = "bool_t";

    // Function to check if a type is a floating point type
    static constexpr bool is_floating_point_type = false;
};

//! Print function for nntile::bool_t
inline std::ostream &operator<<(std::ostream &os, const bool_t &value)
{
    os << value.value;
    return os;
}

//! NNTile wrapper type for double inside NNTile tensors
class fp64_t
{
public:
    //! Storage type of the value
    using storage_t = double;

    //! Representation type of the value
    using repr_t = double;

    //! Internal value of storage_t type to hold actual data
    storage_t value;

    //! Default constructor with no arguments
    NNTILE_HOST_DEVICE fp64_t() = default;

    //! Default copy constructor
    NNTILE_HOST_DEVICE fp64_t(const fp64_t &other) = default;

    //! Constructor from a compatible standard type value
    NNTILE_HOST_DEVICE fp64_t(const repr_t &other):
        value(other)
    {
    }

    //! Default assignment from another value of this type
    NNTILE_HOST_DEVICE fp64_t &operator=(const fp64_t &other) = default;

    //! Assignment from a compatible standard type value
    NNTILE_HOST_DEVICE fp64_t &operator=(const repr_t &other)
    {
        value = other;
        return *this;
    }

    //! Conversion to compatible standard type value
    NNTILE_HOST_DEVICE explicit operator repr_t() const
    {
        return value;
    }

    // Short type name
    static constexpr const char *short_name = "fp64";

    // Long type name
    static constexpr const char *long_name = "fp64_t";

    // Function to check if a type is a floating point type
    static constexpr bool is_floating_point_type = true;

    // Machine precision
    static constexpr repr_t epsilon = std::numeric_limits<double>::epsilon();
};

//! Print function for nntile::fp64_t
inline std::ostream &operator<<(std::ostream &os, const fp64_t &value)
{
    os << value.value;
    return os;
}

//! NNTile wrapper type for float inside NNTile tensors
class fp32_t
{
public:
    //! Storage type of the value
    using storage_t = float;

    //! Representation type of the value
    using repr_t = float;

    //! Internal value of storage_t type to hold actual data
    storage_t value;

    //! Default constructor with no arguments
    NNTILE_HOST_DEVICE fp32_t() = default;

    //! Default copy constructor
    NNTILE_HOST_DEVICE fp32_t(const fp32_t &other) = default;

    //! Constructor from a compatible standard type value
    NNTILE_HOST_DEVICE fp32_t(const repr_t &other):
        value(other)
    {
    }

    //! Default assignment from another value of this type
    NNTILE_HOST_DEVICE fp32_t &operator=(const fp32_t &other) = default;

    //! Assignment from a compatible standard type value
    NNTILE_HOST_DEVICE fp32_t &operator=(const repr_t &other)
    {
        value = other;
        return *this;
    }

    //! Conversion to compatible standard type value
    NNTILE_HOST_DEVICE explicit operator repr_t() const
    {
        return value;
    }

    // Short type name
    static constexpr const char *short_name = "fp32";

    // Long type name
    static constexpr const char *long_name = "fp32_t";

    // Function to check if a type is a floating point type
    static constexpr bool is_floating_point_type = true;

    // Machine precision
    static constexpr repr_t epsilon = std::numeric_limits<float>::epsilon();
};

//! Print function for nntile::fp32_t
inline std::ostream &operator<<(std::ostream &os, const fp32_t &value)
{
    os << value.value;
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
    //! Storage type of the value
    using storage_t = float;

    //! Representation type of the value
    using repr_t = float;

    //! Internal value of storage_t type to hold actual data
    storage_t value;

    //! Default constructor with no arguments
    NNTILE_HOST_DEVICE fp32_fast_tf32_t() = default;

    //! Default copy constructor
    NNTILE_HOST_DEVICE fp32_fast_tf32_t(
        const fp32_fast_tf32_t &other) = default;

    //! Constructor from a compatible standard type value
    NNTILE_HOST_DEVICE fp32_fast_tf32_t(const repr_t &other):
        value(other)
    {
    }

    //! Default assignment from another value of this type
    NNTILE_HOST_DEVICE fp32_fast_tf32_t &operator=(
        const fp32_fast_tf32_t &other) = default;

    //! Assignment from a compatible standard type value
    NNTILE_HOST_DEVICE fp32_fast_tf32_t &operator=(const repr_t &other)
    {
        value = other;
        return *this;
    }

    //! Conversion to compatible standard type value
    NNTILE_HOST_DEVICE explicit operator repr_t() const
    {
        return value;
    }

    // Short type name
    static constexpr const char *short_name = "tf32";

    // Long type name
    static constexpr const char *long_name = "fp32_fast_tf32_t";

    // Function to check if a type is a floating point type
    static constexpr bool is_floating_point_type = true;

    // Machine precision is 1/1024, as only 10 bits are used for the mantissa
    static constexpr repr_t epsilon = double{1.0} / double{1024.0};
};

//! Print function for nntile::fp32_fast_tf32_t
inline std::ostream &operator<<(
    std::ostream &os, const fp32_fast_tf32_t &value)
{
    os << value.value;
    return os;
}

/*! NNTile wrapper type for FP16-accelerated float type inside tensors
 *
 * All memory-bound operations are performed in `float` precision, while
 * all compute-bound operations are performed in `fp16` type.
 */
class fp32_fast_fp16_t
{
public:
    //! Storage type of the value
    using storage_t = float;

    //! Representation type of the value
    using repr_t = float;

    //! Internal value of storage_t type to hold actual data
    storage_t value;

    //! Default constructor with no arguments
    NNTILE_HOST_DEVICE fp32_fast_fp16_t() = default;

    //! Default copy constructor
    NNTILE_HOST_DEVICE fp32_fast_fp16_t(
        const fp32_fast_fp16_t &other) = default;

    //! Constructor from a compatible standard type value
    NNTILE_HOST_DEVICE fp32_fast_fp16_t(const repr_t &other):
        value(other)
    {
    }

    //! Default assignment from another value of this type
    NNTILE_HOST_DEVICE fp32_fast_fp16_t &operator=(
        const fp32_fast_fp16_t &other) = default;

    //! Assignment from a compatible standard type value
    NNTILE_HOST_DEVICE fp32_fast_fp16_t &operator=(const repr_t &other)
    {
        value = other;
        return *this;
    }

    //! Conversion to compatible standard type value
    NNTILE_HOST_DEVICE explicit operator repr_t() const
    {
        return value;
    }

    // Short type name
    static constexpr const char *short_name = "fp32=fp16";

    // Long type name
    static constexpr const char *long_name = "fp32_fast_fp16_t";

    // Function to check if a type is a floating point type
    static constexpr bool is_floating_point_type = true;

    // Machine precision is 1/1024, as only 10 bits are used for the mantissa
    static constexpr repr_t epsilon = double{1.0} / double{1024.0};
};

//! Print function for nntile::fp32_fast_fp16_t
inline std::ostream &operator<<(
    std::ostream &os, const fp32_fast_fp16_t &value)
{
    os << value.value;
    return os;
}

/*! NNTile wrapper type for BF16-accelerated float type inside tensors
 *
 * All memory-bound operations are performed in `float` precision, while
 * all compute-bound operations are performed in `bf16` type.
 */
class fp32_fast_bf16_t
{
public:
    //! Storage type of the value
    using storage_t = float;

    //! Representation type of the value
    using repr_t = float;

    //! Internal value of storage_t type to hold actual data
    storage_t value;

    //! Default constructor with no arguments
    NNTILE_HOST_DEVICE fp32_fast_bf16_t() = default;

    //! Default copy constructor
    NNTILE_HOST_DEVICE fp32_fast_bf16_t(
        const fp32_fast_bf16_t &other) = default;

    //! Constructor from a compatible standard type value
    NNTILE_HOST_DEVICE fp32_fast_bf16_t(const repr_t &other):
        value(other)
    {
    }

    //! Default assignment from another value of this type
    NNTILE_HOST_DEVICE fp32_fast_bf16_t &operator=(
        const fp32_fast_bf16_t &other) = default;

    //! Assignment from a compatible standard type value
    NNTILE_HOST_DEVICE fp32_fast_bf16_t &operator=(const repr_t &other)
    {
        value = other;
        return *this;
    }

    //! Conversion to compatible standard type value
    NNTILE_HOST_DEVICE explicit operator repr_t() const
    {
        return value;
    }

    // Short type name
    static constexpr const char *short_name = "fp32=bf16";

    // Long type name
    static constexpr const char *long_name = "fp32_fast_bf16_t";

    // Function to check if a type is a floating point type
    static constexpr bool is_floating_point_type = true;

    // Machine precision is 1/128, as only 7 bits are used for the mantissa
    static constexpr repr_t epsilon = double{1.0} / double{128.0};
};

//! Print function for nntile::fp32_fast_bf16_t
inline std::ostream &operator<<(
    std::ostream &os, const fp32_fast_bf16_t &value)
{
    os << value.value;
    return os;
}

//! NNTile wrapper type BrainFloat16 type inside tensors
class bf16_t
{
public:
    //! Storage type of the value
    using storage_t = std::uint16_t;

    //! Representation type of the value
    using repr_t = float;

    //! Internal value of storage_t type to hold actual data
    storage_t value;

    //! Default constructor with no arguments
    NNTILE_HOST_DEVICE bf16_t() = default;

    //! Default copy constructor
    NNTILE_HOST_DEVICE bf16_t(const bf16_t &other) = default;

    //! Constructor from a compatible standard type value
    NNTILE_HOST_DEVICE bf16_t(const repr_t &other):
        value(to_storage(other))
    {
    }

    //! Default assignment from another value of this type
    NNTILE_HOST_DEVICE bf16_t &operator=(const bf16_t &other) = default;

    //! Assignment from a compatible standard type value
    NNTILE_HOST_DEVICE bf16_t &operator=(const repr_t &other)
    {
        value = to_storage(other);
        return *this;
    }

    //! Conversion to compatible standard type value
    NNTILE_HOST_DEVICE explicit operator repr_t() const
    {
        return to_repr(value);
    }

    // Short type name
    static constexpr const char *short_name = "bf16";

    // Long type name
    static constexpr const char *long_name = "bf16_t";

    // Function to check if a type is a floating point type
    static constexpr bool is_floating_point_type = true;

    // Machine precision is 1/128, as only 7 bits are used for the mantissa
    static constexpr repr_t epsilon = double{1.0} / double{128.0};

    //! Conversion from repr_t to storage_t
    static NNTILE_HOST_DEVICE storage_t to_storage(const repr_t &value)
    {
#ifdef NNTILE_USE_CUDA
        auto val = __float2bfloat16(value);
        return *reinterpret_cast<storage_t *>(&val);
#else
        auto raw_uint32 = *reinterpret_cast<const std::uint32_t *>(&value);
        return static_cast<storage_t>(raw_uint32 >> 16);
#endif
    }

    //! Conversion from storage_t to repr_t
    static NNTILE_HOST_DEVICE repr_t to_repr(const storage_t &value)
    {
#ifdef NNTILE_USE_CUDA
        auto val = *reinterpret_cast<const __nv_bfloat16 *>(&value);
        return __bfloat162float(val);
#else
        std::uint32_t raw_uint32 = value;
        return *reinterpret_cast<repr_t *>(&raw_uint32);
#endif
    }
};

//! Print function for nntile::bf16_t
inline std::ostream &operator<<(std::ostream &os, const bf16_t &value)
{
    os << static_cast<float>(value);
    return os;
}

//! NNTile wrapper type Float16 type inside tensors
class fp16_t
{
public:
    //! Storage type of the value
    using storage_t = std::uint16_t;

    //! Representation type of the value
    using repr_t = float;

    //! Internal value of storage_t type to hold actual data
    storage_t value;

    //! Default constructor with no arguments
    NNTILE_HOST_DEVICE fp16_t() = default;

    //! Default copy constructor
    NNTILE_HOST_DEVICE fp16_t(const fp16_t &other) = default;

    //! Constructor from a compatible standard type value
    NNTILE_HOST_DEVICE fp16_t(const repr_t &other):
        value(to_storage(other))
    {
    }

    //! Default assignment from another value of this type
    NNTILE_HOST_DEVICE fp16_t &operator=(const fp16_t &other) = default;

    //! Assignment from a compatible standard type value
    NNTILE_HOST_DEVICE fp16_t &operator=(const repr_t &other)
    {
        value = to_storage(other);
        return *this;
    }

    //! Conversion to compatible standard type value
    NNTILE_HOST_DEVICE explicit operator repr_t() const
    {
        return to_repr(value);
    }

    // Short type name
    static constexpr const char *short_name = "fp16";

    // Long type name
    static constexpr const char *long_name = "fp16_t";

    // Function to check if a type is a floating point type
    static constexpr bool is_floating_point_type = true;

    // Machine precision is 1/128, as only 7 bits are used for the mantissa
    static constexpr repr_t epsilon = double{1.0} / double{128.0};

    //! Conversion from repr_t to storage_t
    static NNTILE_HOST_DEVICE storage_t to_storage(const repr_t &value)
    {
#ifdef NNTILE_USE_CUDA
        auto val = __float2half(value);
        return *reinterpret_cast<storage_t *>(&val);
#else
        uint32_t x;
        memcpy(&x, &value, sizeof(value));

        uint32_t sign = (x >> 31) & 0x1;
        uint32_t exp = (x >> 23) & 0xFF;
        uint32_t mantissa = x & 0x7FFFFF;

        // Handle special cases
        if (exp == 0xFF) // NaN or Inf
        {
            return (sign << 15) | 0x7C00 | (mantissa ? 0x200 : 0);
        }

        // Convert exponent (bias adjustment: 127 to 15)
        int32_t exp32 = static_cast<int32_t>(exp) - 127;
        uint32_t exp16;

        if (exp32 > 15) // Overflow -> convert to Inf
        {
            exp16 = 0x1F;
            mantissa = 0;
        }
        else if (exp32 < -14) // Underflow -> denormal or zero
        {
            // For very small numbers, we'll flush to zero
            exp16 = 0;
            mantissa = 0;
        }
        else
        {
            exp16 = static_cast<uint32_t>(exp32 + 15);
        }

        // Round mantissa (10 bits for float16)
        uint32_t mantissa16 = (mantissa + 0x400) >> 13;
        if (mantissa16 & 0x400) // Check for rounding overflow
        {
            mantissa16 = 0;
            exp16++;
            if (exp16 > 30) // Handle exponent overflow after rounding
            {
                exp16 = 0x1F;
                mantissa16 = 0;
            }
        }
        return (sign << 15) | (exp16 << 10) | (mantissa16 & 0x3FF);
#endif
    }

    //! Conversion from storage_t to repr_t
    static NNTILE_HOST_DEVICE repr_t to_repr(const storage_t &value)
    {
#ifdef NNTILE_USE_CUDA
        auto val = *reinterpret_cast<const __half *>(&value);
        return __half2float(val);
#else
        uint32_t sign = (value >> 15) & 0x1;
        uint32_t exp = (value >> 10) & 0x1F;
        uint32_t mantissa = value & 0x3FF;

        if (exp == 0x1F) // NaN or Inf
        {
            uint32_t result = (sign << 31) | 0x7F800000 | (mantissa << 13);
            float f;
            memcpy(&f, &result, sizeof(f));
            return f;
        }

        if (exp == 0) // Zero or denormal
        {
            if (mantissa == 0)
            {
                return 0.0f;
            }
            // Handle denormal numbers (simplified)
            exp = 1; // Make it normal
        }

        uint32_t exp32 = exp + 112; // Adjust bias (15 to 127)
        uint32_t mantissa32 = mantissa << 13;
        uint32_t result = (sign << 31) | (exp32 << 23) | mantissa32;

        float f;
        memcpy(&f, &result, sizeof(f));
        return f;
#endif
    }
};

//! Print function for nntile::fp16_t
inline std::ostream &operator<<(std::ostream &os, const fp16_t &value)
{
    os << static_cast<float>(value);
    return os;
}

} // namespace nntile
