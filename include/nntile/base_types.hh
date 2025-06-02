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
#include <sstream>

// Third-party headers
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

//! Base type for all NNTile types inside NNTile tensors
/*! It is used to avoid code duplication for all NNTile types. Each type only
 * allows construction and assignment from a compatible standard type.
 * Arithmetic operations are intentionally disabled.
 *
 * @tparam StorageT: memory layout type
 * @tparam ReprT: standard type for representation, used for IO operations
 */
template<typename StorageT, typename ReprT = StorageT>
class BaseType
{
public:
    //! Storage type of the value
    using storage_t = StorageT;

    //! Representation type of the value
    using repr_t = ReprT;

    //! Internal value of storage_t type to hold actual data
    storage_t value;

    //! Default conversion function from storage_t to repr_t does nothing
    /*! If a conversion is required, this function shall be overloaded with a
     * corresponding conversion function.
     */
    static NNTILE_HOST_DEVICE repr_t to_repr(const storage_t &value)
    {
        return value;
    }

    //! Default conversion function from repr_t to storage_t does nothing
    /*! If a conversion is required, this function shall be overloaded with a
     * corresponding conversion function.
     */
    static NNTILE_HOST_DEVICE storage_t to_storage(const repr_t &value)
    {
        return value;
    }

    //! Default constructor with no arguments
    NNTILE_HOST_DEVICE BaseType() = default;

    //! Default copy constructor
    NNTILE_HOST_DEVICE explicit BaseType(const BaseType &other) = default;

    //! Constructor from a compatible standard type value
    NNTILE_HOST_DEVICE explicit BaseType(const repr_t &other):
        value(to_storage(other))
    {
    }

    //! Default assignment from another value of this type
    NNTILE_HOST_DEVICE BaseType &operator=(const BaseType &other) = default;

    //! Assignment from a compatible standard type value
    NNTILE_HOST_DEVICE BaseType &operator=(const repr_t &other)
    {
        value = other;
        return *this;
    }

    //! Conversion to compatible standard type value
    NNTILE_HOST_DEVICE explicit operator repr_t() const
    {
        return to_repr(value);
    }
};

//! Print function for nntile::BaseType
template<typename StorageT, typename ReprT>
inline std::ostream &operator<<(
    std::ostream &os,
    const BaseType<StorageT, ReprT> &value
)
{
    os << static_cast<ReprT>(value);
    return os;
}

//! Type postfix template function
template<typename T, typename... Ts>
inline std::string type_postfix()
{
    std::stringstream result;
    result << type_postfix<T>() << "_" << type_postfix<Ts...>();
    return result.str();
}

//! NNTile wrapper type for 64-bit signed integers inside NNTile tensors
class int64_t: public BaseType<std::int64_t>
{
public:
    // Inherit all constructors
    using BaseType<std::int64_t>::BaseType;

    // Inherit all assignment operators
    using BaseType<std::int64_t>::operator=;
};

//! Type postfix template specialization for nntile::int64_t
template<>
inline std::string type_postfix<nntile::int64_t>()
{
    return "int64";
}

//! NNTile wrapper type for bool values inside NNTile tensors
class bool_t: public BaseType<bool>
{
public:
    // Inherit all constructors
    using BaseType<bool>::BaseType;

    // Inherit all assignment operators
    using BaseType<bool>::operator=;
};

//! Type postfix template specialization for nntile::bool_t
template<>
inline std::string type_postfix<nntile::bool_t>()
{
    return "bool";
}

//! NNTile wrapper type for double inside NNTile tensors
class fp64_t: public BaseType<double>
{
public:
    // Inherit all constructors
    using BaseType<double>::BaseType;

    // Inherit all assignment operators
    using BaseType<double>::operator=;
};

//! Type postfix template specialization for nntile::fp64_t
template<>
inline std::string type_postfix<nntile::fp64_t>()
{
    return "fp64";
}

//! NNTile wrapper type for float inside NNTile tensors
class fp32_t: public BaseType<float>
{
public:
    // Inherit all constructors
    using BaseType<float>::BaseType;

    // Inherit all assignment operators
    using BaseType<float>::operator=;
};

//! Type postfix template specialization for nntile::fp32_t
template<>
inline std::string type_postfix<nntile::fp32_t>()
{
    return "fp32";
}

/*! NNTile wrapper type for TensorFloat32-accelerated float type inside tensors
 *
 * All memory-bound operations are performed in `float` precision, while
 * all compute-bound operations are performed in `TensorFloat32` type.
 */
class fp32_fast_tf32_t: public BaseType<float>
{
public:
    // Inherit all constructors
    using BaseType<float>::BaseType;

    // Inherit all assignment operators
    using BaseType<float>::operator=;
};

//! Type postfix template specialization for nntile::fp32_fast_tf32_t
template<>
inline std::string type_postfix<nntile::fp32_fast_tf32_t>()
{
    return "fp32_fast_tf32";
}

/*! NNTile wrapper type for FP16-accelerated float type inside tensors
 *
 * All memory-bound operations are performed in `float` precision, while
 * all compute-bound operations are performed in `fp16` type.
 */
class fp32_fast_fp16_t: public BaseType<float>
{
public:
    // Inherit all constructors
    using BaseType<float>::BaseType;

    // Inherit all assignment operators
    using BaseType<float>::operator=;
};

//! Type postfix template specialization for nntile::fp32_fast_fp16_t
template<>
inline std::string type_postfix<nntile::fp32_fast_fp16_t>()
{
    return "fp32_fast_fp16";
}

/*! NNTile wrapper type for BF16-accelerated float type inside tensors
 *
 * All memory-bound operations are performed in `float` precision, while
 * all compute-bound operations are performed in `bf16` type.
 */
class fp32_fast_bf16_t: public BaseType<float>
{
public:
    // Inherit all constructors
    using BaseType<float>::BaseType;

    // Inherit all assignment operators
    using BaseType<float>::operator=;
};

//! Type postfix template specialization for nntile::fp32_fast_bf16_t
template<>
inline std::string type_postfix<nntile::fp32_fast_bf16_t>()
{
    return "fp32_fast_bf16";
}

//! NNTile wrapper type BrainFloat16 type inside tensors
class bf16_t: public BaseType<std::uint16_t, float>
{
public:
    // Inherit all constructors
    using BaseType<std::uint16_t, float>::BaseType;

    // Inherit all assignment operators
    using BaseType<std::uint16_t, float>::operator=;

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

//! Type postfix template specialization for nntile::bf16_t
template<>
inline std::string type_postfix<nntile::bf16_t>()
{
    return "bf16";
}

} // namespace nntile
