/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/cuda.hh
 * CUDA-related compute types
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/defs.h>

#ifdef NNTILE_USE_CUDA_FP16
#   include <cuda_fp16.h>
#endif

//#ifdef NNTILE_USE_CUDA_BF16
#ifdef NNTILE_USE_CUDA
#   include <cuda_bf16.h>
#endif

#ifdef NNTILE_USE_CUDA_FP8
#   include <cuda_fp8.h>
#endif

#include <cstdint>

namespace nntile::kernel
{

//! Templated class to derive compute type for each of basic NNTile types
template<typename T>
struct CUDAComputeType
{
};

//! Compute type for nntile::int64_t type
template<>
struct CUDAComputeType<int64_t>
{
    // nntile::int64_t -> std::int64_t from <cstdint>
    using value = std::int64_t;
};

//! Compute type for nntile::bool_t type
template<>
struct CUDAComputeType<bool_t>
{
    // nntile::bool_t -> bool from C++ base
    using value = bool;
};

//! Compute type for nntile::fp64_t type
template<>
struct CUDAComputeType<fp64_t>
{
    // nntile::fp64_t -> double from C++ base
    using value = double;
};

//! Compute type for nntile::fp32_t type
template<>
struct CUDAComputeType<fp32_t>
{
    // nntile::fp32_t -> double from C++ base
    using value = float;
};

//! Compute type for nntile::fp32_fast_tf32_t type
template<>
struct CUDAComputeType<fp32_fast_tf32_t>
{
    // No member `value` here is for a reason: this type shall be manually
    // converted into computing types, as memory-bound operations shall be done
    // in `fp32_t`, while compute-bound operations shall use `fp32_t` as data
    // storage type and `tf32_t` as compute type.
    using value = float;
};

//! Compute type for nntile::bf16_t type
template<>
struct CUDAComputeType<bf16_t>
{
#ifdef NNTILE_USE_CUDA
    using value = __nv_bfloat16;
#endif
};

//! Convert any NNTile wrapped type value into a corresponding CUDA value
template<typename T>
typename CUDAComputeType<T>::value cast_scalar_cuda(const Scalar &value)
{
    // Return value will be memcopied
    typename CUDAComputeType<T>::value res;
    // Convert input Scalar into intermediate T
    const T inter(value);
    // Copy internal part into the result
    *reinterpret_cast<typename T::storage_t *>(&res) = inter.value;
    return res;
}

//! Convert any ptr to NNTile wrapped type value into a corresponding CUDA ptr
template<typename T>
typename CUDAComputeType<T>::value *cast_pointer_cuda(T *ptr)
{
    return reinterpret_cast<typename CUDAComputeType<T>::value *>(ptr);
}

//! Convert any ptr to NNTile wrapped type value into a corresponding CUDA ptr
template<typename T>
const typename CUDAComputeType<T>::value *cast_pointer_cuda(const T *ptr)
{
    return reinterpret_cast<const typename CUDAComputeType<T>::value *>(ptr);
}

} // namespace nntile::kernel
