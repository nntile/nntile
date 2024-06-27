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
 * @version 1.0.0
 * */

#pragma once

#include <nntile/defs.h>

#ifdef NNTILE_USE_CUDA_FP16
#   include <cuda_fp16.h>
#endif

#ifdef NNTILE_USE_CUDA_BF16
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

//! Compute type for nntile::int32_t type
template<>
struct CUDAComputeType<int32_t>
{
    // nntile::int32_t -> std::int32_t from <cstdint>
    using value = std::int32_t;
};

//! Compute type for nntile::int16_t type
template<>
struct CUDAComputeType<int16_t>
{
    // nntile::int16_t -> std::int16_t from <cstdint>
    using value = std::int16_t;
};

//! Compute type for nntile::int8_t type
template<>
struct CUDAComputeType<int8_t>
{
    // nntile::int8_t -> std::int8_t from <cstdint>
    using value = std::int8_t;
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
#ifdef NNTILE_USE_CUDA_TF32
    using value = float;
#endif
};

//! Compute type for nntile::fp32_fast_fp16_t type
template<>
struct CUDAComputeType<fp32_fast_fp16_t>
{
#ifdef NNTILE_USE_CUDA_TF32
    using value = float;
#endif
};

//! Compute type for nntile::fp32_fast_bf16_t type
template<>
struct CUDAComputeType<fp32_fast_bf16_t>
{
#ifdef NNTILE_USE_CUDA_TF32
    using value = float;
#endif
};

//! Compute type for nntile::tf32_t type
template<>
struct CUDAComputeType<tf32_t>
{
#ifdef NNTILE_USE_CUDA_TF32
    using value = float;
#endif
};

//! Compute type for nntile::fp16_t type
template<>
struct CUDAComputeType<fp16_t>
{
#ifdef NNTILE_USE_CUDA_TF32
    using value = __nv_half;
#endif
};

//! Compute type for nntile::bf16_t type
template<>
struct CUDAComputeType<bf16_t>
{
#ifdef NNTILE_USE_CUDA_TF32
    using value = __nv_bfloat16;
#endif
};

//! Compute type for nntile::fp8_e4m3_t type
template<>
struct CUDAComputeType<fp8_e4m3_t>
{
#ifdef NNTILE_USE_CUDA_TF32
    using value = __nv_fp8_e4m3;
#endif
};

//! Compute type for nntile::fp8_e5m2_t type
template<>
struct CUDAComputeType<fp8_e5m2_t>
{
#ifdef NNTILE_USE_CUDA_TF32
    using value = __nv_fp8_e5m2;
#endif
};

} // namespace nntile::kernel

