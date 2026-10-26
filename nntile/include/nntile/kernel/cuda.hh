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

// Entire contents of this file is enabled iff CUDA is enabled
#ifdef NNTILE_USE_CUDA

// FP16/BF16 are supported due to CUDA 11.0 minimal requirement
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// FP8 formats may be not supported by CUDA Toolkit
#ifdef NNTILE_USE_CUDA_FP8
#   include <cuda_fp8.h>
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifdef NNTILE_USE_NNHAUL
// One source of truth for the chunk class (rule 7): the kernel-side
// constant below must equal nnhaul::kChunkClassBytes in every NNHaul
// build (kernel translation units see the public macro through the
// target's PUBLIC definition). The kernel files must also build in the
// StarPU configuration, where the nnhaul headers are absent -- hence
// the restatement inside the namespace instead of an alias.
#include <nnhaul/nnhaul.hh>
#endif

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
    using value = __nv_bfloat16;
};

template<>
struct CUDAComputeType<fp16_t>
{
    using value = __half;
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

//! Check CUDA error and throw runtime_error if error occurs
#define CUDA_CHECK(error, message) \
    do { \
        cudaError_t _err = (error); \
        if (_err != cudaSuccess) { \
            throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(_err)); \
        } \
    } while (0)

// ---- (rule 7) the compile-time chunk class of the chunked kernels ----
//
// The ONE chunk class: 16 MiB, a power of two, identical to
// nnhaul::kChunkClassBytes (static_asserted in the NNHaul build below;
// restated because the kernel translation units also build in the
// StarPU configuration, where the nnhaul headers are absent). A
// multi-chunk operand's full chunk is exactly the class -- the nnhaul
// policy cannot create anything else and a violation aborts loudly
// (chunk_shift_of) -- so the chunk-index divmod lowers to shift/mask
// on the class. A single-chunk operand uses the sentinel shift 62:
// every flat offset stays in chunk 0 (f >> 62 == 0, f & mask == f),
// which is correct for a contiguous buffer of ANY size. Scope of the
// "no runtime divisor" claim: the address-resolution HELPERS here and
// the chunked kernels' per-element/per-span chunk resolutions --
// including the piece-4 m1 walk span starts and rebases (sum_fiber,
// sum_slice, multiply_slice, sumprod_fiber, multiply_fiber) -- are
// pure shift/mask; the historical shift = -1 fallback (a runtime
// 64-bit division whose expansion serialized the address path, 4.58x
// on sumprod_fiber) is deleted. What still divides by a runtime
// value: only the straddling-RUN room arithmetic of the walks
// ((elems - off + step - 1) / step, once per run end, never per
// element) and arithmetic unrelated to chunk residency (grid tiling
// and the like).
#ifdef NNTILE_USE_NNHAUL
// The configured class comes from the installed nnhaul headers
// (nnhaul/defs.h, generated by CMake from NNHAUL_CHUNK_SIZE_MB), so
// the engine and the kernels cannot disagree. Standalone StarPU builds
// keep the historical default.
inline constexpr std::size_t kChunkClassBytes = ::nnhaul::kChunkClassBytes;
#else
inline constexpr std::size_t kChunkClassBytes = std::size_t{16} << 20;
#endif

//! Exact log2 of a power of two (constexpr, v must be a power of two).
inline constexpr int chunk_log2(std::size_t v) noexcept
{
    int s = 0;
    while ((v >>= 1) != 0)
    {
        ++s;
    }
    return s;
}

//! Sentinel shift of a single-chunk operand: everything in chunk 0.
inline constexpr int kChunkSingleShift = 62;

//! Per-operand chunk shift (rule 7): the class shift for a multi-chunk
//! operand -- aborting loudly unless its full chunk is exactly the
//! class (the invariant the nnhaul side guarantees; hand-built tables
//! that violate it are caught here) -- and the single-chunk sentinel
//! otherwise. elems is the operand's full-chunk element count the
//! caller derived from its buffer view.
template<typename T>
inline int chunk_shift_of(std::size_t nchunks, std::size_t elems,
    char const *op) noexcept
{
    constexpr std::size_t class_elems = kChunkClassBytes / sizeof(T);
    if (nchunks > 1)
    {
        if (elems != class_elems)
        {
            std::fprintf(stderr,
                "nntile %s: chunked operand full chunk %zu elements != "
                "class %zu (rule 7 compile-time 16 MiB class)\n",
                op, elems, class_elems);
            std::abort();
        }
        return chunk_log2(class_elems);
    }
    return kChunkSingleShift;
}

} // namespace nntile::kernel

#endif // NNTILE_USE_CUDA
