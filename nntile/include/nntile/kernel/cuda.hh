/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file include/nntile/kernel/cuda.hh
 * CUDA-related compute types
 *
 * @version 2.0.0
 * */

#pragma once

#include <nntile/defs.h>

// Entire contents of this file is enabled iff CUDA is enabled
#ifdef NNTILE_USE_CUDA

// FP16/BF16 are supported due to CUDA 11.0 minimal requirement
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cuda_fp8.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

// One source of truth for the chunk class: the kernel-side
// constant below must equal nnhaul::kChunkClassBytes (kernel translation
// units see the public macro through the target's PUBLIC definition).
// The value is restated inside the namespace to keep the public header
// free of an engine alias.
#include <nnhaul/nnhaul.hh>

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

// ----  the runtime chunk class of the chunked kernels ----
//
// The ONE chunk class is a power of two and a RUNTIME value: the
// process-wide size is initialized by nnhaul from the
// NNHAUL_CHUNK_SIZE_MB environment variable. Because it is a power of
// two, every operand's flat offset resolves its chunk with a shift and
// its in-chunk offset with an AND mask; both travel to the kernels as
// runtime data (the per-operand shift/mask fields of their parameter
// tables). A single-chunk operand uses the sentinel shift 62 (every
// flat offset stays in chunk 0) and an all-ones mask.
//
// The chunk class is always the configured nnhaul chunk class; chunked
// kernels are dispatched only through NNHaul.
//
// Scope of the "no runtime divisor" claim is unchanged: chunk paging
// is pure shift/mask; only straddling-run room arithmetic divides.

//! Exact log2 of a power of two (v must be a power of two).
//! __host__ __device__: chunked kernels call it from __global__ code.
__host__ __device__
inline constexpr int chunk_log2(std::size_t v) noexcept
{
    int s = 0;
    while ((v >>= 1) != 0)
    {
        ++s;
    }
    return s;
}

//! The runtime chunk class, in its shift/mask form (bytes).
struct ChunkClass
{
    std::size_t mask = 0; //!< class size - 1, the AND mask
    int shift = 0;        //!< log2(class size), the shift count
};

//! The process-wide runtime chunk class (size - 1 and log2(size)).
inline ChunkClass chunk_class() noexcept
{
    return ChunkClass{::nnhaul::chunk_class_mask(),
        ::nnhaul::chunk_class_shift()};
}

//! Sentinel shift of a single-chunk operand: everything in chunk 0.
inline constexpr int kChunkSingleShift = 62;

//! Per-operand chunk shift: the class shift minus log2(sizeof(T)) for
//! a multi-chunk operand -- aborting loudly unless its full chunk is
//! exactly the runtime class (guaranteed by the runtime side) -- and
//! the single-chunk sentinel otherwise. `elems` is
//! the operand's full-chunk element count the caller derived from its
//! buffer view; `cls` is the runtime class (mask + shift in bytes).
template<typename T>
inline int chunk_shift_of(std::size_t nchunks, std::size_t elems,
    ChunkClass cls, char const *op) noexcept
{
    if (nchunks > 1)
    {
        std::size_t const class_elems = (cls.mask + 1) / sizeof(T);
        if (elems != class_elems)
        {
            std::fprintf(stderr,
                "nntile %s: chunked operand full chunk %zu elements != "
                "runtime class %zu (%zu-byte class)\n",
                op, elems, class_elems, cls.mask + 1);
            std::abort();
        }
        return cls.shift - chunk_log2(sizeof(T));
    }
    return kChunkSingleShift;
}

//! Compatibility overload: resolve against the process-wide runtime
//! class. Chunked drivers that take an explicit class call the
//! four-argument form above.
template<typename T>
inline int chunk_shift_of(std::size_t nchunks, std::size_t elems,
    char const *op) noexcept
{
    return chunk_shift_of<T>(nchunks, elems, chunk_class(), op);
}

//! Per-operand AND mask: the class byte mask divided by sizeof(T) for
//! a multi-chunk operand, all ones (the shift-62 sentinel's mask) for
//! a single-chunk operand.
template<typename T>
inline std::size_t chunk_mask_of(std::size_t nchunks,
    ChunkClass cls) noexcept
{
    return nchunks > 1 ? cls.mask / sizeof(T) : ~std::size_t{0};
}

//! Read-only chunked operand view for one-launch kernels.
template<typename T>
struct ChunkConst
{
    const T* const* ptrs;
    int shift;
    long mask;

    __device__ __forceinline__ const T *at(long i) const
    {
        return ptrs[i >> shift] + (i & mask);
    }
};

//! Mutable chunked operand view for one-launch kernels.
template<typename T>
struct ChunkMut
{
    T* const* ptrs;
    int shift;
    long mask;

    __device__ __forceinline__ T *at(long i) const
    {
        return ptrs[i >> shift] + (i & mask);
    }
};

template<typename T>
inline ChunkConst<T> chunk_const(const T* const* ptrs, std::size_t nchunks,
    std::size_t elems, ChunkClass cls, char const *op) noexcept
{
    return ChunkConst<T>{ptrs, chunk_shift_of<T>(nchunks, elems, cls, op),
        static_cast<long>(chunk_mask_of<T>(nchunks, cls))};
}

template<typename T>
inline ChunkMut<T> chunk_mut(T* const* ptrs, std::size_t nchunks,
    std::size_t elems, ChunkClass cls, char const *op) noexcept
{
    return ChunkMut<T>{ptrs, chunk_shift_of<T>(nchunks, elems, cls, op),
        static_cast<long>(chunk_mask_of<T>(nchunks, cls))};
}

} // namespace nntile::kernel

#endif // NNTILE_USE_CUDA
