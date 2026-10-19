/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/logsumexp/cuda.cu
 * Logsumexp after computed maxsumexp result of a buffer on CUDA
 *
 * @version 1.3.0
 * */

#include "nntile/kernel/logsumexp/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::logsumexp
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, const T * __restrict__ maxsumexp,
        T * __restrict__ logsumexp)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    if(i < nelems)
    {
        Y maxsumexp_even = Y{maxsumexp[2*i]};
        Y maxsumexp_odd = Y{maxsumexp[2*i+1]};
        logsumexp[i] = maxsumexp_even + ::log(maxsumexp_odd);
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, const T *maxsumexp_,
        T *logsumexp_)
    noexcept
//! Logsumexp of buffer
/*! One of the buffers serves as output
 *
 * @param[in] nelems: Number of elements in both buffers
 * @param[in] maxsumexp_: Input buffer, result of maxsumexp operation
 * @param[out] logsumexp_: Output buffers that contains output in the end
 * */
{
    dim3 blocks((nelems+31)/32), threads(32);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, maxsumexp_,
            logsumexp_);
}

//! Chunk descriptors for the one-launch chunked logsumexp (see cuda.hh).
template<typename T>
struct ChunkTables
{
    const T *const* mse; //!< device-ring view of the maxsumexp ptr array
    Index nmse;
    T *const* lse; //!< device-ring view of the logsumexp ptr array
    Index nlse;
    Index elems_mse; //!< elements in a full maxsumexp chunk
    Index elems_lse; //!< elements in a full logsumexp chunk
    int shift_mse; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
    int shift_lse; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
};

//! Chunk index of a flat element offset (power-of-two shift when possible).
static __device__ __forceinline__ long lse_chunk_index(long f, Index elems,
    int shift) noexcept
{
    return f >> shift;
}

//! Chunk-aware logsumexp: the SAME flat thread mapping (i = tid + bid*32)
//! and the SAME per-element expression (an add of a log() call, so there
//! is no FMA-contraction freedom) as cuda_kernel. Element i reads the
//! maxsumexp PAIR at mse-flat offsets 2i and 2i+1 -- the two halves may
//! live in different maxsumexp chunks -- and writes logsumexp[i]; every
//! address is resolved through its operand's chunk table read straight
//! from the device ring (piece 9 / rule 10: no staging). The result is bitwise cuda()'s.
template<typename T>
static __global__
void chunked_cuda_kernel(Index nelems, ChunkTables<T> t)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    if(i < nelems)
    {
        long const fe = 2*long(i);
        long const fo = fe + 1;
        Index const ce = Index(lse_chunk_index(fe, t.elems_mse, t.shift_mse));
        Index const co = Index(lse_chunk_index(fo, t.elems_mse, t.shift_mse));
        Index const cl = Index(lse_chunk_index(long(i), t.elems_lse, t.shift_lse));
        Y maxsumexp_even = Y{t.mse[ce][fe - ce * long(t.elems_mse)]};
        Y maxsumexp_odd = Y{t.mse[co][fo - co * long(t.elems_mse)]};
        t.lse[cl][long(i) - cl * long(t.elems_lse)] =
            maxsumexp_even + ::log(maxsumexp_odd);
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index nelems,
        const T* const* maxsumexp_chunks, Index nmse,
        T* const* logsumexp_chunks, Index nlse, Index elems_mse,
        Index elems_lse)
    noexcept
{
    if(nelems <= 0 || nmse <= 0 || nlse <= 0)
    {
        return;
    }
    ChunkTables<T> t;
    t.mse = maxsumexp_chunks;
    t.nmse = nmse;
    t.lse = logsumexp_chunks;
    t.nlse = nlse;
    t.elems_mse = elems_mse;
    t.elems_lse = elems_lse;
    t.shift_mse = chunk_shift_of<T>(t.nmse, elems_mse, "nntile_logsumexp");
    t.shift_lse = chunk_shift_of<T>(t.nlse, elems_lse, "nntile_logsumexp");
    dim3 blocks((nelems+31)/32), threads(32);
    (chunked_cuda_kernel<T>)<<<blocks, threads,
        0, stream>>>(nelems, t);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, const fp32_t *maxsumexp,
        fp32_t *logsumexp)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, const fp64_t *maxsumexp,
        fp64_t *logsumexp)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems, const bf16_t *maxsumexp,
        bf16_t *logsumexp)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index nelems, const fp16_t *maxsumexp,
        fp16_t *logsumexp)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index nelems,
        const fp32_t* const* maxsumexp_chunks, Index nmse,
        fp32_t* const* logsumexp_chunks, Index nlse, Index elems_mse,
        Index elems_lse) noexcept;

template
void chunked<fp64_t>(cudaStream_t stream, Index nelems,
        const fp64_t* const* maxsumexp_chunks, Index nmse,
        fp64_t* const* logsumexp_chunks, Index nlse, Index elems_mse,
        Index elems_lse) noexcept;

template
void chunked<bf16_t>(cudaStream_t stream, Index nelems,
        const bf16_t* const* maxsumexp_chunks, Index nmse,
        bf16_t* const* logsumexp_chunks, Index nlse, Index elems_mse,
        Index elems_lse) noexcept;

template
void chunked<fp16_t>(cudaStream_t stream, Index nelems,
        const fp16_t* const* maxsumexp_chunks, Index nmse,
        fp16_t* const* logsumexp_chunks, Index nlse, Index elems_mse,
        Index elems_lse) noexcept;

} // namespace nntile::kernel::logsumexp
