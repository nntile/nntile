/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/subtract_indexed_outputs/cuda.cu
 * subtract_indexed_output operation on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/subtract_indexed_outputs/cuda.hh"
#include <cstdio>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::subtract_indexed_outputs
{

template<typename T>
static __global__
void cuda_kernel(Index n_labels, Index n_outputs, Index ignore_index,
                 Scalar val_, const Index* labels, T *dst)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    Y dst_val{0.0};
    Y val{val_};
    if(i < n_outputs)
    {
        if (labels[i] == ignore_index) {
            ::memset(dst + i*n_labels, 0, n_labels * sizeof(T));
            return;
        }
        dst_val = Y{dst[labels[i] + i*n_labels]};
        dst[labels[i] + i*n_labels] = dst_val - val;
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index n_labels, Index n_outputs, Index ignore_index, Scalar val,
        const int64_t *labels_, T *dst_)
    noexcept
//! Subtraction of given val from indexed output of dst
/*! Mnemonically, the following operations are performed:
 *      dst[labels[i], i] -= val
 * for every i in [0, n_outputs)
 *
 * @param[in] n_labels: Number of possible labels
 * @param[in] n_outputs: Number of matrix elemets to update
 * @param[in] val: Value that is subtracted from the matrix elements
 * @param[in] labels_: Index array of size n_outputs
 * @param[inout] dst_: Matrix of size n_labels by n_outputs continuously stored
 *      in Fortran order
 * */
{
    dim3 blocks((n_outputs+255)/256), threads(256);
    using I = typename CUDAComputeType<int64_t>::value;
    auto labels = reinterpret_cast<const I *>(labels_);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(n_labels, n_outputs, ignore_index,
            val, labels, dst_);
}

//! Chunk descriptors for the one-launch chunked subtract_indexed_outputs.
//! (piece 16: the fat struct is gone -- the kernel takes slim scalar
//! parameters; see chunked_cuda_kernel.)

//! Chunk index of a flat element offset (power-of-two shift when possible).
static __device__ __forceinline__ Index sio_chunk_of(long f, Index elems,
    int shift)
{
    return Index(f >> shift);
}

//! Chunk-aware subtract_indexed_outputs: thread i acts only on output column
//! i exactly as cuda() does (zero-fill on ignore, else dst[labels[i], i] -=
//! val), through the shared chunk tables, so the result is bitwise cuda()'s
//! (no atomics, no value freedom).
//!
//! (piece 16) Slim scalar parameters replace the fat descriptor struct, and
//! the dst chunk base for the label-free flat offset (i*n_labels) is
//! SPECULATED at kernel start: the load of dst_t[cg] issues in parallel
//! with the label fetch instead of chaining behind it, and is used whenever
//! the label keeps the column inside that chunk (the label adds < 1 chunk
//! for any label < elems_dst, which is every valid label of a column-sized
//! row -- the common case; the miss path falls back to the true table
//! slot). This takes the second dependent ring-slot hop off the critical
//! chain of the latency-bound 4-block kernel: label-table slot, label
//! element, read-modify-write -- the same depth as cuda() plus one uniform
//! slot load. c = f >> shift >= cg always (labels[i] >= 0 or the ignore
//! branch), and cg < ndst because base < n_outputs*n_labels = the flat
//! element count, so the speculative slot is always in bounds.
template<typename T>
static __global__
void chunked_cuda_kernel(Index n_labels, Index n_outputs, Index ignore_index,
    Scalar val, const int64_t *labels0, const int64_t *const *lab_t,
    Index elems_lab, int shift_lab,
    T *const *dst_t, Index elems_dst, int shift_dst)
{
    long const i = long(blockIdx.x)*blockDim.x + threadIdx.x;
    if(i >= n_outputs)
    {
        return;
    }
    long const base = i * long(n_labels);
    long const cg = base >> shift_dst;
    long const cgoff = cg << shift_dst;
    // Read the chunk-pointer table through the non-coherent read-only path:
    // it is immutable for the kernel's lifetime, never aliases dst, and this
    // keeps the table lines in the texture/read-only cache instead of
    // competing with the streaming RMW traffic in L2.
    T *const pg = dst_t[cg];
    using Y = typename T::repr_t;
    std::int64_t lab;
    if(labels0 != nullptr)
    {
        // Piece C: a single-chunk labels operand passed as a resolved
        // base -- one dependent load, no table hop.
        lab = labels0[i].value;
    }
    else if(shift_lab == kChunkSingleShift)
    {
        // Single-chunk labels operand: one uniform slot load, then the
        // element (one dependent hop instead of slot+divmod+element).
        lab = lab_t[0][i].value;
    }
    else
    {
        Index const lc = sio_chunk_of(i, elems_lab, shift_lab);
        lab = lab_t[lc][i - long(lc) * long(elems_lab)].value;
    }
    if(lab == ignore_index)
    {
        // Zero the whole column through its chunk bounds (same result as
        // cuda()'s memset of the contiguous column).
        long l = 0;
        while(l < n_labels)
        {
            long const f = base + l;
            Index const c = sio_chunk_of(f, elems_dst, shift_dst);
            long const cend = (long(c) + 1) * long(elems_dst);
            long n = cend - f;
            if(n > n_labels - l)
            {
                n = n_labels - l;
            }
            T *p = dst_t[c] + (f - long(c) * long(elems_dst));
            #pragma unroll 1
            for(long k = 0; k < n; ++k)
            {
                p[k] = T{0.0};
            }
            l += n;
        }
    }
    else
    {
        long const f = base + long(lab);
        long const c = f >> shift_dst;
        T *p = (c == cg) ? pg + (f - cgoff)
                         : dst_t[c] + (f - (c << shift_dst));
        Y const dv = Y{p[0]};
        p[0] = static_cast<T>(dv - Y{val});
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index n_labels, Index n_outputs,
        Index ignore_index, Scalar val,
        const int64_t* const* labels_chunks, Index nlab,
        T* const* dst_chunks, Index ndst, Index elems_lab, Index elems_dst,
        const int64_t *labels0)
    noexcept
{
    if(n_labels <= 0 or n_outputs <= 0 or nlab <= 0 or ndst <= 0)
    {
        return;
    }
    int const shift_lab = chunk_shift_of<::int64_t>(nlab, elems_lab,
        "nntile_subtract_indexed_outputs");
    int const shift_dst = chunk_shift_of<T>(ndst, elems_dst,
        "nntile_subtract_indexed_outputs");
    // Same work assignment as the whole-buffer entry (one thread per output
    // column), but spread over more blocks: the chunked kernel's address
    // chain is one dependent load deeper, so giving the 1024 columns more
    // SMs shortens the tail instead of serializing it on 4 SMs.
    dim3 blocks((n_outputs + 63) / 64), threads(64);
    (chunked_cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(n_labels,
        n_outputs, ignore_index, val, labels0, labels_chunks, elems_lab,
        shift_lab, dst_chunks, elems_dst, shift_dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index n_labels, Index n_outputs, Index ignore_index,
        Scalar val, const int64_t *labels, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index n_labels, Index n_outputs, Index ignore_index,
        Scalar val, const int64_t *labels, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index n_labels, Index n_outputs, Index ignore_index,
        Scalar val, const int64_t *labels, bf16_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index n_labels, Index n_outputs, Index ignore_index,
        Scalar val, const int64_t *labels, fp16_t *dst)
    noexcept;

template
void cuda<fp32_fast_bf16_t>(cudaStream_t stream, Index n_labels, Index n_outputs, Index ignore_index,
        Scalar val, const int64_t *labels, fp32_fast_bf16_t *dst)
    noexcept;

template
void cuda<fp32_fast_tf32_t>(cudaStream_t stream, Index n_labels, Index n_outputs, Index ignore_index,
        Scalar val, const int64_t *labels, fp32_fast_tf32_t *dst)
    noexcept;

template
void cuda<fp32_fast_fp16_t>(cudaStream_t stream, Index n_labels, Index n_outputs, Index ignore_index,
        Scalar val, const int64_t *labels, fp32_fast_fp16_t *dst)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index n_labels, Index n_outputs,
        Index ignore_index, Scalar val, const int64_t *const *labels_chunks,
        Index nlab, fp32_t *const *dst_chunks, Index ndst, Index elems_lab,
        Index elems_dst, const int64_t *labels0) noexcept;

template
void chunked<fp64_t>(cudaStream_t stream, Index n_labels, Index n_outputs,
        Index ignore_index, Scalar val, const int64_t *const *labels_chunks,
        Index nlab, fp64_t *const *dst_chunks, Index ndst, Index elems_lab,
        Index elems_dst, const int64_t *labels0) noexcept;

template
void chunked<bf16_t>(cudaStream_t stream, Index n_labels, Index n_outputs,
        Index ignore_index, Scalar val, const int64_t *const *labels_chunks,
        Index nlab, bf16_t *const *dst_chunks, Index ndst, Index elems_lab,
        Index elems_dst, const int64_t *labels0) noexcept;

template
void chunked<fp16_t>(cudaStream_t stream, Index n_labels, Index n_outputs,
        Index ignore_index, Scalar val, const int64_t *const *labels_chunks,
        Index nlab, fp16_t *const *dst_chunks, Index ndst, Index elems_lab,
        Index elems_dst, const int64_t *labels0) noexcept;





} // namespace nntile::kernel::subtract_indexed_outputs
