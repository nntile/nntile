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
template<typename T>
struct SubtractIndexedOutputsChunkTables
{
    const int64_t *const *labels; //!< device view of labels pointer array
    Index nlab;
    const int64_t *labels0; //!< piece C: direct base when labels is 1-chunk
    T *const *dst; //!< device view of dst pointer array
    Index ndst;
    Index elems_lab;
    Index elems_dst;
    int shift_lab;
    int shift_dst;
    Index n_labels;
    Index n_outputs;
    Index ignore_index;
    Scalar val;
};

//! Chunk index of a flat element offset (power-of-two shift when possible).
static __device__ __forceinline__ Index sio_chunk_of(long f, Index elems,
    int shift)
{
    return Index(f >> shift);
}

//! (piece C: the element-wise table helpers sio_get/sio_set are gone --
//! the update branch resolves its chunk once and reads/writes through
//! the resolved pointer, halving the ring-slot hops of the
//! latency-bound 4-block kernel; the ignore branch already walks runs.)

//! Chunk-aware subtract_indexed_outputs: thread i acts only on output column
//! i exactly as cuda() does (zero-fill on ignore, else dst[labels[i], i] -=
//! val), through the shared chunk tables, so the result is bitwise cuda()'s.
template<typename T>
static __global__
void chunked_cuda_kernel(SubtractIndexedOutputsChunkTables<T> t)
{
    // No shared-memory staging: the tables live in the read-only device
    // ring (L2-resident) and every thread touches at most one labels
    // pointer and one dst pointer, so reading the ring directly is cheaper
    // than a per-block fill plus a __syncthreads (the grid is tiny here).
    const int64_t *const *sh_l = t.labels;
    T *const *sh_d = t.dst;

    long const i = long(blockIdx.x)*blockDim.x + threadIdx.x;
    if(i >= t.n_outputs)
    {
        return;
    }
    using Y = typename T::repr_t;
    std::int64_t lab;
    if(t.labels0 != nullptr)
    {
        // Piece C (2026-10-18c): a single-chunk labels operand reads its
        // resolved base directly -- one dependent load instead of the
        // ring-table hop plus the element load.
        lab = t.labels0[i].value;
    }
    else if(t.nlab == 1)
    {
        lab = sh_l[0][i].value;
    }
    else
    {
        Index const lc = sio_chunk_of(i, t.elems_lab, t.shift_lab);
        lab = sh_l[lc][i - long(lc) * long(t.elems_lab)].value;
    }
    long const base = i * long(t.n_labels);
    if(lab == t.ignore_index)
    {
        // Zero the whole column through its chunk bounds (same result as
        // cuda()'s memset of the contiguous column).
        long l = 0;
        while(l < t.n_labels)
        {
            long const f = base + l;
            Index const c = sio_chunk_of(f, t.elems_dst, t.shift_dst);
            long const cend = (long(c) + 1) * long(t.elems_dst);
            long n = cend - f;
            if(n > long(t.n_labels) - l)
            {
                n = long(t.n_labels) - l;
            }
            T *p = sh_d[c] + (f - long(c) * long(t.elems_dst));
            for(long k = 0; k < n; ++k)
            {
                p[k] = T{0.0};
            }
            l += n;
        }
    }
    else
    {
        // Piece C (2026-10-18c): resolve the dst chunk ONCE for the
        // read-modify-write (the old form paid the ring-slot hop twice
        // -- once inside sio_get, once inside sio_set -- for the same
        // element of a latency-bound 4-block kernel). The in-chunk
        // offset is the rule-7 mask, not a multiply. Same address, same
        // value, same order: bitwise identical.
        long const f = base + long(lab);
        Index const c = sio_chunk_of(f, t.elems_dst, t.shift_dst);
        T *p = sh_d[c] + (f & ((1LL << t.shift_dst) - 1));
        Y const dv = Y{p[0]};
        p[0] = static_cast<T>(dv - Y{t.val});
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
    SubtractIndexedOutputsChunkTables<T> t;
    t.labels = labels_chunks;
    t.nlab = nlab;
    t.labels0 = labels0;
    t.dst = dst_chunks;
    t.ndst = ndst;
    t.elems_lab = elems_lab;
    t.elems_dst = elems_dst;
    t.shift_lab = chunk_shift_of<::int64_t>(t.nlab, elems_lab, "nntile_subtract_indexed_outputs");
    t.shift_dst = chunk_shift_of<T>(t.ndst, elems_dst, "nntile_subtract_indexed_outputs");
    t.n_labels = n_labels;
    t.n_outputs = n_outputs;
    t.ignore_index = ignore_index;
    t.val = val;
    // Same geometry as the whole-buffer entry: one thread per output column.
    // No dynamic shared memory (tables are read straight from the ring).
    dim3 blocks((n_outputs + 255) / 256), threads(256);
    (chunked_cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(t);
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
