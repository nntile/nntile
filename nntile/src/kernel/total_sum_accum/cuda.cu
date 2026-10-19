/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/total_sum_accum/cuda.cu
 * total_sum_accum operation on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/total_sum_accum/cuda.hh"
#include <cstdio>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::total_sum_accum
{

template<typename T>
static __global__
void cuda_kernel(Scalar alpha, Index n_labels, Index n_outputs, Index ignore_index,
        const T *logsumexp, const T *src, const Index *labels, float *val)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    if(i < n_outputs)
    {
        __shared__ Y block_val;
        if(threadIdx.x == 0)
        {
            block_val = 0;
        }
        __syncthreads();
        Y val1 = static_cast<Y>(logsumexp[i]);
        Y val2 = val1;
        if (labels[i] != ignore_index)
            val2 = static_cast<Y>(src[labels[i] + i*n_labels]);
        atomicAdd(&block_val, val1-val2);
        __syncthreads();
        if(threadIdx.x == 0)
        {
            atomicAdd(val, static_cast<float>(alpha*block_val));
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Scalar alpha, Index n_labels, Index n_outputs,
        Index ignore_index, const T *logsumexp, const T *src, const int64_t *labels_,
        float *val)
    noexcept
//! Total sum accumulating from logsumexp and corrected by elements from src
/*! Mnemonically, the following operations are performed:
 * for every i in [0, n_outputs)
 *      val += alpha * (logsumexp[i]-src[labels[i], i]);
 *
 * @param[in] alpha: Scalar multiplier
 * @param[in] n_labels: Number of possible labels
 * @param[in] n_outputs: Number of elements to sum up.
 * @param[in] logsumexp: Array with logsumexp values of size n_outputs.
 * @param[in] src: Matrix of size n_labels times n_outputs stored continuously
 *      in Fortran order
 * @param[in] labels: Array of size n_outputs with correct labels
 * @param[inout] val: Scalar that accumulates the total sum
 * */
{
    dim3 blocks((n_outputs+255)/256), threads(256);
    using I = typename CUDAComputeType<int64_t>::value;
    auto labels = reinterpret_cast<const I *>(labels_);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(alpha, n_labels,
            n_outputs, ignore_index, logsumexp, src, labels, val);
}

//! Chunk descriptors for the one-launch chunked total_sum_accum.
template<typename T>
struct TotalSumAccumChunkTables
{
    const T *const *lse; //!< device view of logsumexp pointer array
    Index nlse;
    const T *const *src; //!< device view of src pointer array
    Index nsrc;
    const int64_t *const *labels; //!< device view of labels pointer array
    Index nlab;
    float *const *val; //!< device view of val pointer array
    Index nval;
    Index elems_lse;
    Index elems_src;
    Index elems_lab;
    Index elems_val;
    int shift_lse;
    int shift_src;
    int shift_lab;
    int shift_val;
    Index n_labels;
    Index n_outputs;
    Index ignore_index;
    Scalar alpha;
};

//! Chunk index of a flat element offset (power-of-two shift when possible).
static __device__ __forceinline__ Index tsa_chunk_of(long f, Index elems,
    int shift)
{
    return Index(f >> shift);
}

//! Read element f of an operand through its shared chunk table.
template<typename T>
static __device__ __forceinline__ T tsa_get(const T *const *sh, Index elems,
    int shift, long f)
{
    Index const c = tsa_chunk_of(f, elems, shift);
    return sh[c][f & ((1LL << shift) - 1)];
}

//! Chunk-aware total_sum_accum: thread i reads the same per-column values as
//! cuda() and reduces them with the same block-shared atomicAdd grouping, so
//! the result is cuda()'s up to the same float-atomic ordering
//! nondeterminism.
template<typename T>
static __global__
void chunked_cuda_kernel(TotalSumAccumChunkTables<T> t)
{
    long const i = long(blockIdx.x)*blockDim.x + threadIdx.x;
    using Y = typename T::repr_t;
    if(i < t.n_outputs)
    {
        __shared__ Y block_val;
        if(threadIdx.x == 0)
        {
            block_val = 0;
        }
        __syncthreads();
        Y val1 = static_cast<Y>(tsa_get(t.lse, t.elems_lse, t.shift_lse,
            i));
        Y val2 = val1;
        Index const lc = tsa_chunk_of(i, t.elems_lab, t.shift_lab);
        std::int64_t const lab = t.labels[lc][i - long(lc) * long(t.elems_lab)].value;
        if(lab != t.ignore_index)
        {
            val2 = static_cast<Y>(tsa_get(t.src, t.elems_src, t.shift_src,
                long(lab) + i*long(t.n_labels)));
        }
        atomicAdd(&block_val, val1-val2);
        __syncthreads();
        if(threadIdx.x == 0)
        {
            atomicAdd(t.val[0], static_cast<float>(t.alpha*block_val));
        }
    }
}

//! (piece C, 2026-10-18c) Mixed-shape variant for launches whose small
//! operands (logsumexp, labels, val) are all SINGLE-CHUNK while src is
//! multi-chunk -- the total_sum_accum graph shape (lse 4 KiB, labels
//! 8 KiB, val 4 B beside the 196.32 MiB src). The single-chunk bases
//! travel as plain kernel parameters (the piece-4 degenerate-launch
//! resolution applied per operand: a 1-chunk table is its own base
//! pointer, so the ring keeps serving only the genuinely multi-chunk
//! src). This removes one dependent ring-slot load per operand per
//! thread from the serial chain of this 4-block latency-bound kernel;
//! the read of src keeps its table hop (the row is a random scatter,
//! no run structure to walk). Accumulation order, grouping and the
//! block_val staging are byte-identical to chunked_cuda_kernel, so the
//! result is the same up to the same float-atomic ordering (bitwise on
//! the grouping; the bar for a reduction).
template<typename T>
static __global__
void chunked_smalls_cuda_kernel(Scalar alpha, Index n_labels,
    Index n_outputs, Index ignore_index,
    const T *lse, const T *const *src, Index elems_src, int shift_src,
    const int64_t *labels, float *val)
{
    long const i = long(blockIdx.x)*blockDim.x + threadIdx.x;
    using Y = typename T::repr_t;
    if(i < n_outputs)
    {
        __shared__ Y block_val;
        if(threadIdx.x == 0)
        {
            block_val = 0;
        }
        __syncthreads();
        Y val1 = static_cast<Y>(lse[i]);
        Y val2 = val1;
        std::int64_t const lab = labels[i].value;
        if(lab != ignore_index)
        {
            val2 = static_cast<Y>(tsa_get(src, elems_src, shift_src,
                long(lab) + i*long(n_labels)));
        }
        atomicAdd(&block_val, val1-val2);
        __syncthreads();
        if(threadIdx.x == 0)
        {
            atomicAdd(val, static_cast<float>(alpha*block_val));
        }
    }
}

template<typename T>
void chunked(cudaStream_t stream, Scalar alpha, Index n_labels,
        Index n_outputs, Index ignore_index,
        const T* const* lse_chunks, Index nlse,
        const T* const* src_chunks, Index nsrc,
        const int64_t* const* labels_chunks, Index nlab,
        float* const* val_chunks, Index nval,
        Index elems_lse, Index elems_src, Index elems_lab, Index elems_val,
        const T *lse0, const int64_t *labels0, float *val0)
    noexcept
{
    if(n_labels <= 0 or n_outputs <= 0 or nlse <= 0 or nsrc <= 0 or nlab <= 0
        or nval <= 0)
    {
        return;
    }
    // Piece C: mixed shape (small operands single-chunk, src
    // multi-chunk) takes the smalls-direct variant: the 1-chunk bases
    // arrive as plain parameters (the degenerate resolution per
    // operand), only src reads the ring. Falls back to the fully
    // tabular kernel whenever any small operand is multi-chunk or the
    // direct bases were not supplied.
    if(lse0 != nullptr && labels0 != nullptr && val0 != nullptr)
    {
        int const shift_src = chunk_shift_of<T>(nsrc, elems_src,
            "nntile_total_sum_accum");
        dim3 blocks((n_outputs + 255) / 256), threads(256);
        (chunked_smalls_cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(
            alpha, n_labels, n_outputs, ignore_index,
            lse0, src_chunks, elems_src, shift_src, labels0, val0);
        return;
    }
    TotalSumAccumChunkTables<T> t;
    t.lse = lse_chunks;
    t.nlse = nlse;
    t.src = src_chunks;
    t.nsrc = nsrc;
    t.labels = labels_chunks;
    t.nlab = nlab;
    t.val = val_chunks;
    t.nval = nval;
    t.elems_lse = elems_lse;
    t.elems_src = elems_src;
    t.elems_lab = elems_lab;
    t.elems_val = elems_val;
    t.shift_lse = chunk_shift_of<T>(t.nlse, elems_lse, "nntile_total_sum_accum");
    t.shift_src = chunk_shift_of<T>(t.nsrc, elems_src, "nntile_total_sum_accum");
    t.shift_lab = chunk_shift_of<::int64_t>(t.nlab, elems_lab, "nntile_total_sum_accum");
    t.shift_val = chunk_shift_of<T>(t.nval, elems_val, "nntile_total_sum_accum");
    t.n_labels = n_labels;
    t.n_outputs = n_outputs;
    t.ignore_index = ignore_index;
    t.alpha = alpha;
    // Same geometry as the whole-buffer entry: one thread per column.
    // Rule 10: no dynamic shared memory (the tables are read from the
    // ring directly; only the static block_val reduction cell remains).
    dim3 blocks((n_outputs + 255) / 256), threads(256);
    (chunked_cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(t);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Scalar alpha, Index n_labels,
        Index n_outputs, Index ignore_index, const fp32_t* logsumexp, const fp32_t* src,
        const int64_t* labels, float *val)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Scalar alpha, Index n_labels,
        Index n_outputs, Index ignore_index, const fp64_t* logsumexp, const fp64_t* src,
        const int64_t* labels, float *val)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Scalar alpha, Index n_labels,
        Index n_outputs, Index ignore_index, const bf16_t* logsumexp, const bf16_t* src,
        const int64_t* labels, float *val)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Scalar alpha, Index n_labels,
        Index n_outputs, Index ignore_index, const fp16_t* logsumexp, const fp16_t* src,
        const int64_t* labels, float *val)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Scalar alpha, Index n_labels,
        Index n_outputs, Index ignore_index, const fp32_t *const *lse_chunks,
        Index nlse, const fp32_t *const *src_chunks, Index nsrc,
        const int64_t *const *labels_chunks, Index nlab,
        float *const *val_chunks, Index nval, Index elems_lse, Index elems_src,
        Index elems_lab, Index elems_val, const fp32_t *lse0,
        const int64_t *labels0, float *val0) noexcept;

template
void chunked<fp64_t>(cudaStream_t stream, Scalar alpha, Index n_labels,
        Index n_outputs, Index ignore_index, const fp64_t *const *lse_chunks,
        Index nlse, const fp64_t *const *src_chunks, Index nsrc,
        const int64_t *const *labels_chunks, Index nlab,
        float *const *val_chunks, Index nval, Index elems_lse, Index elems_src,
        Index elems_lab, Index elems_val, const fp64_t *lse0,
        const int64_t *labels0, float *val0) noexcept;

template
void chunked<bf16_t>(cudaStream_t stream, Scalar alpha, Index n_labels,
        Index n_outputs, Index ignore_index, const bf16_t *const *lse_chunks,
        Index nlse, const bf16_t *const *src_chunks, Index nsrc,
        const int64_t *const *labels_chunks, Index nlab,
        float *const *val_chunks, Index nval, Index elems_lse, Index elems_src,
        Index elems_lab, Index elems_val, const bf16_t *lse0,
        const int64_t *labels0, float *val0) noexcept;

template
void chunked<fp16_t>(cudaStream_t stream, Scalar alpha, Index n_labels,
        Index n_outputs, Index ignore_index, const fp16_t *const *lse_chunks,
        Index nlse, const fp16_t *const *src_chunks, Index nsrc,
        const int64_t *const *labels_chunks, Index nlab,
        float *const *val_chunks, Index nval, Index elems_lse, Index elems_src,
        Index elems_lab, Index elems_val, const fp16_t *lse0,
        const int64_t *labels0, float *val0) noexcept;





} // namespace nntile::kernel::total_sum_accum
