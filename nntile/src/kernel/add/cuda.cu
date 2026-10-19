/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/add/cuda.cu
 * Add operation on buffers on CUDA
 *
 * @version 1.1.0
 * */

#include <cstdio>
#include "nntile/kernel/add/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::add
{

template<typename T, int BLOCK, int LOOP>
static __global__
void cuda_kernel(
    Index nelems,
    Scalar alpha,
    const T *src1,
    Scalar beta,
    const T *src2,
    T *dst
)
//! Generic implementation of the add operation on CUDA
/* @copydoc nntile::kernel::add::cuda
 * */
{
    int i = threadIdx.x + blockIdx.x*BLOCK;
    using Y = typename T::repr_t;
    Y dst_block[LOOP];
    Y src_block[LOOP];
    const Y alpha_ = alpha;
    const Y beta_ = beta;
    constexpr int BLOCK_STEP = BLOCK / LOOP;
    if((blockIdx.x+1)*BLOCK <= nelems)
    {
        for(int j = 0; j < LOOP; ++j)
        {
            src_block[j] = static_cast<Y>(src1[i+j*BLOCK_STEP]);
            dst_block[j] = static_cast<Y>(src2[i+j*BLOCK_STEP]);
            dst_block[j] = alpha_ * src_block[j] + beta_ * dst_block[j];
            dst[i+j*BLOCK_STEP] = static_cast<T>(dst_block[j]);
        }
    }
    else
    {
        int j_max = (nelems-i+BLOCK_STEP-1) / BLOCK_STEP;
        for(int j = 0; j < j_max; ++j)
        {
            src_block[j] = static_cast<Y>(src1[i+j*BLOCK_STEP]);
            dst_block[j] = static_cast<Y>(src2[i+j*BLOCK_STEP]);
            dst_block[j] = alpha_ * src_block[j] + beta_ * dst_block[j];
            dst[i+j*BLOCK_STEP] = static_cast<T>(dst_block[j]);
        }
    }
}

template<typename T>
void cuda(
    cudaStream_t stream,
    Index nelems,
    Scalar alpha,
    const T *src1,
    Scalar beta,
    const T *src2,
    T *dst
) noexcept
//! Add two buffers with optional scaling on CUDA
/*! Performs the following operation:
 * dst[i] = alpha*src1[i] + beta*src2[i]
 *
 * This function reads both src1 and src2 even if alpha or beta is zero.
 * If alpha is zero and src1[i] is NaN, then dst[i] will be NaN.
 * If beta is zero and src2[i] is NaN, then dst[i] will be NaN.
 * If such behaviour is not desired, then in a case of alpha or beta being
 * zero, use nntile::kernel::scale instead.
 * If both alpha and beta are zero, then use nntile::kernel::clear instead.
 *
 * @see nntile::kernel::scale
 * @see nntile::kernel::clear
 *
 * @param[in] stream: CUDA stream
 * @param[in] nelems: Size of the src and dst tensors
 * @param[in] alpha: Scalar multiplier for the src1 tensor
 * @param[in] src1: First source tensor
 * @param[in] beta: Scalar multiplier for the src2 tensor
 * @param[in] src2: Second source tensor
 * @param[out] dst: Destination tensor
 * */
{
    dim3 threads(256);
    dim3 blocks((nelems+1023)/1024);
    (cuda_kernel<T, 1024, 4>)<<<blocks, threads, 0, stream>>>(
        nelems, alpha, src1, beta, src2, dst);
}

//! Chunk descriptors for the one-launch chunked add (see cuda.hh).
template<typename T>
struct AddChunkTables
{
    const T* const* src1; //!< device-ring view of the src1 pointer array
    Index nsrc1;
    const T* const* src2; //!< device-ring view of the src2 pointer array
    Index nsrc2;
    T* const* dst; //!< device-ring view of the dst pointer array
    Index ndst;
    Index elems_src1; //!< elements in a full src1 chunk
    Index elems_src2; //!< elements in a full src2 chunk
    Index elems_dst; //!< elements in a full dst chunk
    int shift_src1; //!< log2(elems_src1) on a power of two, else -1
    int shift_src2; //!< log2(elems_src2) on a power of two, else -1
    int shift_dst; //!< log2(elems_dst) on a power of two, else -1
    long nelems;
    Scalar alpha;
    Scalar beta;
};

//! Chunk index of a flat element offset (shift on a power-of-two elems).
static __device__ __forceinline__ Index add_chunk_of(long f, Index elems,
    int shift)
{
    return Index(f >> shift);
}

//! Correctly-rounded FMA, to pin the association nvcc chose for cuda().
static __device__ __forceinline__ float add_fma(float a, float b, float c)
{
    return __fmaf_rn(a, b, c);
}
static __device__ __forceinline__ double add_fma(double a, double b, double c)
{
    return __fma_rn(a, b, c);
}

//! In-chunk element offset of flat offset f (mask when elems is pow2).
static __device__ __forceinline__ long add_off(long f, Index elems, int shift,
    Index c)
{
    return f & ((1LL << shift) - 1);
}

//! Chunk-aware add: the SAME flat thread mapping (i = tid + bid*1024,
//! f = i + j*256) and the SAME per-element expression as cuda_kernel, with
//! every operand address resolved through its chunk table.
template<typename T, int BLOCK, int LOOP>
static __global__
void chunked_cuda_kernel(AddChunkTables<T> t)
{
    int i = threadIdx.x + blockIdx.x*BLOCK;
    using Y = typename T::repr_t;
    const Y alpha = t.alpha;
    const Y beta = t.beta;
    constexpr int BLOCK_STEP = BLOCK / LOOP;
    for(int j = 0; j < LOOP; ++j)
    {
        long f = long(i) + long(j)*BLOCK_STEP;
        if(f < t.nelems)
        {
            Index c1 = add_chunk_of(f, t.elems_src1, t.shift_src1);
            Y v1 = Y{t.src1[c1][add_off(f, t.elems_src1, t.shift_src1, c1)]};
            Index c2 = add_chunk_of(f, t.elems_src2, t.shift_src2);
            Y v2 = Y{t.src2[c2][add_off(f, t.elems_src2, t.shift_src2, c2)]};
            Index cd = add_chunk_of(f, t.elems_dst, t.shift_dst);
            // Same association as cuda_kernel: fl(beta*src2), then
            // fma(alpha, src1, .).
            Y pr = beta * v2;
            t.dst[cd][add_off(f, t.elems_dst, t.shift_dst, cd)] =
                T{add_fma(alpha, v1, pr)};
        }
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index nelems, Scalar alpha,
        const T* const* src1_chunks, Index nsrc1, Scalar beta,
        const T* const* src2_chunks, Index nsrc2, T* const* dst_chunks,
        Index ndst, Index elems_src1, Index elems_src2, Index elems_dst)
    noexcept
{
    if(nelems <= 0 || nsrc1 <= 0 || nsrc2 <= 0 || ndst <= 0)
    {
        return;
    }
    AddChunkTables<T> t;
    t.src1 = src1_chunks;
    t.nsrc1 = nsrc1;
    t.src2 = src2_chunks;
    t.nsrc2 = nsrc2;
    t.dst = dst_chunks;
    t.ndst = ndst;
    t.elems_src1 = elems_src1;
    t.elems_src2 = elems_src2;
    t.elems_dst = elems_dst;
    t.shift_src1 = chunk_shift_of<T>(t.nsrc1, elems_src1, "nntile_add");
    t.shift_src2 = chunk_shift_of<T>(t.nsrc2, elems_src2, "nntile_add");
    t.shift_dst = chunk_shift_of<T>(t.ndst, elems_dst, "nntile_add");
    t.nelems = long(nelems);
    t.alpha = alpha;
    t.beta = beta;
    dim3 threads(256);
    dim3 blocks((nelems+1023)/1024);
    (chunked_cuda_kernel<T, 1024, 4>)<<<blocks, threads, 0,
        stream>>>(t);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp32_t *src1, Scalar beta, const fp32_t *src2, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp64_t *src1, Scalar beta, const fp64_t *src2, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const bf16_t *src1, Scalar beta, const bf16_t *src2, bf16_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp16_t *src1, Scalar beta, const fp16_t *src2, fp16_t *dst)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp32_t* const* src1_chunks, Index nsrc1, Scalar beta,
        const fp32_t* const* src2_chunks, Index nsrc2,
        fp32_t* const* dst_chunks, Index ndst, Index elems_src1,
        Index elems_src2, Index elems_dst) noexcept;

template
void chunked<fp64_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp64_t* const* src1_chunks, Index nsrc1, Scalar beta,
        const fp64_t* const* src2_chunks, Index nsrc2,
        fp64_t* const* dst_chunks, Index ndst, Index elems_src1,
        Index elems_src2, Index elems_dst) noexcept;

template
void chunked<bf16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const bf16_t* const* src1_chunks, Index nsrc1, Scalar beta,
        const bf16_t* const* src2_chunks, Index nsrc2,
        bf16_t* const* dst_chunks, Index ndst, Index elems_src1,
        Index elems_src2, Index elems_dst) noexcept;

template
void chunked<fp16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp16_t* const* src1_chunks, Index nsrc1, Scalar beta,
        const fp16_t* const* src2_chunks, Index nsrc2,
        fp16_t* const* dst_chunks, Index ndst, Index elems_src1,
        Index elems_src2, Index elems_dst) noexcept;






} // namespace nntile::kernel::add
