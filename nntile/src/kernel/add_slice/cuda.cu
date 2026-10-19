/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/add_slice/cuda.cu
 * Per-element addition of a tensor and a broadcasted slice on CUDA
 *
 * @version 1.1.0
 * */

#include <cstdio>
#include "nntile/kernel/add_slice/cuda.hh"
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::add_slice
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index mk, Scalar alpha_,
        const T * __restrict__ src1, Scalar beta_, const T * __restrict__ src2,
        T * __restrict__ dst)
//! Generic implementation of the add_slice operation on CUDA
/*! @copydoc nntile::kernel::add_slice::cuda
 * */
{
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y,
          i2 = threadIdx.z + blockIdx.z*blockDim.z;
    using Y = typename T::repr_t;
    constexpr Y zero = 0.0;
    const Y beta = beta_;
    const Y alpha = alpha_;
    if(i2 < k and i1 < n and i0 < m)
    {
        // Pointer to a corresponding fiber of the input array src2
        const T *src2_fiber = src2 + i1*mk + i0;
        // Pointer to a corresponding fiber of the output array dst
        T *dst_fiber = dst + i1*mk + i0;
        // Value to add to the output fiber
        const Y src1_val = alpha * Y{src1[i1*m+i0]};
        // And update it
        dst_fiber[i2*m] = T{beta * Y{src2_fiber[i2*m]} + src1_val};
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T *src1, Scalar beta, const T *src2, T *dst)
    noexcept
//! Add a tensor and a broadcasted slice with optional scaling on CUDA
/*! This is a global function that does the following operations:
 *      dst[i,l,j] = alpha*src1[i,j] + beta*src2[i,l,j]
 *
 * This function reads both src1 and src2 even if alpha or beta is zero.
 * If alpha is zero and src1[i,j] is NaN, then dst[i,l,j] will be NaN.
 * If beta is zero and src2[i,l,j] is NaN, then dst[i,l,j] will be NaN.
 * If such behaviour is not desired, then in a case of alpha being zero,
 * use nntile::kernel::scale, and in a case of beta being zero,
 * use nntile::kernel::scale_slice instead.
 * If both alpha and beta are zero, then use nntile::kernel::clear instead.
 *
 * @see nntile::kernel::scale
 * @see nntile::kernel::scale_slice
 * @see nntile::kernel::clear
 *
 * @param[in] stream: CUDA stream
 * @param[in] m: Size of the first mode of src1, src2 and dst tensors
 * @param[in] n: Size of the last mode of src1, src2 and dst tensors
 * @param[in] k: Size of the middle mode of src2 and dst tensor
 * @param[in] alpha: Scalar factor for src1
 * @param[in] src1: Input contiguous m-by-n array
 * @param[in] beta: Scaling factor for src1
 * @param[in] src2: Input contiguous m-by-k-by-n array
 * @param[out] dst: Output contiguous m-by-k-by-n array
 * */
{
    // Both source and destination are Fortran-contiguous
    dim3 threads(std::min(int(m), 8), std::min(int(n), 8),
            std::min(int(k), 16));
    dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y,
            (k+threads.z-1)/threads.z);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, m*k, alpha,
            src1, beta, src2, dst);
}

//! Chunk descriptors for the one-launch chunked add_slice (see cuda.hh).
template<typename T>
struct AddSliceChunkTables
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
    Index m;
    Index n;
    Index k;
    Scalar alpha;
    Scalar beta;
};

//! Chunk index of a flat element offset (shift on a power-of-two elems).
static __device__ __forceinline__ Index adsl_chunk_of(long f, Index elems,
    int shift)
{
    return Index(f >> shift);
}

//! Correctly-rounded FMA, to pin the association nvcc chose for cuda().
static __device__ __forceinline__ float adsl_fma(float a, float b, float c)
{
    return __fmaf_rn(a, b, c);
}
static __device__ __forceinline__ double adsl_fma(double a, double b,
    double c)
{
    return __fma_rn(a, b, c);
}

//! In-chunk element offset of flat offset f (mask when elems is pow2).
static __device__ __forceinline__ long adsl_off(long f, Index elems, int shift,
    Index c)
{
    return f & ((1LL << shift) - 1);
}

//! Chunk-aware add_slice: the SAME thread geometry and per-element
//! expression as cuda_kernel, with every operand address resolved through
//! its chunk table (src1 flat i1*m+i0, src2/dst flat i1*m*k+i2*m+i0).
template<typename T>
static __global__
void chunked_cuda_kernel(AddSliceChunkTables<T> t)
{
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y,
          i2 = threadIdx.z + blockIdx.z*blockDim.z;
    using Y = typename T::repr_t;
    const Y beta = t.beta;
    const Y alpha = t.alpha;
    if(i2 < t.k and i1 < t.n and i0 < t.m)
    {
        long const f1 = long(i1)*long(t.m) + long(i0);
        Index const c1 = adsl_chunk_of(f1, t.elems_src1, t.shift_src1);
        const Y s1 = Y{
            t.src1[c1][adsl_off(f1, t.elems_src1, t.shift_src1, c1)]};
        long const f2 = long(i1)*long(t.m)*long(t.k) + long(i2)*long(t.m)
            + long(i0);
        Index const c2 = adsl_chunk_of(f2, t.elems_src2, t.shift_src2);
        const Y s2 = Y{
            t.src2[c2][adsl_off(f2, t.elems_src2, t.shift_src2, c2)]};
        Index const cd = adsl_chunk_of(f2, t.elems_dst, t.shift_dst);
        // Same association as cuda_kernel: fl(beta*src2), then
        // fma(alpha, src1, .).
        Y pr = beta * s2;
        t.dst[cd][adsl_off(f2, t.elems_dst, t.shift_dst, cd)] =
            T{adsl_fma(alpha, s1, pr)};
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T* const* src1_chunks, Index nsrc1, Scalar beta,
        const T* const* src2_chunks, Index nsrc2, T* const* dst_chunks,
        Index ndst, Index elems_src1, Index elems_src2, Index elems_dst)
    noexcept
{
    if(m <= 0 || n <= 0 || k <= 0 || nsrc1 <= 0 || nsrc2 <= 0 || ndst <= 0)
    {
        return;
    }
    AddSliceChunkTables<T> t;
    t.src1 = src1_chunks;
    t.nsrc1 = nsrc1;
    t.src2 = src2_chunks;
    t.nsrc2 = nsrc2;
    t.dst = dst_chunks;
    t.ndst = ndst;
    t.elems_src1 = elems_src1;
    t.elems_src2 = elems_src2;
    t.elems_dst = elems_dst;
    t.shift_src1 = chunk_shift_of<T>(t.nsrc1, elems_src1, "nntile_add_slice");
    t.shift_src2 = chunk_shift_of<T>(t.nsrc2, elems_src2, "nntile_add_slice");
    t.shift_dst = chunk_shift_of<T>(t.ndst, elems_dst, "nntile_add_slice");
    t.m = m;
    t.n = n;
    t.k = k;
    t.alpha = alpha;
    t.beta = beta;
    dim3 threads(std::min(int(m), 8), std::min(int(n), 8),
            std::min(int(k), 16));
    dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y,
            (k+threads.z-1)/threads.z);
    (chunked_cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(t);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp32_t *src, Scalar beta, const fp32_t *src2, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp64_t *src, Scalar beta, const fp64_t *src2, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const bf16_t *src, Scalar beta, const bf16_t *src2, bf16_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp16_t *src, Scalar beta, const fp16_t *src2, fp16_t *dst)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index m, Index n, Index k,
        Scalar alpha, const fp32_t* const* src1_chunks, Index nsrc1,
        Scalar beta, const fp32_t* const* src2_chunks, Index nsrc2,
        fp32_t* const* dst_chunks, Index ndst, Index elems_src1,
        Index elems_src2, Index elems_dst) noexcept;

template
void chunked<fp64_t>(cudaStream_t stream, Index m, Index n, Index k,
        Scalar alpha, const fp64_t* const* src1_chunks, Index nsrc1,
        Scalar beta, const fp64_t* const* src2_chunks, Index nsrc2,
        fp64_t* const* dst_chunks, Index ndst, Index elems_src1,
        Index elems_src2, Index elems_dst) noexcept;

template
void chunked<bf16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Scalar alpha, const bf16_t* const* src1_chunks, Index nsrc1,
        Scalar beta, const bf16_t* const* src2_chunks, Index nsrc2,
        bf16_t* const* dst_chunks, Index ndst, Index elems_src1,
        Index elems_src2, Index elems_dst) noexcept;

template
void chunked<fp16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Scalar alpha, const fp16_t* const* src1_chunks, Index nsrc1,
        Scalar beta, const fp16_t* const* src2_chunks, Index nsrc2,
        fp16_t* const* dst_chunks, Index ndst, Index elems_src1,
        Index elems_src2, Index elems_dst) noexcept;






} // namespace nntile::kernel::add_slice
