/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/add_fiber_inplace/cuda.cu
 * Per-element addition of a tensor and a broadcasted fiber on CUDA
 *
 * @version 1.1.0
 * */

#include <cstdio>
#include "nntile/kernel/add_fiber_inplace/cuda.hh"
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::add_fiber_inplace
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index batch, Scalar alpha_, const T *src,
        Scalar beta_, T *dst)
//! Generic implementation of the add_fiber_inplace operation on CUDA
/*! @copydoc nntile::kernel::add_fiber_inplace::cuda
 * */
{
    Index i2 = threadIdx.x + blockIdx.x*blockDim.x,
          i0 = threadIdx.y + blockIdx.y*blockDim.y,
          i1 = threadIdx.z + blockIdx.z*blockDim.z;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    const Y beta{beta_};
    if(i2 < k and i1 < n and i0 < m)
    {
        for(Index b = 0; b < batch; ++b)
        {
            // Value to add to the output slice
            const Y src_val = alpha * Y{src[i2+b*k]};
            // Output fiber to be updated
            T *dst_fiber = dst + ((i1+b*n)*k+i2)*m;
            // Read value from the output
            T &dst_val = dst_fiber[i0];
            // And update it
            dst_val = T{beta * Y{dst_val} + src_val};
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const T *src, Scalar beta, T *dst)
    noexcept
//! Add a broadcasted fiber into a tensor inplace with optional scaling on CUDA
/*! Performs the following operations:
 *      dst[i,l,j] = beta*dst[i,l,j] + alpha*src[l]
 *
 * This function reads both src and dst even if alpha or beta is zero.
 * If alpha is zero and src[l,b] is NaN, then dst[i,l,j,b] will be NaN.
 * If beta is zero and dst[i,l,j,b] is NaN, then dst[i,l,j,b] will be NaN.
 * If such behaviour is not desired, then in a case of alpha being zero,
 * use nntile::kernel::scale_inplace, and in a case of beta being zero,
 * use nntile::kernel::scale_fiber instead.
 * If both alpha and beta are zero, then use nntile::kernel::clear instead.
 *
 * @see nntile::kernel::scale_inplace
 * @see nntile::kernel::scale_fiber
 * @see nntile::kernel::clear
 *
 * @param[in] m: Size of the first mode of dst tensor
 * @param[in] n: Size of the last mode of dst tensor
 * @param[in] k: Size of the middle mode of dst tensor and the only mode of src
 *      tensors
 * @param[in] batch: Size of the batch dimension
 * @param[in] alpha: Scalar factor for src
 * @param[in] src: Input contiguous vector with k elements
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Input and output contiguous m-by-k-by-n array
 * */
{
    // Both source and destination are Fortran-contiguous
    dim3 threads(std::min(int(k), 1024), std::min(int(m), 1),
            std::min(int(n), 1));
    dim3 blocks((k+threads.x-1)/threads.x, (m+threads.y-1)/threads.y,
            (n+threads.z-1)/threads.z);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, batch, alpha,
            src, beta, dst);
}

//! Chunk descriptors for the one-launch chunked add_fiber_inplace (cuda.hh).
template<typename T>
struct AddFiberInplaceChunkTables
{
    const T* const* src; //!< device-ring view of the src pointer array
    Index nsrc;
    T* const* dst; //!< device-ring view of the dst pointer array
    Index ndst;
    Index elems_src; //!< elements in a full src chunk
    Index elems_dst; //!< elements in a full dst chunk
    int shift_src; //!< log2(elems_src) on a power of two, else -1
    int shift_dst; //!< log2(elems_dst) on a power of two, else -1
    Index m;
    Index n;
    Index k;
    Index batch;
    Scalar alpha;
    Scalar beta;
};

//! Chunk index of a flat element offset (shift on a power-of-two elems).
static __device__ __forceinline__ Index adfbi_chunk_of(long f, Index elems,
    int shift)
{
    return Index(f >> shift);
}

//! Correctly-rounded FMA, to pin the association nvcc chose for cuda().
static __device__ __forceinline__ float adfbi_fma(float a, float b, float c)
{
    return __fmaf_rn(a, b, c);
}
static __device__ __forceinline__ double adfbi_fma(double a, double b,
    double c)
{
    return __fma_rn(a, b, c);
}

//! In-chunk element offset of flat offset f (mask when elems is pow2).
static __device__ __forceinline__ long adfbi_off(long f, Index elems,
    int shift, Index c)
{
    return f & ((1LL << shift) - 1);
}

//! Chunk-aware add_fiber_inplace: the SAME thread geometry and per-element
//! expression as cuda_kernel, with every operand address resolved through
//! its chunk table (src flat i2+b*k, dst flat ((i1+b*n)*k+i2)*m+i0). The
//! kernel reads dst even for beta == 0, as cuda_kernel does.
template<typename T>
static __global__
void chunked_cuda_kernel(AddFiberInplaceChunkTables<T> t)
{
    Index i2 = threadIdx.x + blockIdx.x*blockDim.x,
          i0 = threadIdx.y + blockIdx.y*blockDim.y,
          i1 = threadIdx.z + blockIdx.z*blockDim.z;
    using Y = typename T::repr_t;
    const Y alpha = t.alpha;
    const Y beta = t.beta;
    if(i2 < t.k and i1 < t.n and i0 < t.m)
    {
        for(Index b = 0; b < t.batch; ++b)
        {
            long const f1 = long(i2) + long(b)*long(t.k);
            Index const cs = adfbi_chunk_of(f1, t.elems_src, t.shift_src);
            const Y sv = Y{t.src[cs][adfbi_off(f1, t.elems_src, t.shift_src,
                cs)]};
            long const f2 = (long(i1) + long(b)*long(t.n))*long(t.k)*long(t.m)
                + long(i2)*long(t.m) + long(i0);
            Index const cd = adfbi_chunk_of(f2, t.elems_dst, t.shift_dst);
            T *dp = t.dst[cd] + adfbi_off(f2, t.elems_dst, t.shift_dst, cd);
            // Same association as cuda_kernel: fl(beta*dst), then
            // fma(alpha, src, .).
            Y pr = beta * Y{dp[0]};
            dp[0] = T{adfbi_fma(alpha, sv, pr)};
        }
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const T* const* src_chunks, Index nsrc, Scalar beta,
        T* const* dst_chunks, Index ndst, Index elems_src, Index elems_dst)
    noexcept
{
    if(m <= 0 || n <= 0 || k <= 0 || batch <= 0 || nsrc <= 0 || ndst <= 0)
    {
        return;
    }
    AddFiberInplaceChunkTables<T> t;
    t.src = src_chunks;
    t.nsrc = nsrc;
    t.dst = dst_chunks;
    t.ndst = ndst;
    t.elems_src = elems_src;
    t.elems_dst = elems_dst;
    t.shift_src = chunk_shift_of<T>(t.nsrc, elems_src, "nntile_add_fiber_inplace");
    t.shift_dst = chunk_shift_of<T>(t.ndst, elems_dst, "nntile_add_fiber_inplace");
    t.m = m;
    t.n = n;
    t.k = k;
    t.batch = batch;
    t.alpha = alpha;
    t.beta = beta;
    dim3 threads(std::min(int(k), 1024), std::min(int(m), 1),
            std::min(int(n), 1));
    dim3 blocks((k+threads.x-1)/threads.x, (m+threads.y-1)/threads.y,
            (n+threads.z-1)/threads.z);
    (chunked_cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(t);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const fp32_t *src, Scalar beta, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const fp64_t *src, Scalar beta, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const bf16_t *src, Scalar beta, bf16_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const fp16_t *src, Scalar beta, fp16_t *dst)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index batch, Scalar alpha, const fp32_t* const* src_chunks, Index nsrc,
        Scalar beta, fp32_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst) noexcept;

template
void chunked<fp64_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index batch, Scalar alpha, const fp64_t* const* src_chunks, Index nsrc,
        Scalar beta, fp64_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst) noexcept;

template
void chunked<bf16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index batch, Scalar alpha, const bf16_t* const* src_chunks, Index nsrc,
        Scalar beta, bf16_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst) noexcept;

template
void chunked<fp16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index batch, Scalar alpha, const fp16_t* const* src_chunks, Index nsrc,
        Scalar beta, fp16_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst) noexcept;






} // namespace nntile::kernel::add_fiber_inplace
