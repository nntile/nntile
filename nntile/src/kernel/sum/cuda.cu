/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/kernel/sum/cuda.cu
 * Sum all elements of a buffer on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/sum/cuda.hh"
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::sum
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, Scalar alpha_, const T *src, Scalar beta_, T *dst)
{
    using Y = typename T::repr_t;
    const Y alpha{alpha_}, beta{beta_};
    constexpr Y zero{0.0};

    // Shared memory for block reduction
    __shared__ Y shared_sum[1024];
    __shared__ Y shared_c[1024];

    int tid = threadIdx.x;
    Y sum = zero, c = zero;

    // Each thread computes partial sum using Kahan summation
    for(Index i = tid; i < nelems; i += blockDim.x)
    {
        Y y = Y{src[i]} - c;
        Y t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    // Store partial results in shared memory
    shared_sum[tid] = sum;
    shared_c[tid] = c;
    __syncthreads();

    // Reduce within block using tree reduction with Kahan summation
    for(int s = blockDim.x/2; s > 0; s >>= 1)
    {
        if(tid < s)
        {
            // Accumulate shared_c[tid+s]
            Y y = shared_c[tid + s] - shared_c[tid];
            Y t = shared_sum[tid] + y;
            shared_c[tid] = (t - shared_sum[tid]) - y;
            shared_sum[tid] = t;
            // Accumulate shared_sum[tid+s]
            y = shared_sum[tid + s] - shared_c[tid];
            t = shared_sum[tid] + y;
            shared_c[tid] = (t - shared_sum[tid]) - y;
            shared_sum[tid] = t;
        }
        __syncthreads();
    }

    // Final result computation by thread 0
    if(tid == 0)
    {
        const Y sum_val = alpha * shared_sum[0];
        if(beta == zero)
        {
            dst[0] = static_cast<T>(sum_val);
        }
        else
        {
            dst[0] = static_cast<T>(
                (beta * Y{dst[0]} - alpha*shared_c[0]) + sum_val);
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar alpha, const T *src,
        Scalar beta, T *dst)
    noexcept
//! Sum all elements of a tensor into a scalar
/*! For a provided input array of nelems elements computes the sum of all
 * elements, resulting in a single scalar output.
 * Mnemonically, the following operations are performed:
 *      dst[0] = beta*dst[0] + alpha*sum(src[:])
 *
 * @param[in] stream: CUDA stream
 * @param[in] nelems: Number of elements in the input array
 * @param[in] alpha: Scaling factor for src
 * @param[in] src: Input contiguous array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Output scalar (single element array)
 * */
{
    // Use a single block with up to 1024 threads
    dim3 threads(1024);
    dim3 blocks(1);

    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, alpha, src, beta, dst);
}

template<typename T>
static __global__
void chunked_cuda_kernel(Index nelems, Scalar alpha_, ChunkConst<T> src,
        Scalar beta_, ChunkMut<T> dst)
{
    using Y = typename T::repr_t;
    const Y alpha{alpha_}, beta{beta_};
    constexpr Y zero{0.0};

    // Shared memory for block reduction
    __shared__ Y shared_sum[1024];
    __shared__ Y shared_c[1024];

    int tid = threadIdx.x;
    Y sum = zero, c = zero;

    // Each thread computes partial sum using Kahan summation
    for(Index i = tid; i < nelems; i += blockDim.x)
    {
        Y y = Y{src.at(i)[0]} - c;
        Y t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    // Store partial results in shared memory
    shared_sum[tid] = sum;
    shared_c[tid] = c;
    __syncthreads();

    // Reduce within block using tree reduction with Kahan summation
    for(int s = blockDim.x/2; s > 0; s >>= 1)
    {
        if(tid < s)
        {
            // Accumulate shared_c[tid+s]
            Y y = shared_c[tid + s] - shared_c[tid];
            Y t = shared_sum[tid] + y;
            shared_c[tid] = (t - shared_sum[tid]) - y;
            shared_sum[tid] = t;
            // Accumulate shared_sum[tid+s]
            y = shared_sum[tid + s] - shared_c[tid];
            t = shared_sum[tid] + y;
            shared_c[tid] = (t - shared_sum[tid]) - y;
            shared_sum[tid] = t;
        }
        __syncthreads();
    }

    // Final result computation by thread 0
    if(tid == 0)
    {
        T *dst_ptr = dst.at(0);
        const Y sum_val = alpha * shared_sum[0];
        if(beta == zero)
        {
            dst_ptr[0] = static_cast<T>(sum_val);
        }
        else
        {
            dst_ptr[0] = static_cast<T>(
                (beta * Y{dst_ptr[0]} - alpha*shared_c[0]) + sum_val);
        }
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index nelems, Scalar alpha,
        const T* const* src_chunks, Index nsrc,
        Scalar beta, T* const* dst_chunks, Index ndst,
        Index elems_src, Index elems_dst, ChunkClass cls)
    noexcept
{
    if(nsrc <= 0 || ndst <= 0)
    {
        return;
    }
    // ONE block of up to 1024 threads, exactly like the contiguous kernel.
    dim3 threads(1024);
    dim3 blocks(1);
    (chunked_cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, alpha,
        chunk_const<T>(src_chunks, nsrc, elems_src, cls, "nntile_sum"),
        beta,
        chunk_mut<T>(dst_chunks, ndst, elems_dst, cls, "nntile_sum"));
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp32_t *src, Scalar beta, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp64_t *src, Scalar beta, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const bf16_t *src, Scalar beta, bf16_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp16_t *src, Scalar beta, fp16_t *dst)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp32_t* const* src_chunks, Index nsrc, Scalar beta,
        fp32_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst, ChunkClass cls)
    noexcept;
template
void chunked<fp64_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp64_t* const* src_chunks, Index nsrc, Scalar beta,
        fp64_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst, ChunkClass cls)
    noexcept;
template
void chunked<bf16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const bf16_t* const* src_chunks, Index nsrc, Scalar beta,
        bf16_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst, ChunkClass cls)
    noexcept;
template
void chunked<fp16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp16_t* const* src_chunks, Index nsrc, Scalar beta,
        fp16_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst, ChunkClass cls)
    noexcept;

} // namespace nntile::kernel::sum
