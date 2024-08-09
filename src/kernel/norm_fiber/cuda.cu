/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/norm_fiber/cuda.cu
 * Sums over slices into a fiber of a buffer on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/norm_fiber/cuda.hh"
#include <cmath>
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::norm_fiber
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index batch, Scalar alpha, const T *src,
        Scalar beta, T *dst)
//! Sums over slices along the first and last axes into a fiber of a tensor
/*! For a provided m-by-k-by-n input array computes sums over slices
 * along the first axis with m elements and the last axis with n elements,
 * resulting in output fiber of shape (k).
 * Mnemonically, the following operations are performed:
 *      dst[l,b] = hypot(beta*dst[l,b], alpha*norm(src[:,l,:,b]))
 *
 * @param[in] m: Size of the first mode of src array
 * @param[in] n: Size of the last mode of src array
 * @param[in] k: Size of the middle mode of src array and the only mode of
 *      dst array
 * @param[in] batch: Size of the batch dimension
 * @param[in] alpha: Scaling factor for src
 * @param[in] src: Input contiguous m-by-k-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Output contiguous vector with k elements, that accumulate
 *      norm over slices along the first and the last axes.
 * */
{
    Index i2_batched = threadIdx.x + blockIdx.x*blockDim.x;
    Index i2 = i2_batched % k;
    Index b = i2_batched / k;
    Index i0_start = threadIdx.y, i0_step = blockDim.y;
    Index i1_start = threadIdx.z, i1_step = blockDim.z;
    using Y = typename T::repr_t;
    constexpr Y zero{0.0};
    // Init sum
    Y sum = zero;
    if(b < batch)
    {
        // Cycle over the third axis of input buffer
        for(Index i1 = i1_start; i1 < n; i1 += i1_step)
        {
            // Get sum of a corresponding slice
            const T *src_slice = src + ((i1+b*n)*k+i2)*m;
            // Cycle over the first axis of input buffer
            for(Index i0 = i0_start; i0 < m; i0 += i0_step)
            {
                // Read value from source and Update sum
                sum = ::hypot(sum, Y{src_slice[i2*m]});
            }
        }
        __shared__ Y block_sum[1];
        if(i1_start == 0 and i0_start == 0)
        {
            block_sum[threadIdx.x] = zero;
        }
        __syncthreads();
        atomicAdd(&block_sum[threadIdx.x], sum);
        __syncthreads();
        // Update output value
        if(i1_start == 0 and i0_start == 0)
        {
            // Save result
            sum = block_sum[threadIdx.x];
            if(beta == zero)
            {
                sum = Y{::fabs(alpha) * sum};
            }
            else
            {
                sum = Y{::hypot(beta*Y{dst[i2_batched]}, alpha*sum)};
            }
            dst[i2_batched] = Y{sum};
        }
    }

}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const T *src, Scalar beta, T *dst)
    noexcept
//! Sums over slices along the first and last axes into a fiber of a tensor
/*! For a provided m-by-k-by-n input array computes sums over slices
 * along the first axis with m elements and the last axis with n elements,
 * resulting in output fiber of shape (k).
 * Mnemonically, the following operations are performed:
 *      dst[k,b] = beta*dst[k,b] + alpha*sum(src[:,k,:,b])
 *
 * @param[in] m: Size of the first mode of src array
 * @param[in] n: Size of the last mode of src array
 * @param[in] k: Size of the middle mode of src array and the only mode of
 *      dst array
 * @param[in] batch: Size of the batch dimension
 * @param[in] alpha: Scaling factor for src
 * @param[in] src: Input contiguous m-by-k-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Output contiguous vector with k elements, that accumulate
 *      sums over slices along the first and the last axes.
 * */
{
    // Both source and destination are Fortran-contiguous
    dim3 threads(1, std::min(int(m), 32), std::min(int(n), 32));
    dim3 blocks((k*batch+threads.x-1)/threads.x, 1, 1);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, batch, alpha,
            src, beta, dst);
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
void cuda<fp32_fast_tf32_t>(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const fp32_fast_tf32_t *src, Scalar beta, fp32_fast_tf32_t *dst)
    noexcept;

} // namespace nntile::kernel::norm_fiber
