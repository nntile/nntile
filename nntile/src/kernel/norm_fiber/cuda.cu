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
void cuda_kernel(Index m, Index n, Index k, Index batch, Scalar alpha, const T *src1,
        Scalar beta, const T *src2, T *dst)
//! Sums over slices along the first and last axes into a fiber of a tensor
/*! For a provided m-by-k-by-n input array computes sums over slices
 * along the first axis with m elements and the last axis with n elements,
 * resulting in output fiber of shape (k).
 * Mnemonically, the following operations are performed:
 *      dst[l,b] = hypot(beta*src1[l,b], alpha*norm(src2[:,l,:,b]))
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
    using Y = typename T::repr_t;
    constexpr Y zero{0.0};

    // Calculate global thread ID for batch and k dimensions
    Index i2_batched = blockIdx.x * blockDim.x + threadIdx.x;
    Index i2 = i2_batched % k;
    Index b = i2_batched / k;

    if (b < batch && i2 < k)
    {
        Y sum = zero;

        // Compute the hypot over the entire m*n slice for the current k and batch
        for (Index i1 = 0; i1 < n; ++i1)
        {
            for (Index i0 = 0; i0 < m; ++i0)
            {
                Index src_idx = i0 + i2 * m + i1 * m * k + b * m * k * n;
                Y val = Y{src1[src_idx]};
                sum = ::hypot(sum, val);
            }
        }

        // Apply alpha scaling and beta scaling, then update dst
        Y result;
        if (beta == zero)
        {
            result = Y{::fabs(alpha) * sum};
        }
        else
        {
            Y old_val = Y{src2[i2_batched]};
            result = Y{::hypot(beta * old_val, alpha * sum)};
        }
        dst[i2_batched] = T{result};
    }
}


template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const T *src1, Scalar beta, const T *src2, T *dst)
    noexcept
//! Sums over slices along the first and last axes into a fiber of a tensor
/*! For a provided m-by-k-by-n input array computes sums over slices
 * along the first axis with m elements and the last axis with n elements,
 * resulting in output fiber of shape (k).
 * Mnemonically, the following operations are performed:
 *      dst[k,b] = beta*src1[k,b] + alpha*sum(src2[:,k,:,b])
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
            src1, beta, src2, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const fp32_t *src1, Scalar beta, const fp32_t *src2, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const fp64_t *src1, Scalar beta, const fp64_t *src2, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const bf16_t *src1, Scalar beta, const bf16_t *src2, bf16_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const fp16_t *src1, Scalar beta, const fp16_t *src2, fp16_t *dst)
    noexcept;

} // namespace nntile::kernel::norm_fiber
