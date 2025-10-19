/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/multiply_slice_inplace/cuda.cu
 * CUDA kernel for in-place multiplication of a tensor and a broadcasted slice
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/multiply_slice_inplace/cuda.hh"

namespace nntile::kernel::multiply_slice_inplace
{

template<typename T>
__global__ void cuda_kernel(Index m, Index n, Index k, Scalar alpha_, const T *src, Scalar beta_, T *dst)
{
    using Y = typename T::repr_t;
    const Y alpha{alpha_}, beta{beta_};

    Index i = threadIdx.x + blockIdx.x * blockDim.x;
    Index l = threadIdx.y + blockIdx.y * blockDim.y;
    Index j = threadIdx.z + blockIdx.z * blockDim.z;
    if (i < m && l < n && j < k)
    {
        Y src_val = Y{src[i*k + j]};
        Y dst_val = Y{dst[i*n*k + l*k + j]};
        dst[i*n*k + l*k + j] = T{beta * dst_val * alpha * src_val};
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha_,
        const T *src, Scalar beta_, T *dst)
    noexcept
//! In-place multiplication of a tensor and a broadcasted slice on CUDA
/*! Performs the following operations:
 *      dst[i,l,j] = beta * dst[i,l,j] * alpha * src[i,j]
 *
 * @param[in] stream: CUDA stream
 * @param[in] m: Size of the first mode of dst
 * @param[in] n: Size of the second mode of dst
 * @param[in] k: Size of the third mode of dst
 * @param[in] alpha: Scalar factor for src
 * @param[in] src: Input slice
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Resulting tensor
 * */
{
    dim3 threads(8, 8, 8);
    dim3 blocks((m + threads.x - 1) / threads.x,
                (n + threads.y - 1) / threads.y,
                (k + threads.z - 1) / threads.z);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, alpha_, src, beta_, dst);
}

// Explicit instantiation for all supported types
template void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha, const fp32_t *src, Scalar beta, fp32_t *dst);
template void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha, const fp64_t *src, Scalar beta, fp64_t *dst);
template void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha, const bf16_t *src, Scalar beta, bf16_t *dst);
template void cuda<fp16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha, const fp16_t *src, Scalar beta, fp16_t *dst);

} // namespace nntile::kernel::multiply_slice_inplace