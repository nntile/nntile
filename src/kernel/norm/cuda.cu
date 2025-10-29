/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/norm/cuda.cu
 * Euclidean norm of all elements in a buffer on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/norm/cuda.hh"
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::norm
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, Scalar alpha_, const T *src, Scalar beta_, T *dst)
{
    using Y = typename T::repr_t;
    const Y alpha = static_cast<Y>(alpha_), beta = static_cast<Y>(beta_);
    constexpr Y zero = 0.0, one = 1.0;

    // Simple reduction: compute sum of squares
    __shared__ Y shared_norm[1024];

    int tid = threadIdx.x;
    Y norm_max = zero;
    Y norm_ssq = zero;
    Y c = zero;

    // Each thread computes partial sum of squares
    for(Index i = tid; i < nelems; i += blockDim.x)
    {
        Y val = ::fabs(static_cast<Y>(src[i]));
        if(val > 0)
        {
            if(norm_max >= val)
            {
                Y tmp1 = val / norm_max;
                Y y = tmp1*tmp1 - c;
                Y t = norm_ssq + y;
                c = (t-norm_ssq) - y;
                norm_ssq = t;
            }
            else
            {
                Y tmp1 = norm_max / val;
                Y tmp2 = tmp1 * tmp1;
                Y y = one - c*tmp2;
                norm_ssq *= tmp2;
                Y t = norm_ssq + y;
                c = (t-norm_ssq) - y;
                norm_ssq = t;
                norm_max = val;
            }
        }
    }

    // Store partial sum
    if (norm_ssq > zero)
    {
        shared_norm[tid] = norm_max * ::sqrt(norm_ssq);
    }
    else
    {
        shared_norm[tid] = zero;
    }
    __syncthreads();

    // Reduce within block (only active threads participate)
    for(int s = blockDim.x/2; s > 0; s >>= 1)
    {
        if(tid < s)
        {
            shared_norm[tid] = ::hypot(shared_norm[tid], shared_norm[tid + s]);
        }
        __syncthreads();
    }

    // Final result computation by thread 0
    if(tid == 0)
    {
        Y norm_val = shared_norm[0] * ::fabs(alpha);
        if(beta == zero)
        {
            dst[0] = norm_val;
        }
        else
        {
            Y dst_val = static_cast<Y>(dst[0]);
            dst[0] = ::hypot(norm_val, beta * dst_val);
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar alpha, const T *src,
        Scalar beta, T *dst)
    noexcept
//! Euclidean norm of all elements in a buffer (out-of-place version)
/*! For a provided array src of nelems elements compute the Euclidean norm
 * and combine it with the existing dst[0] value.
 * Mnemonically, the following operations are performed:
 *      dst[0] = alpha * norm(src[...]) + beta * dst[0]
 *
 * @param[in] stream: CUDA stream
 * @param[in] nelems: Number of elements in src array
 * @param[in] alpha: Scaling factor for the norm
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

} // namespace nntile::kernel::norm
