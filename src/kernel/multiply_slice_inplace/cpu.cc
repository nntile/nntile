/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/multiply_slice_inplace/cpu.cc
 * CPU kernel for in-place multiplication of a tensor and a broadcasted slice
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/multiply_slice_inplace/cpu.hh"

namespace nntile::kernel::multiply_slice_inplace
{

template<typename T>
void cpu(Index m, Index n, Index k, Scalar alpha_, const T *src, Scalar beta_,
        T *dst)
    noexcept
//! In-place multiplication of a tensor and a broadcasted slice on CPU
/*! Performs the following operations:
 *      dst[i,l,j] = beta * dst[i,l,j] * alpha * src[i,j]
 *
 * @param[in] m: Size of the first mode of dst
 * @param[in] n: Size of the second mode of dst
 * @param[in] k: Size of the third mode of dst
 * @param[in] alpha_: Scalar factor for src
 * @param[in] src: Input slice
 * @param[in] beta_: Scaling factor for dst
 * @param[inout] dst: Resulting tensor
 * */
{
    using Y = typename T::repr_t;
    const Y alpha{alpha_}, beta{beta_};

    // Task body
    for(Index i = 0; i < m; ++i)
    {
        for(Index l = 0; l < n; ++l)
        {
            for(Index j = 0; j < k; ++j)
            {
                Y src_val = Y{src[i*k + j]};
                Y dst_val = Y{dst[i*n*k + l*k + j]};
                dst[i*n*k + l*k + j] = T{beta * dst_val * alpha * src_val};
            }
        }
    }
}

// Explicit instantiation for all supported types
template void cpu<fp32_t>(Index m, Index n, Index k, Scalar alpha, const fp32_t *src, Scalar beta, fp32_t *dst);
template void cpu<fp64_t>(Index m, Index n, Index k, Scalar alpha, const fp64_t *src, Scalar beta, fp64_t *dst);
template void cpu<fp32_fast_tf32_t>(Index m, Index n, Index k, Scalar alpha, const fp32_fast_tf32_t *src, Scalar beta, fp32_fast_tf32_t *dst);
template void cpu<fp32_fast_fp16_t>(Index m, Index n, Index k, Scalar alpha, const fp32_fast_fp16_t *src, Scalar beta, fp32_fast_fp16_t *dst);
template void cpu<fp32_fast_bf16_t>(Index m, Index n, Index k, Scalar alpha, const fp32_fast_bf16_t *src, Scalar beta, fp32_fast_bf16_t *dst);
template void cpu<bf16_t>(Index m, Index n, Index k, Scalar alpha, const bf16_t *src, Scalar beta, bf16_t *dst);
template void cpu<fp16_t>(Index m, Index n, Index k, Scalar alpha, const fp16_t *src, Scalar beta, fp16_t *dst);

} // namespace nntile::kernel::multiply_slice_inplace
