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
void cpu(Index m, Index n, Index k, Scalar alpha_, const T *src,
        T *dst)
    noexcept
//! In-place multiplication of a tensor and a broadcasted slice on CPU
/*! Performs the following operations:
 *      dst[i,l,j] = alpha * dst[i,l,j] * src[i,j]
 *
 * @param[in] m: Size of the first mode of src and dst tensors
 * @param[in] n: Size of the last mode of src and dst tensors
 * @param[in] k: Size of the middle mode of dst tensor
 * @param[in] alpha_: Scalar factor
 * @param[in] src: Input contiguous m-by-n array
 * @param[inout] dst: Input and output contiguous m-by-k-by-n array
 * */
{
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    const Index mk = m * k;
    // Cycle over column of the output buffer dst
    for(Index i2 = 0; i2 < n; ++i2)
    {
        // Cycle over row of the output buffer dst
        for(Index i1 = 0; i1 < m; ++i1)
        {
            // Pointer to a corresponding fiber of the output array dst
            T *dst_fiber = dst + i2*mk + i1;
            // Value to multiply by the output fiber
            const Y src_val = alpha * Y{src[i2*m+i1]};
            // Cycle over output fiber elements
            for(Index i0 = 0; i0 < k; ++i0)
            {
                dst_fiber[i0] *= src_val;
            }
        }
    }
}

// Explicit instantiation for all supported types
template void cpu<fp32_t>(Index m, Index n, Index k, Scalar alpha, const fp32_t *src, fp32_t *dst);
template void cpu<fp64_t>(Index m, Index n, Index k, Scalar alpha, const fp64_t *src, fp64_t *dst);
template void cpu<fp32_fast_tf32_t>(Index m, Index n, Index k, Scalar alpha, const fp32_fast_tf32_t *src, fp32_fast_tf32_t *dst);
template void cpu<fp32_fast_fp16_t>(Index m, Index n, Index k, Scalar alpha, const fp32_fast_fp16_t *src, fp32_fast_fp16_t *dst);
template void cpu<fp32_fast_bf16_t>(Index m, Index n, Index k, Scalar alpha, const fp32_fast_bf16_t *src, fp32_fast_bf16_t *dst);
template void cpu<bf16_t>(Index m, Index n, Index k, Scalar alpha, const bf16_t *src, bf16_t *dst);
template void cpu<fp16_t>(Index m, Index n, Index k, Scalar alpha, const fp16_t *src, fp16_t *dst);

} // namespace nntile::kernel::multiply_slice_inplace
