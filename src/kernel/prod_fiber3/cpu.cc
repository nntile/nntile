/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/prod_fiber3/cpu.cc
 * Per-element multiplication of a tensor by a broadcasted fiber on CPU
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/prod_fiber3/cpu.hh"

namespace nntile::kernel::prod_fiber3
{

template<typename T>
void cpu(Index m, Index n, Index k, T alpha, const T *src1, const T *src2,
        T *dst)
    noexcept
//! Per-element product of a tensor and a broadcasted fiber on CPU
/*! Performs the following operations:
 *      dst[i,l,j] = alpha * src1[l] * src2[i,l,j]
 *
 * @param[in] m: Size of the first mode of dst tensor
 * @param[in] n: Size of the last mode of dst tensor
 * @param[in] k: Size of the middle mode of dst tensor and the only mode of src
 *      tensor
 * @param[in] alpha: Scalar factor
 * @param[in] src1: Input contiguous vector with k elements
 * @param[in] src2: Input contiguous m-by-k-by-n array
 * @param[out] dst: Output contiguous m-by-k-by-n array
 * */
{
    // Cycle over input src vector
    for(Index i2 = 0; i2 < k; ++i2)
    {
        const T src1_val = alpha * src1[i2];
        // Cycle over the third axis of output buffer
        for(Index i1 = 0; i1 < n; ++i1)
        {
            // Input fiber to be used
            const T *src2_fiber = src2 + (i1*k+i2)*m;
            // Output fiber to be updated
            T *dst_fiber = dst + (i1*k+i2)*m;
            // Cycle over the output fiber
            for(Index i0 = 0; i0 < m; ++i0)
            {
                // Update output value
                dst_fiber[i0] = src1_val * src2_fiber[i0];
            }
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index n, Index k, fp32_t alpha, const fp32_t *src1,
        const fp32_t *src2, fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, Index k, fp64_t alpha, const fp64_t *src1,
        const fp64_t *src2, fp64_t *dst)
    noexcept;

} // namespace nntile::kernel::prod_fiber3
