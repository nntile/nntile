/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/prod_fiber/cpu.cc
 * Per-element multiplication of a tensor by a broadcasted fiber on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/prod_fiber/cpu.hh"
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::prod_fiber
{

template<typename T>
void cpu(Index m, Index n, Index k, Scalar alpha_, const T *src_, T *dst_)
    noexcept
//! Per-element product of a tensor and a broadcasted fiber on CPU
/*! Performs the following operations:
 *      dst[i,l,j] = alpha * dst[i,l,j] * src[l]
 *
 * @param[in] m: Size of the first mode of dst tensor
 * @param[in] n: Size of the last mode of dst tensor
 * @param[in] k: Size of the middle mode of dst tensor and the only mode of src
 *      tensor
 * @param[in] alpha_: Scalar factor
 * @param[in] src_: Input contiguous vector with k elements
 * @param[inout] dst_: Input and output contiguous m-by-k-by-n array
 * */
{
    using Y = typename CPUComputeType<T>::value;
    auto src = reinterpret_cast<const Y *>(src_);
    auto dst = reinterpret_cast<Y *>(dst_);
    const Y alpha{alpha_};
    // Cycle over input src vector
    for(Index i2 = 0; i2 < k; ++i2)
    {
        const Y src_val = alpha * src[i2];
        // Cycle over the third axis of output buffer
        for(Index i1 = 0; i1 < n; ++i1)
        {
            // Output fiber to be updated
            Y *dst_fiber = dst + (i1*k+i2)*m;
            // Cycle over the output fiber
            for(Index i0 = 0; i0 < m; ++i0)
            {
                // Update output value
                dst_fiber[i0] *= src_val;
            }
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index n, Index k, Scalar alpha, const fp32_t *src,
        fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, Index k, Scalar alpha, const fp64_t *src,
        fp64_t *dst)
    noexcept;

} // namespace nntile::kernel::prod_fiber
