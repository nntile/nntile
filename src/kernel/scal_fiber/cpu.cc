/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/scal_fiber/cpu.cc
 * Per-element addition of a tensor and a broadcasted fiber on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/scal_fiber/cpu.hh"
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::scal_fiber
{

template<typename T>
//Убрать beta  и src2  и оставить условия где beta = 0
void cpu(Index m, Index n, Index k, Index batch, Scalar alpha_, const T *src,
         T *dst)
    noexcept
//! Per-element addition of a tensor and a broadcasted fiber on CPU
/*! Performs the following operations:
 *      dst[i,l,j,b] = alpha*src[l,b]
 *
 * @param[in] m: Size of the first mode of dst tensor
 * @param[in] n: Size of the last mode of dst tensor
 * @param[in] k: Size of the middle mode of dst tensor and the only mode of src
 *      tensors
 * @param[in] batch: Size of the batch dimension
 * @param[in] alpha_: Scalar factor for src
 * @param[in] src: Input contiguous vector with k elements
 * @param[inout] dst: Input and output contiguous m-by-k-by-n array
 * */
{
    using Y = typename T::repr_t;
    const Y zero{0.0}, alpha{alpha_};
    // Cycle over batch
    for(Index b = 0; b < batch; ++b)
    {
        // Cycle over input fiber src
        for(Index i2 = 0; i2 < k; ++i2)
        {
            // Value to add to the output slice
            const Y src_val = alpha * Y{src[i2+b*k]};
            // Cycle over the third axis of output buffer dst
            for(Index i1 = 0; i1 < n; ++i1)
            {
                // Output fiber to be updated
                T *dst_fiber = dst + ((i1+b*n)*k+i2)*m;
                // Cycle over output fiber elements
                for(Index i0 = 0; i0 < m; ++i0)
                {
                    // Set output value
                    dst_fiber[i0] = static_cast<T>(src_val);
                }
            }
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index n, Index k, Index batch, Scalar alpha,
        const fp32_t *src, fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, Index k, Index batch, Scalar alpha,
        const fp64_t *src, fp64_t *dst)
    noexcept;

template
void cpu<bf16_t>(Index m, Index n, Index k, Index batch, Scalar alpha,
        const bf16_t *src, bf16_t *dst)
    noexcept;

} // namespace nntile::kernel::scal_fiber
