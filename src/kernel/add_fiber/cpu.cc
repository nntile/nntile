/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/add_fiber/cpu.cc
 * Per-element addition of a tensor and a broadcasted fiber on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/add_fiber/cpu.hh"
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::add_fiber
{

template<typename T>
void cpu(Index m, Index n, Index k, Index batch, Scalar alpha_, const T *src,
        Scalar beta_, T *dst)
    noexcept
//! Per-element addition of a tensor and a broadcasted fiber on CPU
/*! Performs the following operations:
 *      dst[i,l,j,b] = beta*dst[i,l,j,b] + alpha*src[l,b]
 *
 * @param[in] m: Size of the first mode of dst tensor
 * @param[in] n: Size of the last mode of dst tensor
 * @param[in] k: Size of the middle mode of dst tensor and the only mode of src
 *      tensors
 * @param[in] batch: Size of the batch dimension
 * @param[in] alpha_: Scalar factor for src
 * @param[in] src: Input contiguous vector with k elements
 * @param[in] beta_: Scaling factor for dst
 * @param[inout] dst: Input and output contiguous m-by-k-by-n array
 * */
{
    using Y = typename T::repr_t;
    const Y zero{0.0}, alpha{alpha_}, beta{beta_};
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
                // Overwrite or update output depending on beta
                if(beta == zero)
                {
                    // Cycle over output fiber elements
                    for(Index i0 = 0; i0 < m; ++i0)
                    {
                        // Set output value
                        dst_fiber[i0] = static_cast<T>(src_val);
                    }
                }
                else
                {
                    // Cycle over output fiber elements
                    for(Index i0 = 0; i0 < m; ++i0)
                    {
                        // Read value from the output
                        T &dst_val = dst_fiber[i0];
                        // And update it
                        dst_val = static_cast<T>(beta * Y{dst_val} + src_val);
                    }
                }
            }
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index n, Index k, Index batch, Scalar alpha,
        const fp32_t *src, Scalar beta, fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, Index k, Index batch, Scalar alpha,
        const fp64_t *src, Scalar beta, fp64_t *dst)
    noexcept;

template
void cpu<bf16_t>(Index m, Index n, Index k, Index batch, Scalar alpha,
        const bf16_t *src, Scalar beta, bf16_t *dst)
    noexcept;

} // namespace nntile::kernel::add_fiber
