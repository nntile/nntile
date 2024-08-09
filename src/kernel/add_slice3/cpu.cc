/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/add_slice3/cpu.cc
 * Per-element addition of a tensor and a broadcasted slice on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/add_slice3/cpu.hh"
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::add_slice3
{

template<typename T>
void cpu(Index m, Index n, Index k, Scalar alpha_, const T *src1, Scalar beta_,
        const T *src2, T *dst)
    noexcept
//! Per-element addition of a tensor and a broadcasted slice on CPU
/*! Performs the following operations:
 *      dst[i,l,j] = alpha*src1[i,j] + beta*src2[i,l,j]
 *
 * @param[in] m: Size of the first mode of src1, src2 and dst tensors
 * @param[in] n: Size of the last mode of src1, src2 and dst tensors
 * @param[in] k: Size of the middle mode of src2 and dst tensor
 * @param[in] alpha_: Scalar factor for src1
 * @param[in] src1_: Input contiguous m-by-n array
 * @param[in] beta_: Scaling factor for src1
 * @param[in] src2_: Input contiguous m-by-k-by-n array
 * @param[out] dst_: Output contiguous m-by-k-by-n array
 * */
{
    using Y = typename T::repr_t;
    const Y zero{0.0}, alpha{alpha_}, beta{beta_};
    const Index mk = m * k;
    // Cycle over column of the output buffer dst
    for(Index i2 = 0; i2 < n; ++i2)
    {
        // Cycle over row of the output buffer dst
        for(Index i1 = 0; i1 < m; ++i1)
        {
            // Pointer to a corresponding fiber of the input array src2
            const T *src2_fiber = src2 + i2*mk + i1;
            // Pointer to a corresponding fiber of the output array dst
            T *dst_fiber = dst + i2*mk + i1;
            // Value to add to the output fiber
            const Y src1_val = alpha * Y{src1[i2*m+i1]};
            // Overwrite or update output depending on beta
            if(beta == zero)
            {
                // Cycle over output fiber elements
                for(Index i0 = 0; i0 < k; ++i0)
                {
                    // Set output value
                    dst_fiber[i0*m] = static_cast<T>(src1_val);
                }
            }
            else
            {
                // Cycle over output fiber elements
                for(Index i0 = 0; i0 < k; ++i0)
                {
                    // And update it
                    dst_fiber[i0*m] = static_cast<T>(beta * Y{src2_fiber[i0*m]} + src1_val);
                }
            }
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index n, Index k, Scalar alpha, const fp32_t *src1,
        Scalar beta, const fp32_t *src2, fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, Index k, Scalar alpha, const fp64_t *src1,
        Scalar beta, const fp64_t *src2, fp64_t *dst)
    noexcept;

template
void cpu<bf16_t>(Index m, Index n, Index k, Scalar alpha, const bf16_t *src1,
        Scalar beta, const bf16_t *src2, bf16_t *dst)
    noexcept;

} // namespace nntile::kernel::add_slice3
