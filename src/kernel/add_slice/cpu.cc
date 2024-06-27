/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/add_slice/cpu.cc
 * Per-element addition of a tensor and a broadcasted slice on CPU
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/add_slice/cpu.hh"
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::add_slice
{

template<typename T>
void cpu(Index m, Index n, Index k, T alpha_, const T *src_, T beta_,
        T *dst_)
    noexcept
//! Per-element addition of a tensor and a broadcasted slice on CPU
/*! Performs the following operations:
 *      dst[i,l,j] = beta*dst[i,l,j] + alpha*src[i,j]
 *
 * @param[in] m: Size of the first mode of src and dst tensors
 * @param[in] n: Size of the last mode of src and dst tensors
 * @param[in] k: Size of the middle mode of dst tensor
 * @param[in] alpha_: Scalar factor for src
 * @param[in] src_: Input contiguous m-by-n array
 * @param[in] beta_: Scaling factor for dst
 * @param[inout] dst_: Input and output contiguous m-by-k-by-n array
 * */
{
    using Y = typename CPUComputeType<T>::value;
    auto src = reinterpret_cast<const Y *>(src_);
    auto dst = reinterpret_cast<Y *>(dst_);
    const Y zero{0.0}, alpha{alpha_}, beta{beta_};
    const Index mk = m * k;
    // Cycle over column of the output buffer dst
    for(Index i2 = 0; i2 < n; ++i2)
    {
        // Cycle over row of the output buffer dst
        for(Index i1 = 0; i1 < m; ++i1)
        {
            // Pointer to a corresponding fiber of the output array dst
            Y *dst_fiber = dst + i2*mk + i1;
            // Value to add to the output fiber
            const Y src_val = alpha * src[i2*m+i1];
            // Overwrite or update output depending on beta
            if(beta == zero)
            {
                // Cycle over output fiber elements
                for(Index i0 = 0; i0 < k; ++i0)
                {
                    // Set output value
                    dst_fiber[i0*m] = src_val;
                }
            }
            else
            {
                // Cycle over output fiber elements
                for(Index i0 = 0; i0 < k; ++i0)
                {
                    // Read value from the output
                    Y &dst_val = dst_fiber[i0*m];
                    // And update it
                    dst_val = beta*dst_val + src_val;
                }
            }
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index n, Index k, fp32_t alpha, const fp32_t *src,
        fp32_t beta, fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, Index k, fp64_t alpha, const fp64_t *src,
        fp64_t beta, fp64_t *dst)
    noexcept;

} // namespace nntile::kernel::add_slice
