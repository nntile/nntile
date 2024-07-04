/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/norm_fiber/cpu.cc
 * Euclidean norms over slices into a fiber of a product of buffers on CPU
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/norm_fiber/cpu.hh"
#include <cmath>

namespace nntile::kernel::norm_fiber
{

template<typename T>
void cpu(Index m, Index n, Index k, Index batch, T alpha, const T *src, T beta,
        T *dst)
    noexcept
//! Norms over slices along the first and last axes into a fiber of a tensor
/*! For a provided m-by-k-by-n input array computes norms over slices
 * along the first axis with m elements and the last axis with n elements,
 * resulting in output fiber of shape (k).
 * Mnemonically, the following operations are performed:
 *      dst[l,b] = hypot(beta*dst[l,b], alpha*norm(src[:,l,:,b]))
 *
 * @param[in] m: Size of the first mode of src array
 * @param[in] n: Size of the last mode of src array
 * @param[in] k: Size of the middle mode of src array and the only mode of
 *      dst array
 * @param[in] batch: Size of the batch dimension
 * @param[in] alpha: Scaling factor for src
 * @param[in] src: Input contiguous m-by-k-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Output contiguous vector with k elements, that accumulate
 *      norm over slices along the first and the last axes.
 * */
{
    constexpr T zero = 0;
    alpha = std::fabs(alpha); // norm is always nonnegative
    // Cycle over batch
    for(Index b = 0; b < batch; ++b)
    {
        // Cycle over the only axis of output buffer
        for(Index i2 = 0; i2 < k; ++i2)
        {
            // Init norm of the slice
            T norm_max = zero, norm_ssq = zero, c = zero, y, t;
            // Output value
            T &result = dst[i2+b*k];
            // Cycle over the third axis of input buffer
            for(Index i1 = 0; i1 < n; ++i1)
            {
                // Get sum of a corresponding slice
                const T *src_slice = src + ((i1+b*n)*k+i2)*m;
                // Cycle over the first axis of input buffer
                for(Index i0 = 0; i0 < m; ++i0)
                {
                    // Read value from source
                    T val = std::fabs(src_slice[i0]);
                    // Update norm only if new value is non-zero
                    if(val > 0)
                    {
                        if(norm_max >= val)
                        {
                            T tmp1 = val / norm_max;
                            y = tmp1*tmp1 - c;
                            t = norm_ssq + y;
                            c = (t-norm_ssq) - y;
                            norm_ssq = t;
                        }
                        else
                        {
                            T tmp1 = norm_max / val;
                            T tmp2 = tmp1 * tmp1;
                            y = one - c*tmp2;
                            norm_ssq *= tmp2;
                            t = norm_ssq + y;
                            c = (t-norm_ssq) - y;
                            norm_ssq = t;
                            norm_max = val;
                        }
                    }
                }
                // Get the scaled norm
                norm_max *= alpha;
                //T norm = norm_max * std::sqrt(norm_ssq);
                // Update output value
                if(beta == zero)
                {
                    //result = norm;
                    result = norm_max * std::sqrt(norm_ssq);
                }
                else if(norm_max > 0)
                {
                    //result = std::hypot(beta*result, norm);
                    T tmp_res = std::fabs(beta * result);
                    if(norm_max >= tmp_res)
                    {
                        T tmp1 = tmp_res / norm_max;
                        result = norm_max * std::sqrt((tmp1*tmp1-c)+norm_ssq);
                    }
                    else
                    {
                        T tmp1 = norm_max / tmp_res;
                        T tmp2 = tmp1 * tmp1;
                        c *= tmp2;
                        norm_ssq *= tmp2;
                        result = tmp_res * std::sqrt((one-c)+norm_ssq);
                    }
                }
                // norm_max==0
                else
                {
                    result = std::fabs(beta * result);
                }
            }
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index n, Index k, Index batch, fp32_t alpha,
        const fp32_t *src, fp32_t beta, fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, Index k, Index batch, fp64_t alpha,
        const fp64_t *src, fp64_t beta, fp64_t *dst)
    noexcept;

} // namespace nntile::kernel::norm_fiber

