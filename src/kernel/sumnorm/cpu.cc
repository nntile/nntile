/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/sumnorm/cpu.cc
 * Sum and Euclidean norm of a buffer on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/sumnorm/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::sumnorm
{

template<typename T>
void cpu(Index m, Index n, Index k, const T *src_, T *sumnorm_)
    noexcept
//! Sum and Euclidean norm along middle axis
/*! For a provided m-by-k-by-n input array src compute sums and norms of slices
 * along second axis with k elements, resulting in 2-by-m-by-n output array
 * sumnorm. Input value sumnorm[0, i, j] is increased by a sum of elements of a
 * slice src[i, :, j] on output, while output value of sumnorm[1, i, j] is a
 * square root of sum of squares of input sumnorm[1, i, j] and norm of a slice
 * src[i, :, j]. Values of array sumnorm are updated by this routine in
 * read-write mode, therefore sumnorm must be initialized before use with zeros
 * (e.g., by clear() function).
 *
 * Mnemonically, the following operations are performed:
 *      sumnorm[0,i,j] = sumnorm[0,i,j] + sum(src[i,:,j])
 *      sumnorm[1,i,j] = sqrt(sumnorm[1,i,j] + norm(src[i,:,j])^2)
 *
 * @param[in] m: Size of the first mode of src and the second mode of sumnorm
 *      arrays.
 * @param[in] n: Size of the last mode of src and sumnorm arrays
 * @param[in] k: Size of the middle mode of src array
 * @param[in] src_: Input contiguous m-by-k-by-n array
 * @param[inout] sumnorm_: Output contiguous 2-by-m-by-n array, that accumulates
 *      sums and norms of slices along middle axis.
 * */
{
    using Y = typename CPUComputeType<T>::value;
    auto src = reinterpret_cast<const Y *>(src_);
    auto sumnorm = reinterpret_cast<Y *>(sumnorm_);
    constexpr Y zero{0.0}, one{1.0};
    const Index mk = m * k;
    Index dst_offset = 0;
    // Cycle over row of output buffer
    for(Index i2 = 0; i2 < n; ++i2)
    {
        // Cycle over column of output buffer
        for(Index i1 = 0; i1 < m; ++i1)
        {
            // Get sum and norm of a corresponding slice
            const Y *src_slice = src + i2*mk + i1;
            // Init sum and norm
            // Norm is computed with help of scaled sum of squares
            Y sum = sumnorm[dst_offset];
            Y scale = sumnorm[dst_offset+1];
            Y ssq = one;
            // Cycle over slice of input buffer
            for(Index i0 = 0; i0 < k; ++i0)
            {
                // Read value from source
                Y val = src_slice[i0*m];
                // Nothing to update in case of 0
                if(val == zero)
                {
                    continue;
                }
                // Update sum, scale and scaled sum of squares
                sum += val;
                Y absval = std::fabs(val);
                if(absval > scale)
                {
                    Y tmp = scale / absval;
                    scale = absval;
                    ssq = ssq*tmp*tmp + one;
                }
                else
                {
                    Y tmp = absval / scale;
                    ssq += tmp*tmp;
                }
            }
            // Save result. Due to roundings an average value may become larger
            // than a root-mean-square value, which is impossible for precise
            // numbers
            sumnorm[dst_offset] = sum;
            sumnorm[dst_offset+1] = scale * std::sqrt(ssq);
            dst_offset += 2;
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, Index n, Index k, const fp32_t *src,
        fp32_t *sumnorm)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, Index k, const fp64_t *src,
        fp64_t *sumnorm)
    noexcept;

} // namespace nntile::kernel::sumnorm
