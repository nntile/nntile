/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/cpu/sumnorm.cc
 * Sum and Euclidian norm of a buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-01
 * */

#include "nntile/kernel/cpu/sumnorm.hh"
#include <cmath>
#include <starpu_data_interfaces.h>

namespace nntile
{

//! Sum and Euclidian norm along middle axis
//
// For a provided m-by-k-by-n input tensor src compute sums and norms of slices
// along second axis with k elements, resulting in 2-by-m-by-n output matrix
// sumnorm. sumnorm[0, i, j] is increased by a sum of elements of a slice
// src[i, :, j], while sumnorm[1, i, j] is a square root of sumnorm[1, i, j]
// and norm of a slice src[i, :, j]. Values of tensor sumnorm are updated by
// this routine in read-write mode, therefore sumnorm must be initialized
// before use with zeros (clear).
//
// @param[in] m: Size of the first mode of src and sumnorm tensors
// @param[in] n: Size of the last mode of src and sumnorm tensors
// @param[in] k: Size of the middle mode of src tensor
// @param[in] src: Input tensor to compute sums and norms of slices
// @param[inout] sumnorm: Sums and norms of slices
//
// @sa clear_kernel_cpu
template<typename T>
void sumnorm_kernel_cpu(Index m, Index n, Index k, const T *src, T *sumnorm)
    noexcept
{
    const Index mk = m * k;
    Index dst_offset = 0;
    constexpr T zero = 0, one = 1;
    // Cycle over row of output buffer
    for(Index i2 = 0; i2 < n; ++i2)
    {
        // Cycle over column of output buffer
        for(Index i1 = 0; i1 < m; ++i1)
        {
            // Get sum and norm of a corresponding slice
            const T *src_slice = src + i2*mk + i1;
            // Init sum and norm
            // Norm is computed with help of scaled sum of squares
            T sum = sumnorm[dst_offset];
            T scale = sumnorm[dst_offset+1];
            T ssq = one;
            // Cycle over slice of input buffer
            for(Index i0 = 0; i0 < k; ++i0)
            {
                // Read value from source
                T val = src_slice[i0*m];
                // Nothing to update in case of 0
                if(val == zero)
                {
                    continue;
                }
                // Update sum, scale and scaled sum of squares
                sum += val;
                T absval = std::abs(val);
                if(absval > scale)
                {
                    T tmp = scale / absval;
                    scale = absval;
                    ssq = ssq*tmp*tmp + one;
                }
                else
                {
                    T tmp = absval / scale;
                    ssq += tmp*tmp;
                }
            }
            // Save result
            sumnorm[dst_offset] = sum;
            sumnorm[dst_offset+1] = scale * std::sqrt(ssq);
            dst_offset += 2;
        }
    }
}

//! Sum and Euclidian norm along middle axis of StarPU buffer
//
// For a provided m-by-k-by-n input tensor src compute sums and norms of slices
// along second axis with k elements, resulting in 2-by-m-by-n output matrix
// sumnorm. sumnorm[0, i, j] is increased by a sum of elements of a slice
// src[i, :, j], while sumnorm[1, i, j] is a square root of sumnorm[1, i, j]
// and norm of a slice src[i, :, j]. Values of tensor sumnorm are updated by
// this routine in read-write mode, therefore sumnorm must be initialized
// before use with zeros (clear).
//
// @param[in] buffers: input src and output sumnorm tensors through StarPU
//      handles
// @param[in] cl_args: Sizes m, n and k
//
// @sa sumnorm_kernel_cpu, clear_starpu_cpu
template<typename T>
void sumnorm_starpu_cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get sizes
    Index m, n, k;
    starpu_codelet_unpack_args(cl_args, &m, &n, &k);
    // Get pointers
    const T *src = reinterpret_cast<T *>(STARPU_NDIM_GET_PTR(buffers[0]));
    T *sumnorm = reinterpret_cast<T *>(STARPU_NDIM_GET_PTR(buffers[1]));
    sumnorm_kernel_cpu<T>(m, n, k, src, sumnorm);
}

// Explicit instantiation
template
void sumnorm_starpu_cpu<fp32_t>(void *buffers[], void *cl_args)
    noexcept;

template
void sumnorm_starpu_cpu<fp64_t>(void *buffers[], void *cl_args)
    noexcept;

} // namespace nntile

