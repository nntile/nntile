/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/norm.cu
 * Functions that compute different norms.
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/tile/norm.hh"
#include <cmath>

namespace nntile
{

// Compute sum and scaled sum of squares of a tile
template<typename T>
static
void cuda_sum_ssq_init(void *buffers[], void *cl_args)
    noexcept
{
    const Index *src_ndim_ptr, *axes_ndim_ptr, *src_nelems_ptr, *src_shape,
          *sum_ssq_shape, *axes;
    Starpu::unpack_args_ptr(cl_args, src_ndim_ptr, axes_ndim_ptr,
            src_nelems_ptr, src_shape, sum_ssq_shape, axes);
    Index src_ndim = *src_ndim_ptr, axes_ndim = *axes_ndim_ptr,
          src_nelems = *src_nelems_ptr;
    Index sum_ssq_ndim = src_ndim + 1 - axes_ndim;
    // Get pointers
    const T *src = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
    T *sum_ssq = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[1]));
    Index *src_stride = reinterpret_cast<Index *>(
            STARPU_VARIABLE_GET_PTR(buffers[2]));
    Index *src_index = src_stride + src_ndim;
    Index *sum_ssq_stride = src_index + src_ndim;
    Index *sum_ssq_index = sum_ssq_stride + sum_ssq_ndim;
    // Define number of slices and size of each slice
    Index slice_nelems = 1;
    for(Index i = 0; i < axes_ndim; ++i)
    {
        slice_nelems *= src_shape[axes[i]];
    }
    Index nslices = src_nelems / slice_nelems;
    // Define stride arrays
    src_stride[0] = 1;
    src_index[0] = 0;
    for(Index i = 1; i < src_ndim; ++i)
    {
        src_stride[i] = src_stride[i-1] * src_shape[i-1];
        src_index[i] = 0;
    }
    sum_ssq_stride[0] = 1;
    sum_ssq_index[0] = 0;
    for(Index i = 1; i < sum_ssq_ndim; ++i)
    {
        sum_ssq_stride[i] = sum_ssq_stride[i-1] * sum_ssq_shape[i-1];
        sum_ssq_index[i] = 0;
    }
    // Compute sum and scaled sum of squares for each slice
    Index sum_ssq_linear_offset = 0, src_linear_offset = 0;
    for(Index i = 0; i < nslices; ++i)
    {
        T val = src[src_linear_offset];
        T sum = val, scale = std::abs(val), ssq = 1;
        for(Index j = 1; j < slice_nelems; ++j)
        {
            // Update src_index
            Index nchecked_axes = 0, k = axes[nchecked_axes];
            ++src_index[k];
            while(src_index[k] == src_shape[k])
            {
                src_index[k] = 0;
                ++nchecked_axes;
                k = axes[nchecked_axes];
                ++src_index[k];
            }
            // Update linear offset for src
            src_linear_offset = src_index[0];
            for(Index m = 1; m < src_ndim; ++m)
            {
                src_linear_offset += src_index[m] * src_stride[m];
            }
            // Get corresponding value
            T val = src[src_linear_offset];
            // No need to update anything if new value is zero
            // This way we avoid absval=scale=0 situation with division by zero
            if(val == 0)
            {
                continue;
            }
            // Update sum
            sum += val;
            // Update scale and scaled sum of scares
            T absval = std::abs(val);
            if(absval > scale)
            {
                T tmp = scale / absval;
                scale = absval;
                ssq = ssq*tmp*tmp + T{1};
            }
            else
            {
                T tmp = absval / scale;
                ssq += tmp*tmp;
            }
        }
        // Set output
        sum_ssq[sum_ssq_linear_offset] = sum;
        sum_ssq[sum_ssq_linear_offset+1] = scale;
        sum_ssq[sum_ssq_linear_offset+2] = ssq;
        // Update linear offset for sum_ssq
        if(i == nslices-1)
        {
            break;
        }
        ++sum_ssq_index[1];
        Index j = 0, k = 0;
        while(k < axes_ndim and axes[k] == j)
        {
            ++k;
            ++j;
        }
        while(sum_ssq_index[j-k+1] == src_shape[j])
        {
            sum_ssq_index[j-k+1] = 0;
            ++j;
            while(k < axes_ndim and axes[k] == j)
            {
                ++j;
                ++k;
            }
            ++sum_ssq_index[j-k+1];
        }
        sum_ssq_linear_offset = 0;
        for(Index k = 1; k < sum_ssq_ndim; ++k)
        {
            sum_ssq_linear_offset += sum_ssq_index[k] * sum_ssq_stride[k];
        }
        // Update src_index
        for(Index k = 0; k < src_ndim; ++k)
        {
            src_index[k] = 0;
        }
        Index nchecked_axes = 0;
        for(Index k = 0; k < src_ndim; ++k)
        {
            if(nchecked_axes < axes_ndim and k == axes[nchecked_axes])
            {
                ++nchecked_axes;
                continue;
            }
            src_index[k] = sum_ssq_index[k-nchecked_axes+1];
        }
        // Update linear offset for src
        src_linear_offset = src_index[src_ndim-1];
        for(Index k = src_ndim-2; k >= 0; --k)
        {
            src_linear_offset *= src_shape[k];
            src_linear_offset += src_index[k];
        }
    }
}
} // namespace nntile

