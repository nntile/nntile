/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/norm.hh
 * Functions that compute different norms.
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/tile/norm.hh"

namespace nntile
{

template<typename T>
static void cpu_sum_ssq(void *buffers[], void *cl_args)
{
    Index src_ndim, axes_ndim;
    // Read number of dimensions of corresponding arrays
    starpu_codelet_unpack_args(cl_args, &src_ndim, &axes_ndim, nullptr);
    Index sum_ssq_ndim = src_ndim + 1 - axes_ndim;
    // Allocate space for arrays
    std::vector<Index> src_shape(src_ndim), sum_ssq_shape(sum_ssq_ndim),
        axes(axes_ndim);
    Index src_nelems;
    // Read arrays
    starpu_codelet_unpack_args(cl_args, &src_ndim, &axes_ndim, &src_nelems,
            &(src_shape[0]), &(sum_ssq_shape[0]), &(axes[0]));
    // Get pointers
    const T *src = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
    T *sum_ssq = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[1]));
    // Define number of slices and size of each slice
    Index slice_nelems = 1;
    for(Index i = 0; i < axes_ndim; ++i)
    {
        slice_nelems *= src_shape[axes[i]];
    }
    Index nslices = src_nelems / slice_nelems;
    // Define stride arrays
    std::vector<Index> src_stride(src_ndim), sum_ssq_stride(sum_ssq_ndim);
    src_stride[0] = 1;
    for(Index i = 1; i < src_ndim; ++i)
    {
        src_stride[i] = src_stride[i-1] * src_shape[i-1];
    }
    sum_ssq_stride[0] = 1;
    for(Index i = 1; i < sum_ssq_ndim; ++i)
    {
        sum_ssq_stride[i] = sum_ssq_stride[i-1] * sum_ssq_shape[i-1];
    }
    // Non-slice axes
    std::vector<Index> sum_ssq_axes;
    sum_ssq_axes.reserve(sum_ssq_ndim-1);
    for(Index i = 0; i < src_ndim; ++i)
    {
        if(axes[i-sum_ssq_axes.size()] != i)
        {
            sum_ssq_axes.push_back(i);
        }
    }
    // Compute sum and scaled sum of squares for each slice
    std::vector<Index> sum_ssq_index(sum_ssq_ndim), src_index(src_ndim);
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
            // Get correspondign value
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
                ssq = ssq*tmp*tmp + 1;
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

template<typename T>
void norm_sum_ssq_async(const Tile<T> &src, const Tile<T> &sum_ssq,
        const std::vector<Index> &axes)
{
    static struct starpu_codelet codelet_sum_ssq =
    {
        .cpu_funcs = {cpu_sum_ssq<T>},
        .nbuffers = 2,
        .modes = {STARPU_R, STARPU_W}
    };
    // Check inputs
    if(src.ndim+1 != sum_ssq.ndim+axes.size())
    {
        throw std::runtime_error("src.ndim+1 != sum_ssq.ndim+axes.size()");
    }
    // Treat special case of src.ndim=0
    if(src.ndim == 0)
    {
        throw std::runtime_error("Scalar input makes no sense");
    }
    // Treat special case of empty axes
    if(axes.size() == 0)
    {
        throw std::runtime_error("Empty axes");
    }
    // Check axes
    if(axes[0] < 0)
    {
        throw std::runtime_error("axes[0] < 0");
    }
    if(axes[axes.size()-1] >= src.ndim)
    {
        throw std::runtime_error("axes[axes.size()-1] >= src.ndim");
    }
    for(Index i = 1; i < axes.size(); ++i)
    {
        if(axes[i] <= axes[i-1])
        {
            throw std::runtime_error("axes[i] <= axes[i-1]");
        }
    }
    // Check shapes of src and sum_ssq
    if(sum_ssq.shape[0] != 3)
    {
        throw std::runtime_error("sum_ssq.shape[0] != 3");
    }
    // Number of checked items in axes
    Index nchecked_axes = 0;
    for(Index i = 0; i < src.ndim; ++i)
    {
        if(nchecked_axes < axes.size() and i == axes[nchecked_axes])
        {
            ++nchecked_axes;
            continue;
        }
        if(src.shape[i] != sum_ssq.shape[i-nchecked_axes+1])
        {
            throw std::runtime_error("src.shape[i] != "
                    "sum_ssq.shape[i-nchecked_axes+1]");
        }
    }
    // Insert task
    Index axes_ndim = axes.size();
    starpu_task_insert(&codelet_sum_ssq,
            STARPU_VALUE, &(src.ndim), sizeof(src.ndim),
            STARPU_VALUE, &(axes_ndim), sizeof(axes_ndim),
            STARPU_VALUE, &(src.nelems), sizeof(src.nelems),
            STARPU_VALUE, &(src.shape[0]), src.ndim*sizeof(src.shape[0]),
            STARPU_VALUE, &(sum_ssq.shape[0]),
            sum_ssq.ndim*sizeof(sum_ssq.shape[0]),
            STARPU_VALUE, &(axes[0]), axes_ndim*sizeof(axes[0]),
            STARPU_R, static_cast<starpu_data_handle_t>(src),
            STARPU_W, static_cast<starpu_data_handle_t>(sum_ssq),
            0);
}

template
void norm_sum_ssq_async(const Tile<fp32_t> &src, const Tile<fp32_t> &sum_ssq,
        const std::vector<Index> &axes);

template
void norm_sum_ssq_async(const Tile<fp64_t> &src, const Tile<fp64_t> &sum_ssq,
        const std::vector<Index> &axes);

} // namespace nntile

