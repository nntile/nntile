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
#include <cmath>

namespace nntile
{

// Compute sum and scaled sum of squares of a tile
template<typename T>
static void cpu_sum_ssq_init(void *buffers[], void *cl_args)
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

// Accumulate sum and scaled sum of squares of a tile
template<typename T>
static void cpu_sum_ssq_update(void *buffers[], void *cl_args)
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
    // Compute sum and scaled sum of squares for each slice
    std::vector<Index> sum_ssq_index(sum_ssq_ndim), src_index(src_ndim);
    Index sum_ssq_linear_offset = 0, src_linear_offset = 0;
    for(Index i = 0; i < nslices; ++i)
    {
        // Set output
        T &sum = sum_ssq[sum_ssq_linear_offset];
        T &scale = sum_ssq[sum_ssq_linear_offset+1];
        T &ssq = sum_ssq[sum_ssq_linear_offset+2];
        // The first value in slice
        T val = src[src_linear_offset];
        if(val != 0)
        {
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
        // All other values in slice
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
void norm_sum_ssq_work(const Tile<T> &src, const Tile<T> &sum_ssq,
        const std::vector<Index> &axes, bool init_output)
{
    static struct starpu_codelet codelet_sum_ssq_init =
    {
        .cpu_funcs = {cpu_sum_ssq_init<T>},
        .nbuffers = 2,
        .modes = {STARPU_R, STARPU_W},
        .name = "sum_ssq_init"
    };
    static struct starpu_codelet codelet_sum_ssq_update =
    {
        .cpu_funcs = {cpu_sum_ssq_update<T>},
        .nbuffers = 2,
        .modes = {STARPU_R, Starpu::STARPU_RW_COMMUTE},
        .name = "sum_ssq_update"
    };
    // Insert task
    Index axes_ndim = axes.size();
    if(init_output)
    {
        starpu_task_insert(&codelet_sum_ssq_init,
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
    else
    {
        starpu_task_insert(&codelet_sum_ssq_update,
                STARPU_VALUE, &(src.ndim), sizeof(src.ndim),
                STARPU_VALUE, &(axes_ndim), sizeof(axes_ndim),
                STARPU_VALUE, &(src.nelems), sizeof(src.nelems),
                STARPU_VALUE, &(src.shape[0]), src.ndim*sizeof(src.shape[0]),
                STARPU_VALUE, &(sum_ssq.shape[0]),
                sum_ssq.ndim*sizeof(sum_ssq.shape[0]),
                STARPU_VALUE, &(axes[0]), axes_ndim*sizeof(axes[0]),
                STARPU_R, static_cast<starpu_data_handle_t>(src),
                Starpu::STARPU_RW_COMMUTE,
                static_cast<starpu_data_handle_t>(sum_ssq),
                0);
    }
}

template
void norm_sum_ssq_work(const Tile<fp32_t> &src, const Tile<fp32_t> &sum_ssq,
        const std::vector<Index> &axes, bool init_output=true);

template
void norm_sum_ssq_work(const Tile<fp64_t> &src, const Tile<fp64_t> &sum_ssq,
        const std::vector<Index> &axes, bool init_output=true);

template<typename T>
static void cpu_sum_ssq_single_axis_init(void *buffers[], void *cl_args)
{
    // Get sizes
    Index m, n, k;
    starpu_codelet_unpack_args(cl_args, &m, &n, &k);
    const Index mk = m * k;
    // Get pointers
    const T *src = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
    T *sum_ssq = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[1]));
    Index dst_offset = 0;
    for(Index i2 = 0; i2 < n; ++i2)
    {
        for(Index i1 = 0; i1 < m; ++i1)
        {
            const T *src_slice = src + i2*mk + i1;
            T sum = 0, scale = 0, ssq = 0;
            for(Index i0 = 0; i0 < k; ++i0)
            {
                // Read value from source
                T val = src_slice[i0*m];
                // Update scale and scaled sum of squares
                if(val == 0)
                {
                    continue;
                }
                sum += val;
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
            sum_ssq[dst_offset] = sum;
            sum_ssq[dst_offset+1] = scale;
            sum_ssq[dst_offset+2] = ssq;
            dst_offset += 3;
        }

//        const T *src_ptr = src + i2*mk;
//        Index dst_offset = 3 * i2 * m;
//        for(Index i1 = 0; i1 < m; ++i1)
//        {
//            // Read value from source
//            T val = src_ptr[i1];
//            // Update scale and scaled sum of squares
//            if(val == 0)
//            {
//                sum_ssq[dst_offset] = 0;
//                sum_ssq[dst_offset+1] = 0;
//                sum_ssq[dst_offset+2] = 0;
//                dst_offset += 3;
//                continue;
//            }
//            sum_ssq[dst_offset] = val;
//            sum_ssq[dst_offset+1] = val;
//            sum_ssq[dst_offset+2] = 1;
//            dst_offset += 3;
//        }
//        for(Index i0 = 1; i0 < k; ++i0)
//        {
//            const T *src_ptr = src + i2*mk + i0*m;
//            Index dst_offset = 3 * i2 * m;
//            for(Index i1 = 0; i1 < m; ++i1)
//            {
//                // Read value from source
//                T val = src_ptr[i1];
//                // Update scale and scaled sum of squares
//                if(val == 0)
//                {
//                    continue;
//                }
//                sum_ssq[dst_offset] += val;
//                T absval = std::abs(val);
//                T &scale = sum_ssq[dst_offset+1];
//                T &ssq = sum_ssq[dst_offset+2];
//                if(absval > scale)
//                {
//                    T tmp = scale / absval;
//                    scale = absval;
//                    ssq = ssq*tmp*tmp + T{1};
//                }
//                else
//                {
//                    T tmp = absval / scale;
//                    ssq += tmp*tmp;
//                }
//                dst_offset += 3;
//            }
//        }
    }
}

template<typename T>
static void cpu_sum_ssq_single_axis_update(void *buffers[], void *cl_args)
{
    // Get sizes
    Index m, n, k;
    starpu_codelet_unpack_args(cl_args, &m, &n, &k);
    const Index mk = m * k;
    // Get pointers
    const T *src = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
    T *sum_ssq = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[1]));
    Index dst_offset = 0;
    for(Index i2 = 0; i2 < n; ++i2)
    {
        for(Index i1 = 0; i1 < m; ++i1)
        {
            const T *src_slice = src + i2*mk + i1;
            T &sum = sum_ssq[dst_offset];
            T &scale = sum_ssq[dst_offset+1];
            T &ssq = sum_ssq[dst_offset+2];
            for(Index i0 = 0; i0 < k; ++i0)
            {
                // Read value from source
                T val = src_slice[i0*m];
                // Update scale and scaled sum of squares
                if(val == 0)
                {
                    continue;
                }
                sum += val;
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
            dst_offset += 3;
        }

//        for(Index i0 = 0; i0 < k; ++i0)
//        {
//            const T *src_ptr = src + i2*mk + i0*m;
//            Index dst_offset = 3 * i2 * m;
//            for(Index i1 = 0; i1 < m; ++i1)
//            {
//                // Read value from source
//                T val = src_ptr[i1];
//                // Update scale and scaled sum of squares
//                if(val == 0)
//                {
//                    continue;
//                }
//                sum_ssq[dst_offset] += val;
//                T absval = std::abs(val);
//                T &scale = sum_ssq[dst_offset+1];
//                T &ssq = sum_ssq[dst_offset+2];
//                if(absval > scale)
//                {
//                    T tmp = scale / absval;
//                    scale = absval;
//                    ssq = ssq*tmp*tmp + T{1};
//                }
//                else
//                {
//                    T tmp = absval / scale;
//                    ssq += tmp*tmp;
//                }
//                dst_offset += 3;
//            }
//        }
    }
}

template<typename T>
static void cpu_sum_ssq_single_axis_m1(void *buffers[], void *cl_args)
{
    // Get sizes
    Index n, k;
    starpu_codelet_unpack_args(cl_args, &n, &k);
    // Get pointers
    const T *src = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
    T *sum_ssq = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[1]));
    Index dst_offset = 0;
    for(Index i2 = 0; i2 < n; ++i2)
    {
        const T *src_slice = src + i2*k;
        T sum = 0, scale = 0, ssq = 0;
        for(Index i0 = 0; i0 < k; ++i0)
        {
            // Read value from source
            T val = src_slice[i0];
            // Update scale and scaled sum of squares
            if(val == 0)
            {
                continue;
            }
            sum += val;
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
        sum_ssq[dst_offset] = sum;
        sum_ssq[dst_offset+1] = scale;
        sum_ssq[dst_offset+2] = ssq;
        dst_offset += 3;
    }
}

template<typename T>
void norm_sum_ssq_work(const Tile<T> &src, const Tile<T> &sum_ssq,
        Index axis, bool init_output)
{
    static struct starpu_codelet codelet_sum_ssq_single_axis_init =
    {
        .cpu_funcs = {cpu_sum_ssq_single_axis_init<T>},
        .nbuffers = 2,
        .modes = {STARPU_R, STARPU_W}
    };
    static struct starpu_codelet codelet_sum_ssq_single_axis_update =
    {
        .cpu_funcs = {cpu_sum_ssq_single_axis_update<T>},
        .nbuffers = 2,
        .modes = {STARPU_R, Starpu::STARPU_RW_COMMUTE}
    };
    // Get sizes
    Index m, n, k;
    if(axis == 0)
    {
        m = 1;
        n = sum_ssq.nelems / 3;
        k = src.shape[0];
    }
    else if(axis == src.ndim-1)
    {
        m = sum_ssq.nelems / 3;
        n = 1;
        k = src.shape[axis];
    }
    else
    {
        m = src.stride[axis];
        n = src.matrix_shape[axis+1][1];
        k = src.shape[axis];
    }
    // Insert task
    if(init_output)
    {
        // Init output
        starpu_task_insert(&codelet_sum_ssq_single_axis_init,
                STARPU_VALUE, &m, sizeof(m),
                STARPU_VALUE, &n, sizeof(n),
                STARPU_VALUE, &k, sizeof(k),
                STARPU_R, static_cast<starpu_data_handle_t>(src),
                STARPU_W, static_cast<starpu_data_handle_t>(sum_ssq),
                0);
    }
    else
    {
        // Update output
        starpu_task_insert(&codelet_sum_ssq_single_axis_update,
                STARPU_VALUE, &m, sizeof(m),
                STARPU_VALUE, &n, sizeof(n),
                STARPU_VALUE, &k, sizeof(k),
                STARPU_R, static_cast<starpu_data_handle_t>(src),
                Starpu::STARPU_RW_COMMUTE,
                static_cast<starpu_data_handle_t>(sum_ssq),
                0);
    }
}

template
void norm_sum_ssq_work(const Tile<fp32_t> &src, const Tile<fp32_t> &sum_ssq,
        Index axis, bool init_output=false);

template
void norm_sum_ssq_work(const Tile<fp64_t> &src, const Tile<fp64_t> &sum_ssq,
        Index axis, bool init_output=false);

template<typename T>
static void cpu_avg_dev(void *buffers[], void *cl_args)
{
    // Get sizes
    Index m, nelems;
    T eps;
    starpu_codelet_unpack_args(cl_args, &m, &nelems, &eps);
    const T inv_nelems = T{1} / static_cast<T>(nelems);
    // Get pointers
    const T *sum_ssq = reinterpret_cast<T *>(
            STARPU_VARIABLE_GET_PTR(buffers[0]));
    T *avg_dev = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[1]));
    for(Index i = 0; i < m; ++i)
    {
        const T avg = sum_ssq[3*i] * inv_nelems;
        T scale = sum_ssq[3*i+1];
        T scaled_avg_sqr = sum_ssq[3*i+2] * inv_nelems;
        avg_dev[2*i] = avg;
        // Mean of square values minus square of mean values
        // |avg| <= scale since |1/n sum x_i| <= max|x_i|
        T tmp = avg / scale;
        scaled_avg_sqr -= tmp * tmp;
        // Update by eps
        if(eps > 0)
        {
            if(scale >= eps)
            {
                T tmp = eps / scale;
                scaled_avg_sqr += tmp*tmp;
            }
            else
            {
                T tmp = scale / eps;
                scale = eps;
                scaled_avg_sqr *= tmp*tmp;
                scaled_avg_sqr += T{1};
            }
        }
        // Set deviation
        avg_dev[2*i+1] = scale * std::sqrt(scaled_avg_sqr);
    }
}

template<typename T>
void norm_avg_dev_async(const Tile<T> &sum_ssq, const Tile<T> &avg_dev,
        Index nelems, T eps)
{
    static struct starpu_codelet codelet_avg_dev =
    {
        .cpu_funcs = {cpu_avg_dev<T>},
        .nbuffers = 2,
        .modes = {STARPU_R, STARPU_W}
    };
    // Check inputs
    if(sum_ssq.ndim != avg_dev.ndim)
    {
        throw std::runtime_error("sum_ssq.ndim != avg_dev.ndim");
    }
    // Input shape dimension shall be at least 1
    if(sum_ssq.ndim == 0)
    {
        throw std::runtime_error("sum_ssq.ndim == 0");
    }
    // Check number of elements
    if(nelems <= 0)
    {
        throw std::runtime_error("nelems <= 0");
    }
    // Check regularization
    if(eps < 0)
    {
        throw std::runtime_error("eps < 0");
    }
    // Check shapes
    if(sum_ssq.shape[0] != 3)
    {
        throw std::runtime_error("sum_ssq.shape[0] != 3");
    }
    if(avg_dev.shape[0] != 2)
    {
        throw std::runtime_error("avg_dev.shape[0] != 2");
    }
    for(Index i = 1; i < sum_ssq.ndim; ++i)
    {
        if(sum_ssq.shape[i] != avg_dev.shape[i])
        {
            throw std::runtime_error("sum_ssq.shape[i] != avg_dev.shape[i]");
        }
    }
    // Get sizes
    Index m = avg_dev.nelems / 2; // 2 elements per m
    // Insert task
    starpu_task_insert(&codelet_avg_dev,
            STARPU_VALUE, &m, sizeof(m),
            STARPU_VALUE, &nelems, sizeof(nelems),
            STARPU_VALUE, &eps, sizeof(eps),
            STARPU_R, static_cast<starpu_data_handle_t>(sum_ssq),
            STARPU_W, static_cast<starpu_data_handle_t>(avg_dev),
            0);
}

template
void norm_avg_dev_async(const Tile<fp32_t> &sum_ssq,
        const Tile<fp32_t> &avg_dev, Index nelems, fp32_t eps);

template
void norm_avg_dev_async(const Tile<fp64_t> &sum_ssq,
        const Tile<fp64_t> &avg_dev, Index nelems, fp64_t eps);

} // namespace nntile

