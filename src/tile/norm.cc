/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/norm.cc
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

template<typename T>
void cpu_sum_ssq_single_axis_init(void *buffers[], void *cl_args)
    noexcept
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
    }
}

template<typename T>
static
void cpu_sum_ssq_single_axis_update(void *buffers[], void *cl_args)
    noexcept
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
static
void cpu_sum_ssq_single_axis_m1(void *buffers[], void *cl_args)
    noexcept
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
    static struct starpu_perfmodel model_norm_sum_ssq_init =
    {
        .type = STARPU_HISTORY_BASED,
        .symbol = "norm_sum_ssq_init",
    };
    static struct starpu_codelet codelet_sum_ssq_single_axis_init =
    {
#       if !defined(PREFER_CUDA)
        .cpu_funcs = {cpu_sum_ssq_single_axis_init<T>},
#       endif
#       if defined(NNTILE_USE_CUDA)
        .cuda_funcs = {norm_sum_ssq_codelet_cuda_single_axis_init<T>},
        .cuda_flags = {STARPU_CUDA_ASYNC},
#       endif
        .nbuffers = 2,
        .modes = {STARPU_R, STARPU_W},
        .model = &model_norm_sum_ssq_init,
        .name = "norm_sum_ssq_init",
    };
    static struct starpu_perfmodel model_norm_sum_ssq_update =
    {
        .type = STARPU_HISTORY_BASED,
        .symbol = "norm_sum_ssq_update",
    };
    static struct starpu_codelet codelet_sum_ssq_single_axis_update =
    {
#       if !defined(PREFER_CUDA)
        .cpu_funcs = {cpu_sum_ssq_single_axis_update<T>},
#       endif
#       if defined(NNTILE_USE_CUDA)
        .cuda_funcs = {norm_sum_ssq_codelet_cuda_single_axis_update<T>},
        .cuda_flags = {STARPU_CUDA_ASYNC},
#       endif
        .nbuffers = 2,
        .modes = {STARPU_R, Starpu::STARPU_RW_COMMUTE},
        .model = &model_norm_sum_ssq_update,
        .name = "norm_sum_ssq_update",
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
    int ret;
    if(init_output)
    {
        // Init output
        ret = starpu_task_insert(&codelet_sum_ssq_single_axis_init,
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
        ret = starpu_task_insert(&codelet_sum_ssq_single_axis_update,
                STARPU_VALUE, &m, sizeof(m),
                STARPU_VALUE, &n, sizeof(n),
                STARPU_VALUE, &k, sizeof(k),
                STARPU_R, static_cast<starpu_data_handle_t>(src),
                Starpu::STARPU_RW_COMMUTE,
                static_cast<starpu_data_handle_t>(sum_ssq),
                0);
    }
    if(ret != 0)
    {
        throw std::runtime_error("ret != 0");
    }
}

template
void norm_sum_ssq_work(const Tile<fp32_t> &src, const Tile<fp32_t> &sum_ssq,
        Index axis, bool init_output=false);

template
void norm_sum_ssq_work(const Tile<fp64_t> &src, const Tile<fp64_t> &sum_ssq,
        Index axis, bool init_output=false);

template<typename T>
static
void cpu_avg_dev(void *buffers[], void *cl_args)
    noexcept
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
void norm_avg_dev_work(const Tile<T> &sum_ssq, const Tile<T> &avg_dev,
        Index nelems, T eps)
{
    static struct starpu_perfmodel model_norm_avg_dev =
    {
        .type = STARPU_HISTORY_BASED,
        .symbol = "norm_avg_dev",
    };
    static struct starpu_codelet codelet_avg_dev =
    {
#       if !defined(PREFER_CUDA)
        .cpu_funcs = {cpu_avg_dev<T>},
#       endif
#       if defined(NNTILE_USE_CUDA)
        .cuda_funcs = {norm_avg_dev_codelet_cuda_single_axis<T>},
        .cuda_flags = {STARPU_CUDA_ASYNC},
#       endif
        .nbuffers = 2,
        .modes = {STARPU_R, STARPU_W},
        .model = &model_norm_avg_dev,
        .name = "norm_avg_dev",
    };
    // Get sizes
    Index m = avg_dev.nelems / 2; // 2 elements per m
    // Insert task
    int ret = starpu_task_insert(&codelet_avg_dev,
            STARPU_VALUE, &m, sizeof(m),
            STARPU_VALUE, &nelems, sizeof(nelems),
            STARPU_VALUE, &eps, sizeof(eps),
            STARPU_R, static_cast<starpu_data_handle_t>(sum_ssq),
            STARPU_W, static_cast<starpu_data_handle_t>(avg_dev),
            0);
    if(ret != 0)
    {
        throw std::runtime_error("ret != 0");
    }
}

template
void norm_avg_dev_work(const Tile<fp32_t> &sum_ssq,
        const Tile<fp32_t> &avg_dev, Index nelems, fp32_t eps);

template
void norm_avg_dev_work(const Tile<fp64_t> &sum_ssq,
        const Tile<fp64_t> &avg_dev, Index nelems, fp64_t eps);

} // namespace nntile

