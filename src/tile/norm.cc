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

