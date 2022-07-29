/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/norm.hh
 * Functions that compute different norms.
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{

#ifdef NNTILE_USE_CUDA
template<typename T>
void norm_avg_dev_codelet_cuda_single_axis(void *buffers[], void *cl_args);

extern template
void norm_avg_dev_codelet_cuda_single_axis<fp32_t>(void *buffers[],
        void *cl_args);

extern template
void norm_avg_dev_codelet_cuda_single_axis<fp64_t>(void *buffers[],
        void *cl_args);
#endif // NNTILE_USE_CUDA

//! Tile-wise average and deviation from sum and scaled sum of squares
//
// Main computational routine that does NO argument checking.
template<typename T>
void norm_avg_dev_work(const Tile<T> &sum_ssq, const Tile<T> &avg_dev,
        Index nelems, T eps);

//! Tile-wise average and deviation from sum and scaled sum of squares
//
// Checks input arguments
template<typename T>
void norm_avg_dev_async(const Tile<T> &sum_ssq, const Tile<T> &avg_dev,
        Index nelems, T eps)
{
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
    // Launch codelet
    norm_avg_dev_work<T>(sum_ssq, avg_dev, nelems, eps);
}

//! Tile-wise average and deviation from sum and scaled sum of squares
//
// Checks input arguments and blocks until finished
template<typename T>
void norm_avg_dev(const Tile<T> &sum_ssq, const Tile<T> &avg_dev,
        Index nelems, T eps)
{
    norm_avg_dev_async(sum_ssq, avg_dev, nelems, eps);
    starpu_task_wait_for_all();
}

} // namespace nntile

