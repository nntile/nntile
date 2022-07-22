/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/bias2.hh
 * Bias operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{

template<typename T>
void bias2_kernel_cpu(Index, Index, Index, const T *, T *)
    noexcept;

template<typename T>
void bias2_starpu_cpu(void *[], void *)
    noexcept;

#ifdef NNTILE_USE_CUDA
template<typename T>
void bias2_kernel_cuda(Index, Index, Index, Index, const T *, T *)
    noexcept;

template<typename T>
void bias2_starpu_cuda(void *[], void *)
    noexcept;
#endif // NNTILE_USE_CUDA

extern starpu_perfmodel bias2_perfmodel_fp32, bias2_perfmodel_fp64;
extern StarpuCodelet bias2_codelet_fp32, bias2_codelet_fp64;
void bias2_restrict_where(uint32_t where);
void bias2_restore_where();

template<typename T>
constexpr StarpuCodelet *bias2_codelet()
{
    throw std::runtime_error("Non-supported type");
    return nullptr;
}

template<>
constexpr StarpuCodelet *bias2_codelet<fp32_t>()
{
    return &bias2_codelet_fp32;
}

template<>
constexpr StarpuCodelet *bias2_codelet<fp64_t>()
{
    return &bias2_codelet_fp64;
}

//! Tile-wise bias operation by averages and deviations
//
// Main computational routine that does NO argument checking.
// @param[in] avg_dev: Source of the bias (averages and deviations)
// @param[inout] dst: Destination of the bias
// @param[in] axis: Dimension index of the bias
template<typename T>
void bias2_work(const Tile<T> &avg_dev, const Tile<T> &dst,
        Index axis);

//! Tile-wise bias operation by averages and deviations
//
// Checks input arguments
template<typename T>
void bias2_async(const Tile<T> &avg_dev, const Tile<T> &dst,
        Index axis)
{
    // Check dimensions
    if(dst.ndim != avg_dev.ndim)
    {
        throw std::runtime_error("dst.ndim != avg_dev.ndim");
    }
    // Check axis
    if(axis < 0)
    {
        throw std::runtime_error("axis < 0");
    }
    if(axis >= dst.ndim)
    {
        throw std::runtime_error("axis >= dst.ndim");
    }
    // Check shapes
    if(avg_dev.shape[0] != 2)
    {
        throw std::runtime_error("avg_dev.shape[0] != 2");
    }
    for(Index i = 0; i < axis; ++i)
    {
        if(dst.shape[i] != avg_dev.shape[i+1])
        {
            throw std::runtime_error("dst.shape[i] != avg_dev.shape[i+1]");
        }
    }
    for(Index i = axis+1; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != avg_dev.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i]");
        }
    }
    // Launch codelet
    bias2_work<T>(avg_dev, dst, axis);
}

//! Tile-wise bias operation by averages and deviations
//
// Checks input arguments and blocks until finished
template<typename T>
void bias2(const Tile<T> &avg_dev, const Tile<T> &dst, Index axis)
{
    bias2_async<T>(avg_dev, dst, axis);
    starpu_task_wait_for_all();
}

} // namespace nntile

