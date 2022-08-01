/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/normalize.hh
 * Normalize operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{

extern starpu_perfmodel normalize_perfmodel_fp32, normalize_perfmodel_fp64;

extern StarpuCodelet normalize_codelet_fp32, normalize_codelet_fp64;

template<typename T>
constexpr StarpuCodelet *normalize_codelet()
{
    throw std::runtime_error("Non-supported type");
    return nullptr;
}

template<>
constexpr StarpuCodelet *normalize_codelet<fp32_t>()
{
    return &normalize_codelet_fp32;
}

template<>
constexpr StarpuCodelet *normalize_codelet<fp64_t>()
{
    return &normalize_codelet_fp64;
}

// Normalization operation over single axis
template<typename T>
void normalize_work(const StarpuVariableHandle &gamma_beta,
        const Tile<T> &sumnorm, const Tile<T> &dst, Index l, T eps,
        Index axis);

template<typename T>
void normalize_async(const StarpuVariableHandle &gamma_beta,
        const Tile<T> &sumnorm, const Tile<T> &dst, Index l, T eps,
        Index axis)
{
    // Check inputs
    if(sumnorm.ndim != dst.ndim)
    {
        throw std::runtime_error("sumnorm.ndim != dst.ndim");
    }
    // Input shape dimension shall be at least 1
    if(sumnorm.ndim == 0)
    {
        throw std::runtime_error("sumnorm.ndim == 0");
    }
    // Check number of elements
    if(l <= 0)
    {
        throw std::runtime_error("l <= 0");
    }
    // Check regularization
    if(eps < 0)
    {
        throw std::runtime_error("eps < 0");
    }
    // Check shapes
    if(sumnorm.shape[0] != 2)
    {
        throw std::runtime_error("sumnorm.shape[0] != 2");
    }
    for(Index i = 0; i < axis; ++i)
    {
        if(dst.shape[i] != sumnorm.shape[i+1])
        {
            throw std::runtime_error("dst.shape[i] != sumnorm.shape[i+1]");
        }
    }
    for(Index i = axis+1; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != sumnorm.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != sumnorm.shape[i]");
        }
    }
    // Launch codelet
    normalize_work<T>(gamma_beta, sumnorm, dst, l, eps, axis);
}

//! Tile-wise average and deviation from sum and scaled sum of squares
//
// Checks input arguments and blocks until finished
template<typename T>
void normalize(const StarpuVariableHandle &gamma_beta,
        const Tile<T> &sumnorm, const Tile<T> &dst, Index l, T eps,
        Index axis)
{
    normalize_async(gamma_beta, sumnorm, dst, l, eps, axis);
    starpu_task_wait_for_all();
}

} // namespace nntile

