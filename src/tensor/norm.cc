/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/norm.hh
 * Functions that compute different norms.
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/tensor/norm.hh"
#include "nntile/tile/norm.hh"
#include "nntile/tile/copy.hh"

namespace nntile
{

template<typename T>
void norm_avg_dev_async(const Tensor<T> &sumnorm, const Tensor<T> &avg_dev,
        Index nelems, T eps)
{
    // Check dimensions
    if(sumnorm.ndim != avg_dev.ndim)
    {
        throw std::runtime_error("sumnorm.ndim != avg_dev.ndim");
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
    // Check shapes of inputs
    if(sumnorm.shape[0] != 3)
    {
        throw std::runtime_error("sumnorm.shape[0] != 3");
    }
    if(sumnorm.basetile_shape[0] != 3)
    {
        throw std::runtime_error("sumnorm.basetile_shape[0] != 3");
    }
    if(avg_dev.shape[0] != 2)
    {
        throw std::runtime_error("avg_dev.shape[0] != 2");
    }
    if(avg_dev.basetile_shape[0] != 2)
    {
        throw std::runtime_error("avg_dev.basetile_shape[0] != 2");
    }
    for(Index i = 1; i < sumnorm.ndim; ++i)
    {
        if(sumnorm.shape[i] != avg_dev.shape[i])
        {
            throw std::runtime_error("sumnorm.shape[i] != avg_dev.shape[i]");
        }
        if(sumnorm.basetile_shape[i] != avg_dev.basetile_shape[i])
        {
            throw std::runtime_error("sumnorm.basetile_shape[i] != "
                    "avg_dev.basetile_shape[i]");
        }
    }
    // Transform sum and sum of squares into averages and deviations
    for(Index i = 0; i < sumnorm.grid.nelems; ++i)
    {
        norm_avg_dev_async(sumnorm.get_tile(i), avg_dev.get_tile(i), nelems,
                eps);
    }
}

template
void norm_avg_dev_async(const Tensor<fp32_t> &sumnorm,
        const Tensor<fp32_t> &avg_dev, Index nelems, fp32_t eps);

template
void norm_avg_dev_async(const Tensor<fp64_t> &sumnorm,
        const Tensor<fp64_t> &avg_dev, Index nelems, fp64_t eps);

} // namespace nntile

