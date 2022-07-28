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
void norm_sum_ssq_accumulate_work(const Tensor<T> &sum_ssq,
        const Tensor<T> &sum_ssq_total)
{
}

template
void norm_sum_ssq_accumulate_work(const Tensor<fp32_t> &sum_ssq,
        const Tensor<fp32_t> &sum_ssq_total);

template
void norm_sum_ssq_accumulate_work(const Tensor<fp64_t> &sum_ssq,
        const Tensor<fp64_t> &sum_ssq_total);

template<typename T>
void norm_sum_ssq_async(const Tensor<T> &src, const Tensor<T> &sum_ssq,
        Index axis)
{
    // Check dimensions
    if(src.ndim != sum_ssq.ndim)
    {
        throw std::runtime_error("src.ndim != sum_ssq.ndim");
    }
    // Treat special case of src.ndim=0
    if(src.ndim == 0)
    {
        throw std::runtime_error("Scalar input makes no sense");
    }
    // Check axis
    if(axis < 0)
    {
        throw std::runtime_error("axis < 0");
    }
    if(axis >= src.ndim)
    {
        throw std::runtime_error("axis >= src.ndim");
    }
    // Check shapes of src and sum_ssq
    if(sum_ssq.shape[0] != 3)
    {
        throw std::runtime_error("sum_ssq.shape[0] != 3");
    }
    if(sum_ssq.basetile_shape[0] != 3)
    {
        throw std::runtime_error("sum_ssq.basetile_shape[0] != 3");
    }
    for(Index i = 0; i < axis; ++i)
    {
        if(src.shape[i] != sum_ssq.shape[i+1])
        {
            throw std::runtime_error("src.shape[i] != sum_ssq.shape[i+1]");
        }
        if(src.basetile_shape[i] != sum_ssq.basetile_shape[i+1])
        {
            throw std::runtime_error("src.basetile_shape[i] != "
                    "sum_ssq.basetile_shape[i+1]");
        }
    }
    for(Index i = axis+1; i < src.ndim; ++i)
    {
        if(src.shape[i] != sum_ssq.shape[i])
        {
            throw std::runtime_error("src.shape[i] != sum_ssq.shape[i]");
        }
        if(src.basetile_shape[i] != sum_ssq.basetile_shape[i])
        {
            throw std::runtime_error("src.basetile_shape[i] != "
                    "sum_ssq.basetile_shape[i]");
        }
    }
    // Compute sum and sum of squares for each tile in grid of src tensor
    for(Index i = 0; i < sum_ssq.grid.nelems; ++i)
    {
        auto dst_tile = sum_ssq.get_tile(i);
        auto dst_tile_index = sum_ssq.grid.linear_to_index(i);
        std::vector<Index> src_tile_index(src.ndim);
        for(Index j = 0; j < axis; ++j)
        {
            src_tile_index[j] = dst_tile_index[j+1];
        }
        src_tile_index[axis] = 0;
        for(Index j = axis+1; j < src.ndim; ++j)
        {
            src_tile_index[j] = dst_tile_index[j];
        }
        // Launch per-tile kernel
        auto src_tile = src.get_tile(src_tile_index);
        norm_sum_ssq_async(src_tile, dst_tile, axis, true);
        for(Index j = 1; j < src.grid.shape[axis]; ++j)
        {
            src_tile_index[axis] = j;
            auto src_tile = src.get_tile(src_tile_index);
            norm_sum_ssq_async(src_tile, dst_tile, axis, false);
        }
    }
}

template
void norm_sum_ssq_async(const Tensor<fp32_t> &src,
        const Tensor<fp32_t> &sum_ssq, Index axis);

template
void norm_sum_ssq_async(const Tensor<fp64_t> &src,
        const Tensor<fp64_t> &sum_ssq, Index axis);

template<typename T>
void norm_avg_dev_async(const Tensor<T> &sum_ssq, const Tensor<T> &avg_dev,
        Index nelems, T eps)
{
    // Check dimensions
    if(sum_ssq.ndim != avg_dev.ndim)
    {
        throw std::runtime_error("sum_ssq.ndim != avg_dev.ndim");
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
    if(sum_ssq.shape[0] != 3)
    {
        throw std::runtime_error("sum_ssq.shape[0] != 3");
    }
    if(sum_ssq.basetile_shape[0] != 3)
    {
        throw std::runtime_error("sum_ssq.basetile_shape[0] != 3");
    }
    if(avg_dev.shape[0] != 2)
    {
        throw std::runtime_error("avg_dev.shape[0] != 2");
    }
    if(avg_dev.basetile_shape[0] != 2)
    {
        throw std::runtime_error("avg_dev.basetile_shape[0] != 2");
    }
    for(Index i = 1; i < sum_ssq.ndim; ++i)
    {
        if(sum_ssq.shape[i] != avg_dev.shape[i])
        {
            throw std::runtime_error("sum_ssq.shape[i] != avg_dev.shape[i]");
        }
        if(sum_ssq.basetile_shape[i] != avg_dev.basetile_shape[i])
        {
            throw std::runtime_error("sum_ssq.basetile_shape[i] != "
                    "avg_dev.basetile_shape[i]");
        }
    }
    // Transform sum and sum of squares into averages and deviations
    for(Index i = 0; i < sum_ssq.grid.nelems; ++i)
    {
        norm_avg_dev_async(sum_ssq.get_tile(i), avg_dev.get_tile(i), nelems,
                eps);
    }
}

template
void norm_avg_dev_async(const Tensor<fp32_t> &sum_ssq,
        const Tensor<fp32_t> &avg_dev, Index nelems, fp32_t eps);

template
void norm_avg_dev_async(const Tensor<fp64_t> &sum_ssq,
        const Tensor<fp64_t> &avg_dev, Index nelems, fp64_t eps);

} // namespace nntile

