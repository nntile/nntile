/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/total_sum_accum.cc
 * Total sum accumulating for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-03-15
 * */

#include "nntile/tile/total_sum_accum.hh"
#include "nntile/starpu/total_sum_accum.hh"

namespace nntile
{
namespace tile
{

template<typename T>
void total_sum_accum_async(const Tile<T> &logsumexp, const Tile<T> &src,
                           const Tile<Index> &class_labels, const Tile<T> &val)
// TODO - add description
{
    if(val.ndim != 0)
    {
        throw std::runtime_error("val.ndim != 0");
    }
    if(logsumexp.ndim != 1)
    {
        throw std::runtime_error("logsumexp.ndim != 1");
    }
    if(class_labels.ndim != 1)
    {
        throw std::runtime_error("class_labels.ndim != 1");
    }
    if(src.ndim != 2)
    {
        throw std::runtime_error("src.ndim != 2");
    }
    if(logsumexp.shape[0] != class_labels.shape[0])
    {
        throw std::runtime_error("logsumexp.shape[0] != class_labels.shape[0]");
    }
    if(src.shape[0] != class_labels.shape[0])
    {
        throw std::runtime_error("src.shape[0] != class_labels.shape[0]");
    }
    // Insert task
    starpu::total_sum_accum::submit<T>(class_labels.shape[0], logsumexp, 
                                       src, class_labels, val);
}

//! Tile-wise max and sum of exponents along single given axis
template<typename T>
void total_sum_accum(const Tile<T> &logsumexp, const Tile<T> &src,
                           const Tile<Index> &class_labels, const Tile<T> &val)
{
    total_sum_accum_async<T>(logsumexp, src, class_labels, val);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void total_sum_accum_async<fp32_t>(const Tile<fp32_t> &logsumexp, const Tile<fp32_t> &src,
                           const Tile<Index> &class_labels, const Tile<fp32_t> &val);

template
void total_sum_accum_async<fp64_t>(const Tile<fp64_t> &logsumexp, const Tile<fp64_t> &src,
                           const Tile<Index> &class_labels, const Tile<fp64_t> &val);

// Explicit instantiation
template
void total_sum_accum<fp32_t>(const Tile<fp32_t> &logsumexp, const Tile<fp32_t> &src,
                           const Tile<Index> &class_labels, const Tile<fp32_t> &val);

template
void total_sum_accum<fp64_t>(const Tile<fp64_t> &logsumexp, const Tile<fp64_t> &src,
                           const Tile<Index> &class_labels, const Tile<fp64_t> &val);

} // namespace tile
} // namespace nntile