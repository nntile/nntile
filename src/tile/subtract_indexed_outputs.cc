/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/subtract_indexed_outputs.cc
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-06-28
 * */

#include "nntile/tile/subtract_indexed_outputs.hh"
#include "nntile/starpu/subtract_indexed_outputs.hh"

namespace nntile
{
namespace tile
{

template<typename T>
void subtract_indexed_outputs_async(T val, const Tile<Index> &labels,
        const Tile<T> &dst)
{
// TODO - add description
    if(labels.ndim != dst.ndim-1)
    {
        throw std::runtime_error("labels.ndim != dst.ndim-1");
    }
    for(Index i = 0; i < labels.ndim; ++i)
    {
        if(labels.shape[i] != dst.shape[i+1])
        {
            throw std::runtime_error("labels.shape[i] != dst.shape[i+1]");
        }
    }
    // Insert task
    starpu::subtract_indexed_outputs::submit<T>(dst.shape[0], labels.nelems,
            val, labels, dst);
}

template<typename T>
void subtract_indexed_outputs(T val, const Tile<Index> &labels,
        const Tile<T> &dst)
{
    subtract_indexed_outputs_async<T>(val, labels, dst);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void subtract_indexed_outputs_async<fp32_t>(fp32_t val,
        const Tile<Index> &labels, const Tile<fp32_t> &dst);

template
void subtract_indexed_outputs_async<fp64_t>(fp64_t val,
        const Tile<Index> &labels, const Tile<fp64_t> &dst);

// Explicit instantiation
template
void subtract_indexed_outputs<fp32_t>(fp32_t val, const Tile<Index> &labels,
        const Tile<fp32_t> &dst);

template
void subtract_indexed_outputs<fp64_t>(fp64_t val, const Tile<Index> &labels,
        const Tile<fp64_t> &dst);

} // namespace tile
} // namespace nntile
