/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/subtract_indexed_column.cc
 * Total sum accumulating for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-03-18
 * */

#include "nntile/tile/subtract_indexed_column.hh"
#include "nntile/starpu/subtract_indexed_column.hh"

namespace nntile
{
namespace tile
{

template<typename T>
void subtract_indexed_column_async(T val, const Tile<Index> &class_labels, const Tile<T> &dst)
{
    // Insert task
    starpu::subtract_indexed_column::submit<T>(class_labels.shape[0], val, 
                                       class_labels, dst);
}

//! Tile-wise max and sum of exponents along single given axis
template<typename T>
void subtract_indexed_column(T val, const Tile<Index> &class_labels, const Tile<T> &dst)
{
    subtract_indexed_column_async<T>(val, class_labels, dst);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void subtract_indexed_column_async<fp32_t>(fp32_t val, const Tile<Index> &class_labels,
                                           const Tile<fp32_t> &dst);

template
void subtract_indexed_column_async<fp64_t>(fp64_t val, const Tile<Index> &class_labels,
                                           const Tile<fp64_t> &dst);

// Explicit instantiation
template
void subtract_indexed_column<fp32_t>(fp32_t val, const Tile<Index> &class_labels,
                                     const Tile<fp32_t> &dst);

template
void subtract_indexed_column<fp64_t>(fp64_t val, const Tile<Index> &class_labels,
                                     const Tile<fp64_t> &dst);

} // namespace tile
} // namespace nntile