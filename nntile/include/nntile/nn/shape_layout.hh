/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/nn/shape_layout.hh
 * Graph API shape labels vs tile storage shape labels (reversed axes).
 *
 * NNGraph exposes user-facing ``shape()`` with outermost dimension first.
 * ``tensor::*`` and tile storage use the reversed label order. Conversion is
 * metadata only (no payload reordering). External BLAS (storage layout) uses the
 * storage label convention at the kernel boundary.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>

#include <vector>

namespace nntile::nn
{

//! Reverse dimension labels (graph vs storage).
inline std::vector<Index> reverse_shape(const std::vector<Index> &shape)
{
    return std::vector<Index>(shape.rbegin(), shape.rend());
}

//! Graph axis (0 = outermost) -> storage axis (0 = innermost).
inline Index graph_axis_to_storage(Index graph_axis, Index ndim)
{
    return ndim - 1 - graph_axis;
}

//! Storage axis -> graph axis.
inline Index storage_axis_to_graph(Index storage_axis, Index ndim)
{
    return ndim - 1 - storage_axis;
}

//! Graph API shape -> tile storage shape for ``tensor::*``.
inline std::vector<Index> graph_shape_to_storage(
    const std::vector<Index> &graph_shape)
{
    return reverse_shape(graph_shape);
}

//! Tile storage shape -> graph API shape for ``NNGraph::TensorNode``.
inline std::vector<Index> storage_shape_to_graph(
    const std::vector<Index> &storage_shape)
{
    return reverse_shape(storage_shape);
}

} // namespace nntile::nn
