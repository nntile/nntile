/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/tests/tile/tile_graph_shape_helpers.hh
 * Helpers for TileGraph C-order vs core Fortran-order parity tests.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/tensor/shape_layout.hh>

#include <vector>

namespace nntile::test::tile_graph_shapes
{

inline std::vector<Index> storage_shape(std::vector<Index> graph_shape)
{
    return nntile::tensor::graph_shape_to_storage(std::move(graph_shape));
}

inline std::vector<Index> graph_shape(std::vector<Index> storage_shape)
{
    return nntile::tensor::storage_shape_to_graph(std::move(storage_shape));
}

inline Index storage_axis(Index graph_axis, Index ndim)
{
    return nntile::tensor::graph_axis_to_storage(graph_axis, ndim);
}

inline Index graph_axis(Index storage_axis, Index ndim)
{
    return nntile::tensor::storage_axis_to_graph(storage_axis, ndim);
}

} // namespace nntile::test::tile_graph_shapes
