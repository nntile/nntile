/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/tensor_graph_tiling.hh
 * Tile layout derived from TensorGraph axis descriptors (arbitrary partitions).
 *
 * @version 1.1.0
 * */

#pragma once

#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tensor/graph_decl.hh>
#include <nntile/graph/tensor/graph_data_node.hh>

namespace nntile::graph
{

//! Per-tensor layout: product grid over axis segments from AxisDescriptor::tile_sizes.
class TensorAxisLayout
{
public:
    explicit TensorAxisLayout(const TensorGraph::TensorNode* node);

    const std::vector<Index>& tensor_shape() const { return shape_; }
    const std::vector<Index>& grid_shape() const { return grid_shape_; }

    //! Row-major linear index of a grid coordinate (dim 0 slowest).
    Index grid_linear(const std::vector<Index>& grid_coord) const;

    void grid_coord_from_linear(Index linear, std::vector<Index>& grid_coord) const;

    //! Extent of the tile at grid_coord along each dimension.
    std::vector<Index> tile_shape_at(const std::vector<Index>& grid_coord) const;

    Index tile_nelems_at(const std::vector<Index>& grid_coord) const;

    Index grid_volume() const { return grid_volume_; }

    //! Global tensor coordinate = tile origin + local; tile origin from grid_coord.
    void global_coord(const std::vector<Index>& grid_coord,
                      const std::vector<Index>& local_within_tile,
                      std::vector<Index>& global_out) const;

    //! Max segment length per axis (for TensorDescriptor::tile_shape summary).
    std::vector<Index> max_tile_extents() const;

    //! Global inclusive index range of the tile at grid_coord along axis dim.
    void tile_axis_global_range(const std::vector<Index>& grid_coord, Index dim,
        Index& global_lo, Index& global_hi_inclusive) const;

    //! Segment index on axis dim that contains global_index in [0, shape(dim)).
    Index tile_index_containing(Index dim, Index global_index) const;

private:
    std::vector<Index> shape_;
    //! segments_[d][k] is k-th tile length along axis d.
    std::vector<std::vector<Index>> segments_;
    //! axis_origin_[d][k] = starting index along d for tile k.
    std::vector<std::vector<Index>> axis_origin_;
    std::vector<Index> grid_shape_;
    Index grid_volume_ = 1;
};

//! Maps each tensor data node to its axis layout (from merged AxisDescriptors).
class TensorGraphTiling
{
public:
    static TensorGraphTiling from_tensor_graph(const TensorGraph& tg);

    const TensorAxisLayout* find(const TensorGraph::TensorNode* node) const;

    bool contains(const TensorGraph::TensorNode* node) const
    {
        return layouts_.count(node) != 0;
    }

    const std::map<const TensorGraph::TensorNode*, TensorAxisLayout>& layouts()
        const
    {
        return layouts_;
    }

private:
    std::map<const TensorGraph::TensorNode*, TensorAxisLayout> layouts_;
};

} // namespace nntile::graph
