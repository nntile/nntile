/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/tensor_graph_tiling.hh
 * Tile layout derived from TensorGraph axis descriptors (arbitrary partitions).
 *
 * @version 1.1.0
 * */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <sstream>

#include <nntile/base_types.hh>
#include <nntile/dtype.hh>
#include <nntile/tensor/graph_decl.hh>
#include <nntile/tensor/graph_data_node.hh>

namespace nntile
{

//! Per-tensor layout: product grid over axis segments from AxisDescriptor.
class TensorAxisLayout
{
  public:
    explicit TensorAxisLayout(const TensorGraph::TensorNode *node);

    const std::vector<Index> &tensor_shape() const
    {
        return shape_;
    }
    const std::vector<Index> &grid_shape() const
    {
        return grid_shape_;
    }

    //! Row-major linear index of a grid coordinate (dim 0 slowest).
    Index grid_linear(const std::vector<Index> &grid_coord) const;

    void grid_coord_from_linear(
        Index linear, std::vector<Index> &grid_coord) const;

    //! Extent of the tile at grid_coord along each dimension.
    std::vector<Index> tile_shape_at(
        const std::vector<Index> &grid_coord) const;

    Index tile_nelems_at(const std::vector<Index> &grid_coord) const;

    Index grid_volume() const
    {
        return grid_volume_;
    }

    //! Global tensor coordinate = tile origin + local.
    void global_coord(const std::vector<Index> &grid_coord,
        const std::vector<Index> &local_within_tile,
        std::vector<Index> &global_out) const;

    //! Max segment length per axis (TensorDescriptor::tile_shape summary).
    std::vector<Index> max_tile_extents() const;

    //! Global inclusive index range of the tile at grid_coord along axis dim.
    void tile_axis_global_range(const std::vector<Index> &grid_coord,
        Index dim,
        Index &global_lo,
        Index &global_hi_inclusive) const;

    //! Segment index on axis dim that contains global_index.
    Index tile_index_containing(Index dim, Index global_index) const;

    //! Stable string for debug / export (cached). Prefer hash in hot path.
    std::string const &layout_fingerprint() const;

    //! O(1) reuse check for incremental lowering.
    std::uint64_t layout_fingerprint_hash() const;

  private:
    std::vector<Index> shape_;
    std::vector<std::vector<Index>> segments_;
    std::vector<std::vector<Index>> axis_origin_;
    std::vector<Index> grid_shape_;
    Index grid_volume_ = 1;
    mutable std::string fingerprint_;
    mutable std::uint64_t fingerprint_hash_ = 0;
    mutable bool fingerprint_hash_ready_ = false;
};

//! Maps each tensor data node to its axis layout (dense by NodeId).
class TensorGraphTiling
{
  public:
    static TensorGraphTiling from_tensor_graph(const TensorGraph &tg);

    static TensorGraphTiling from_phase(
        const TensorGraph &tg, const TensorGraph::PhaseSnapshot &phase);

    void ensure_phase_layouts(
        const TensorGraph &tg, const TensorGraph::PhaseSnapshot &phase);

    void clear()
    {
        layouts_by_id_.clear();
    }

    void erase(const TensorGraph::TensorNode *node)
    {
        if (node == nullptr)
        {
            return;
        }
        auto const id = static_cast<size_t>(node->id());
        if (id < layouts_by_id_.size())
        {
            layouts_by_id_[id].reset();
        }
    }

    const TensorAxisLayout *find(
        const TensorGraph::TensorNode *node) const;

    bool contains(const TensorGraph::TensorNode *node) const
    {
        return find(node) != nullptr;
    }

  private:
    void set_layout(
        const TensorGraph::TensorNode *node, TensorAxisLayout layout);

    //! Dense by ``TensorNode::id()``; empty optional = hole.
    std::vector<std::optional<TensorAxisLayout>> layouts_by_id_;
};

} // namespace nntile
