/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/axis_descriptor.hh
 * AxisDescriptor - shared descriptor for a dimension group in TensorGraph.
 *
 * @version 1.1.0
 * */

#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nntile/base_types.hh>

namespace nntile::graph
{

class TensorGraph;

//! Shared descriptor for a group of tensor dimensions that must be
//! tiled identically. Tensors sharing the same AxisDescriptor (via
//! shared_ptr) belong to the same dimension group.
struct AxisDescriptor
{
    Index extent;
    std::string name;

    //! (tensor_node, axis_index) pairs for all members of this group.
    //! Updated during merge_axis().
    std::vector<std::pair<void*, int>> members;

    //! Tile sizes along this axis. Empty means not tiled (single tile
    //! covering the full extent).
    std::vector<Index> tile_sizes;

    //! Set uniform tiling: all tiles have the given size (last tile
    //! may be smaller if extent is not divisible).
    void set_tiling(Index tile_size);

    //! Set heterogeneous tiling: explicit tile size per tile.
    //! Sum of tile_sizes must equal extent.
    void set_tiling(const std::vector<Index>& sizes);

    //! True if tiling has been set.
    bool is_tiled() const { return !tile_sizes.empty(); }

    //! Number of tiles along this axis (1 if not tiled).
    Index num_tiles() const;

    //! Human-readable tile sizes for display: "N" for single/uniform,
    //! "{a,b,c}" for heterogeneous. Returns empty string if not tiled.
    std::string tile_sizes_to_string() const;
};

//! Merge two axis groups. All tensors holding `replace` are redirected
//! to hold `keep`. Throws if extents differ. No-op if already same.
void merge_axis(std::shared_ptr<AxisDescriptor>& keep,
                std::shared_ptr<AxisDescriptor>& replace);

} // namespace nntile::graph
