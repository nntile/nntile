/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/graph_data_node.hh
 * TileGraph::TileNode - data node for TileGraph (shape, dtype, name).
 *
 * @version 1.1.0
 * */

#pragma once

#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>

namespace nntile::graph
{

class TileGraph;

//! Data node for TileGraph - represents a single tile.
class TileGraph::TileNode
{
public:
    using NodeId = uint64_t;

    TileNode(
        NodeId id,
        TileGraph* graph,
        std::vector<Index> shape,
        DataType dtype,
        const std::string& name = "");

    NodeId id() const { return id_; }
    const std::string& name() const { return name_; }
    DataType dtype() const { return dtype_; }
    const std::vector<Index>& shape() const { return shape_; }
    Index ndim() const { return static_cast<Index>(shape_.size()); }
    Index dim(int idx) const;
    Index nelems() const;
    size_t size_bytes() const;

    TileGraph* graph();
    const TileGraph* graph() const;

    bool is_input() const { return is_input_; }
    bool is_output() const { return is_output_; }
    void mark_input(bool v = true) { is_input_ = v; }
    void mark_output(bool v = true) { is_output_ = v; }

    //! Parent tensor descriptor (nullptr if not part of a tensor tiling)
    const TileGraph::TensorDescriptor* tensor_descriptor() const
    {
        return tensor_desc_;
    }

    //! Tile coordinate within the tensor's grid (empty if no descriptor)
    const std::vector<Index>& tile_coord() const { return tile_coord_; }

    //! Set tensor descriptor and tile coordinate
    void set_tensor_info(TileGraph::TensorDescriptor* desc,
                         std::vector<Index> coord);

    void set_bind_hint(std::vector<std::uint8_t> data);
    const std::vector<std::uint8_t>* get_bind_hint() const;

    std::string to_string() const;

private:
    NodeId id_;
    TileGraph* graph_;
    std::vector<Index> shape_;
    DataType dtype_;
    std::string name_;
    bool is_input_ = false;
    bool is_output_ = false;
    std::optional<std::vector<std::uint8_t>> bind_hint_;
    TileGraph::TensorDescriptor* tensor_desc_ = nullptr;
    std::vector<Index> tile_coord_;

    friend class TileGraph;
};

} // namespace nntile::graph
