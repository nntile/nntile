/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/graph_decl.hh
 * TileGraph class declaration (included by graph.hh).
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tensor/graph_decl.hh>
#include <nntile/graph/tensor/graph_data_node.hh>
#include <nntile/graph/tensor/tensor_graph_tiling.hh>

namespace nntile::graph
{

//! Tile graph - defines computation at tiles; layout comes from TensorGraph
//! axis descriptors (see TensorGraphTiling).
class TileGraph
{
public:
    class TileNode;
    class Runtime;
    class OpNode;
    using NodeId = uint64_t;

    //! Describes how one tensor was split into tiles.
    //! Holds a back-pointer to the source TensorNode so that
    //! tile data can be obtained from the tensor at compile time.
    struct TensorDescriptor
    {
        std::string tensor_name;
        std::vector<Index> tensor_shape;
        std::vector<Index> tile_shape;
        std::vector<Index> grid_shape;
        DataType dtype;
        std::vector<TileNode*> tiles;
        const TensorGraph::TensorNode* source_node = nullptr;
    };

    explicit TileGraph(const std::string& name = "")
        : name_(name)
    {
    }

    //! Create a data node (tile)
    TileNode* data(
        std::vector<Index> shape,
        const std::string& name,
        DataType dtype = DataType::FP32);

    //! Add an operation to the graph
    void add_op(std::shared_ptr<TileGraph::OpNode> op_node,
                const std::string& name = "");

    //! Register a TensorDescriptor (returns non-owning pointer)
    TensorDescriptor* add_tensor_descriptor(TensorDescriptor desc);

    //! Build a TileGraph using axis tiling from the tensor graph.
    static TileGraph from_tensor_graph(const TensorGraph& tg);

    //! Build a TileGraph with an explicit tiling scheme (must match tg axes).
    static TileGraph from_tensor_graph(
        const TensorGraph& tg, const TensorGraphTiling& tiling);

    //! Tiling used for logical bind/get_output (set by from_tensor_graph).
    const TensorGraphTiling* tiling_scheme() const
    {
        return tiling_scheme_.get();
    }

    void set_tiling_scheme(std::shared_ptr<const TensorGraphTiling> scheme)
    {
        tiling_scheme_ = std::move(scheme);
    }

    const std::string& name() const { return name_; }
    size_t num_data() const { return data_.size(); }
    size_t num_ops() const { return ops_.size(); }
    size_t num_tensors() const { return tensors_.size(); }

    TileNode* get_tile_node(const std::string& name);
    const TileNode* get_tile_node(const std::string& name) const;

    TensorDescriptor* get_tensor_descriptor(const std::string& tensor_name);
    const TensorDescriptor* get_tensor_descriptor(
        const std::string& tensor_name) const;

    const std::vector<std::unique_ptr<TensorDescriptor>>& tensor_descriptors()
        const
    {
        return tensors_;
    }

    std::vector<std::string> data_names() const
    {
        std::vector<std::string> names;
        names.reserve(data_by_name_.size());
        for(const auto& pair : data_by_name_)
        {
            names.push_back(pair.first);
        }
        return names;
    }

    const std::vector<std::unique_ptr<TileNode>>& tile_nodes() const
    {
        return data_;
    }

    const std::vector<std::shared_ptr<TileGraph::OpNode>>& ops() const
    {
        return ops_;
    }

    std::string to_string() const;
    std::string to_mermaid() const;

private:
    std::shared_ptr<const TensorGraphTiling> tiling_scheme_;
    std::string name_;
    std::vector<std::unique_ptr<TileNode>> data_;
    std::vector<std::shared_ptr<TileGraph::OpNode>> ops_;
    std::vector<std::unique_ptr<TensorDescriptor>> tensors_;
    std::map<std::string, TileNode*> data_by_name_;
    std::map<std::string, TensorDescriptor*> tensors_by_name_;

    NodeId next_data_id_ = 0;
    NodeId next_op_id_ = 0;
};

} // namespace nntile::graph
