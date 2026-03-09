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

#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>

namespace nntile::graph
{

class TensorGraph;

//! Tile graph - defines computation at tile level.
//! Constructed from a TensorGraph by splitting tensors into tiles.
//! For now each tensor maps to a single tile.
class TileGraph
{
public:
    class TileNode;
    class Runtime;
    class OpNode;
    using NodeId = uint64_t;

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

    //! Build a TileGraph from a TensorGraph (1 tile per tensor)
    static TileGraph from_tensor_graph(const TensorGraph& tg);

    const std::string& name() const { return name_; }
    size_t num_data() const { return data_.size(); }
    size_t num_ops() const { return ops_.size(); }

    TileNode* get_tile_node(const std::string& name);
    const TileNode* get_tile_node(const std::string& name) const;

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
    std::string name_;
    std::vector<std::unique_ptr<TileNode>> data_;
    std::vector<std::shared_ptr<TileGraph::OpNode>> ops_;
    std::map<std::string, TileNode*> data_by_name_;

    NodeId next_data_id_ = 0;
    NodeId next_op_id_ = 0;
};

} // namespace nntile::graph
