/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/graph_decl.hh
 * TensorGraph class declaration (included by graph.hh).
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

namespace nntile::graph
{

//! Tensor graph - defines computation at tensor level
class TensorGraph
{
public:
    class TensorNode;
    class Runtime;
    class OpNode;
    using NodeId = uint64_t;

    explicit TensorGraph(const std::string& name = "")
        : name_(name)
    {
    }

    //! Create an input data node (not produced by any operation)
    TensorNode* data(
        std::vector<Index> shape,
        const std::string& name,
        DataType dtype = DataType::FP32);

    //! Add an operation to the graph
    void add_op(std::shared_ptr<TensorGraph::OpNode> op_node,
                const std::string& name = "");

    // Queries
    const std::string& name() const { return name_; }
    size_t num_data() const { return data_.size(); }
    size_t num_ops() const { return ops_.size(); }

    TensorNode* get_tensor_node(const std::string& name);
    const TensorNode* get_tensor_node(const std::string& name) const;

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

    const std::vector<std::unique_ptr<TensorNode>>& tensor_nodes() const
    {
        return data_;
    }

    const std::vector<std::shared_ptr<TensorGraph::OpNode>>& ops() const
    {
        return ops_;
    }

    std::string to_string() const;
    std::string to_mermaid() const;

private:
    std::string name_;
    std::vector<std::unique_ptr<TensorNode>> data_;
    std::vector<std::shared_ptr<TensorGraph::OpNode>> ops_;
    std::map<std::string, TensorNode*> data_by_name_;

    NodeId next_data_id_ = 0;
    NodeId next_op_id_ = 0;
};

} // namespace nntile::graph
