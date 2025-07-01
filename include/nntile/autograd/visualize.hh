/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/autograd/visualize.hh
 * Graph visualization for autograd
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/autograd/tensor.hh>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace nntile::autograd
{

//! Graph visualization helper
class GraphVisualizer
{
private:
    //! Node information
    struct NodeInfo {
        std::string name;
        bool is_input;
        bool requires_grad;
        std::string op_name;
        std::vector<std::string> inputs;
    };

    //! Graph nodes
    std::unordered_map<const void*, NodeInfo> nodes_;
    //! Visited nodes for DFS
    std::unordered_set<const void*> visited_;
    //! Node counter for unique names
    size_t node_counter_;

    //! Get unique node name
    std::string get_node_name(const AutogradTensor& tensor);

    //! Add node to graph
    void add_node(const AutogradTensor& tensor, const std::string& name);

    //! Build graph recursively
    void build_graph(const AutogradTensor& tensor);

    //! Generate node representation
    std::string node_to_string(const NodeInfo& node) const;

public:
    //! Constructor
    GraphVisualizer(): node_counter_(0) {}

    //! Visualize computation graph
    std::string visualize(const AutogradTensor& tensor);
};

} // namespace nntile::autograd
