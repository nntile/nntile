/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/nn_graph.hh
 * NNGraph class for defining computation graphs with gradients.
 *
 * @version 1.1.0
 * */

#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <nntile/graph/tensor_graph.hh>

namespace nntile::graph
{

//! Neural network graph - wraps tensor graph and tracks gradients
class NNGraph
{
public:
    //! Tensor node - full definition in tensor_node.hh
    class TensorNode;

    //! Op node (AutoGradFunction) - full definition in op_node.hh
    class OpNode;

    //! Destructor defined in .cc (needs complete TensorNode for unique_ptr)
    ~NNGraph();

private:
    std::string name_;
    TensorGraph tensor_graph_;
    std::vector<std::unique_ptr<TensorNode>> tensors_;
    std::vector<std::shared_ptr<OpNode>> op_nodes_;
    std::map<std::string, TensorNode*> tensor_by_name_;

public:
    explicit NNGraph(const std::string& name = "");

    // -----------------------------------------------------------------
    // Tensor Creation
    // -----------------------------------------------------------------

    TensorNode* tensor(
        std::vector<Index> shape,
        const std::string& name,
        DataType dtype = DataType::FP32,
        bool requires_grad = true
    );

    TensorNode* tensor(TensorGraph::DataNode* data,
                       bool requires_grad = false);

    // -----------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------

    const std::string& name() const { return name_; }
    size_t num_tensors() const { return tensors_.size(); }
    size_t num_ops() const { return tensor_graph_.num_ops(); }

    TensorNode* get_tensor(const std::string& name);
    const TensorNode* get_tensor(const std::string& name) const;

    std::vector<std::string> tensor_names() const;

    const std::vector<std::unique_ptr<TensorNode>>& tensors() const
    {
        return tensors_;
    }
    const std::vector<std::shared_ptr<TensorGraph::OpNode>>& ops() const
    {
        return tensor_graph_.ops();
    }

    TensorGraph& tensor_graph() { return tensor_graph_; }
    const TensorGraph& tensor_graph() const { return tensor_graph_; }

    //! Clear op_nodes_ (used when retain_graph=false after backward)
    void clear_op_nodes();

    //! Set producer_=nullptr on all tensors that had producers
    void clear_producers_on_tensors();

    // -----------------------------------------------------------------
    // Gradient helpers
    // -----------------------------------------------------------------

    bool requires_grad(const TensorNode* tensor) const;
    void set_requires_grad(TensorNode* tensor, bool requires = true);

    bool is_first_grad(const TensorNode* tensor) const;

    TensorNode* get_or_create_grad(
        TensorNode* tensor,
        const std::string& grad_name);

    //! Create and register an NNGraph-level op (AutoGradFunction).
    OpNode* create_op(std::shared_ptr<OpNode> op);

    // -----------------------------------------------------------------
    // String representation
    // -----------------------------------------------------------------

    std::string to_string() const;

    std::string to_mermaid() const { return tensor_graph_.to_mermaid(); }
};

// -----------------------------------------------------------------
// Operation registration (part of graph API)
// -----------------------------------------------------------------

//! Register op for backward. Creates OpNode only when GradMode enabled
//! and any input requires grad. Sets producer on outputs.
void register_op(NNGraph& graph, std::shared_ptr<NNGraph::OpNode> op);

//! True if any input requires grad.
bool any_input_requires_grad(
    const std::vector<NNGraph::TensorNode*>& inputs);

} // namespace nntile::graph
