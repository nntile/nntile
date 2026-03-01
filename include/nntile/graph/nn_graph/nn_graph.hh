/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn_graph/nn_graph.hh
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

#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Neural network graph - wraps logical graph and tracks gradients
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
    LogicalGraph logical_;
    std::vector<std::unique_ptr<TensorNode>> tensors_;
    std::vector<std::unique_ptr<OpNode>> op_nodes_;
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

    TensorNode* tensor(LogicalGraph::TensorNode& data,
                       bool requires_grad = false);

    // -----------------------------------------------------------------
    // Operation Builder API
    // -----------------------------------------------------------------

    void add_op(
        OpType type,
        OpAttrs attrs,
        const std::vector<TensorNode*>& inputs,
        const std::vector<TensorNode*>& outputs,
        const std::string& name = ""
    );

    // -----------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------

    const std::string& name() const { return name_; }
    size_t num_tensors() const { return tensors_.size(); }
    size_t num_ops() const { return logical_.num_ops(); }

    TensorNode* get_tensor(const std::string& name);
    const TensorNode* get_tensor(const std::string& name) const;

    std::vector<std::string> tensor_names() const;

    const std::vector<std::unique_ptr<TensorNode>>& tensors() const
    {
        return tensors_;
    }
    const std::vector<std::unique_ptr<LogicalGraph::OpNode>>& ops() const
    {
        return logical_.ops();
    }

    LogicalGraph& logical_graph() { return logical_; }
    const LogicalGraph& logical_graph() const { return logical_; }

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
    //! attrs: opaque (std::shared_ptr<void>); only forward/backward know the type.
    //! buffers: tensors saved from forward for reuse in backward (internal-only).
    OpNode* create_op(
        std::vector<TensorNode*> inputs,
        std::vector<TensorNode*> outputs,
        std::shared_ptr<void> attrs,
        std::function<void(const OpNode*)> backward_fn,
        std::vector<TensorNode*> buffers = {});

    // -----------------------------------------------------------------
    // String representation
    // -----------------------------------------------------------------

    std::string to_string() const;

    std::string to_mermaid() const { return logical_.to_mermaid(); }
};

} // namespace nntile::graph
