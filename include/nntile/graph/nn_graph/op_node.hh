/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn_graph/op_node.hh
 * NNGraph::OpNode - operation node (AutoGradFunction) for autograd.
 *
 * Include this only via nn_graph.hh (after NNGraph and TensorNode are declared).
 *
 * @version 1.1.0
 * */

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include <nntile/graph/nn_graph/nn_graph.hh>
#include <nntile/graph/nn_graph/tensor_node.hh>

namespace nntile::graph
{

//! NNGraph-level operation node (AutoGradFunction). Wraps NNBaseOpNode.
//! The op holds all params and tensors; OpNode delegates to it.
class NNGraph::OpNode
{
    friend class NNGraph;
    friend class TensorNode;

private:
    std::shared_ptr<NNBaseOpNode> op_;

public:
    explicit OpNode(std::shared_ptr<NNBaseOpNode> op) : op_(std::move(op)) {}

    const std::vector<TensorNode*>& inputs() const { return op_->inputs(); }
    const std::vector<TensorNode*>& outputs() const { return op_->outputs(); }
    const std::vector<TensorNode*>& buffers() const { return op_->buffers(); }
    //! Convenience for single-output ops. Undefined if outputs().size() != 1.
    TensorNode* output() const { return outputs().empty() ? nullptr : outputs()[0]; }
    const std::shared_ptr<NNBaseOpNode>& op() const { return op_; }
    void run_backward() const
    {
        if(op_)
        {
            op_->backward();
        }
    }
};

} // namespace nntile::graph
