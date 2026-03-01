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
#include <vector>

#include <nntile/graph/logical_graph.hh>
#include <nntile/graph/nn_graph/nn_graph.hh>
#include <nntile/graph/nn_graph/tensor_node.hh>

namespace nntile::graph
{

//! NNGraph-level operation node (AutoGradFunction). Represents one NNGraph op
//! that may use multiple LogicalGraph ops in forward/backward.
class NNGraph::OpNode
{
    friend class NNGraph;
    friend class TensorNode;

public:
    using BackwardFn = std::function<void(const OpNode* op)>;

private:
    std::vector<TensorNode*> inputs_;
    std::vector<TensorNode*> outputs_;
    OpAttrs attrs_;
    BackwardFn backward_fn_;
    std::vector<TensorNode*> buffers_;

public:
    OpNode(std::vector<TensorNode*> inputs,
           std::vector<TensorNode*> outputs,
           OpAttrs attrs,
           BackwardFn backward_fn,
           std::vector<TensorNode*> buffers = {})
        : inputs_(std::move(inputs))
        , outputs_(std::move(outputs))
        , attrs_(std::move(attrs))
        , backward_fn_(std::move(backward_fn))
        , buffers_(std::move(buffers))
    {
    }

    const std::vector<TensorNode*>& inputs() const { return inputs_; }
    const std::vector<TensorNode*>& outputs() const { return outputs_; }
    const std::vector<TensorNode*>& buffers() const { return buffers_; }
    //! Convenience for single-output ops. Undefined if outputs().size() != 1.
    TensorNode* output() const { return outputs_.empty() ? nullptr : outputs_[0]; }
    const OpAttrs& attrs() const { return attrs_; }
    void run_backward() const
    {
        if(backward_fn_)
        {
            backward_fn_(this);
        }
    }
};

} // namespace nntile::graph
