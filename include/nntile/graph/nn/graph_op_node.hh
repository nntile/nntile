/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/graph_op_node.hh
 * NNGraph::OpNode - operation node (AutoGradFunction) for autograd.
 *
 * Include via nn.hh or nn/graph.hh (after NNGraph and TensorNode are declared).
 *
 * @version 1.1.0
 * */

#pragma once

#include <memory>
#include <vector>

#include <nntile/graph/nn/graph.hh>
#include <nntile/graph/nn/graph_tensor_node.hh>

namespace nntile::graph
{

//! NNGraph-level operation node (AutoGradFunction).
//! Base class for autograd ops; holds params, inputs, outputs, buffers.
//! Implements forward() and backward().
class NNGraph::OpNode
{
    friend class NNGraph;
    friend class TensorNode;

public:
    virtual ~OpNode() = default;

    const std::vector<TensorNode*>& inputs() const { return inputs_; }
    const std::vector<TensorNode*>& outputs() const { return outputs_; }
    const std::vector<TensorNode*>& buffers() const { return buffers_; }

    //! Convenience for single-output ops. Undefined if outputs().size() != 1.
    TensorNode* output() const
    {
        return outputs().empty() ? nullptr : outputs()[0];
    }

    //! Add forward pass ops to TensorGraph. Creates outputs, adds ops, returns
    //! primary output. Called before register_op. (PyTorch-style: outputs
    //! appear in forward.)
    virtual TensorNode* forward(const std::string& output_name) = 0;

    //! Run backward pass. Uses own inputs/outputs/params. Adds ops to TensorGraph.
    virtual void backward() const = 0;

protected:
    OpNode() = default;

    std::vector<TensorNode*> inputs_;
    std::vector<TensorNode*> outputs_;
    std::vector<TensorNode*> buffers_;
};

} // namespace nntile::graph
