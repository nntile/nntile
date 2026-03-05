/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/cross_entropy.hh
 * NNGraph cross_entropy autograd operation.
 *
 * Forward: loss = scale * sum over batch of (logsumexp(x, axis) - x[label])
 * Backward: grad_x = scale * (softmax(x) - one_hot(labels))
 *
 * Based on wrappers/python/nntile/loss/crossentropy.py
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/nn/graph_op_node.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{

//! Cross-entropy op: loss = scale * sum(logsumexp(x) - x[label]).
//! Input x: logits [nclasses, batch_size] (axis 0 = class dimension).
//! Input labels: int64 [batch_size].
//! Output: scalar loss.
struct NNCrossEntropyOp : NNGraph::OpNode
{
    Index axis;
    int redux;
    Scalar scale;
    Index ignore_index;
    NNGraph::TensorNode* x = nullptr;
    NNGraph::TensorNode* labels = nullptr;
    TensorGraph::TensorNode* maxsumexp_data_ = nullptr;

    NNCrossEntropyOp() = default;
    NNCrossEntropyOp(NNGraph::TensorNode* x_,
                    NNGraph::TensorNode* labels_,
                    Index axis_ = 0,
                    int redux_ = 0,
                    Scalar scale_ = 1.0,
                    Index ignore_index_ = -100)
        : axis(axis_), redux(redux_), scale(scale_), ignore_index(ignore_index_)
        , x(x_), labels(labels_)
    {
        inputs_ = {x, labels};
    }

    NNGraph::TensorNode* forward(const std::string& output_name);
    void backward() const override;
};

NNGraph::TensorNode* cross_entropy(
    NNGraph::TensorNode* x,
    NNGraph::TensorNode* labels,
    const std::string& output_name,
    Index axis = 0,
    int redux = 0,
    Scalar scale = 1.0,
    Index ignore_index = -100);

} // namespace nntile::graph
