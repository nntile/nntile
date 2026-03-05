/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn/cross_entropy.cc
 * NNGraph cross_entropy autograd implementation.
 *
 * Based on wrappers/python/nntile/loss/crossentropy.py
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn/cross_entropy.hh"
#include "nntile/graph/nn/graph_data_node.hh"

#include <stdexcept>

#include "nntile/graph/tensor/add_inplace.hh"
#include "nntile/graph/tensor/clear.hh"
#include "nntile/graph/tensor/logsumexp.hh"
#include "nntile/graph/tensor/maxsumexp.hh"
#include "nntile/graph/tensor/softmax.hh"
#include "nntile/graph/tensor/subtract_indexed_outputs.hh"
#include "nntile/graph/tensor/total_sum_accum.hh"
#include "nntile/graph/tensor/transpose.hh"

namespace nntile::graph
{

namespace
{
constexpr Scalar grad_overwrite = 0.0;
constexpr Scalar grad_accumulate = 1.0;
} // anonymous namespace

NNGraph::TensorNode* NNCrossEntropyOp::forward(const std::string& output_name)
{
    if(x == nullptr)
    {
        throw std::invalid_argument(
            "NNCrossEntropyOp::forward: x must be non-null");
    }
    if(labels == nullptr)
    {
        throw std::invalid_argument(
            "NNCrossEntropyOp::forward: labels must be non-null");
    }
    if(labels->dtype() != DataType::INT64)
    {
        throw std::invalid_argument(
            "NNCrossEntropyOp::forward: labels must have INT64 dtype");
    }
    if(axis < 0 || axis >= x->ndim())
    {
        throw std::invalid_argument(
            "NNCrossEntropyOp::forward: axis out of range");
    }

    NNGraph* graph = x->graph();
    const auto& x_shape = x->shape();

    // labels shape: x.shape without axis (shape[1:] for axis 0)
    std::vector<Index> labels_shape;
    labels_shape.reserve(x->ndim() - 1);
    for(Index i = 0; i < x->ndim(); ++i)
    {
        if(i != axis)
        {
            labels_shape.push_back(x_shape[i]);
        }
    }
    if(labels->shape() != labels_shape)
    {
        throw std::invalid_argument(
            "NNCrossEntropyOp::forward: labels shape must match x shape "
            "without axis dimension");
    }

    bool out_requires_grad = any_input_requires_grad({x});

    TensorGraph& tg = graph->tensor_graph();

    // maxsumexp shape: [2] + shape without axis
    std::vector<Index> maxsumexp_shape;
    maxsumexp_shape.reserve(x->ndim());
    maxsumexp_shape.push_back(2);
    for(Index i = 0; i < x->ndim(); ++i)
    {
        if(i != axis)
        {
            maxsumexp_shape.push_back(x_shape[i]);
        }
    }
    maxsumexp_data_ =
        tg.data(maxsumexp_shape, output_name + "_mse", x->dtype());

    // logsumexp shape: shape without axis
    TensorGraph::TensorNode* logsumexp_data =
        tg.data(labels_shape, output_name + "_lse", x->dtype());

    // val: scalar
    TensorGraph::TensorNode* val_data =
        tg.data({}, output_name, x->dtype());

    // Forward: clear maxsumexp, maxsumexp, logsumexp, total_sum_accum
    // total_sum_accum and subtract_indexed_outputs require class dimension at
    // axis 0. When axis != 0, transpose x so class dimension is first.
    graph::tensor::clear(maxsumexp_data_);
    graph::tensor::maxsumexp(x->data(), maxsumexp_data_, axis, redux);
    graph::tensor::logsumexp(maxsumexp_data_, logsumexp_data);

    // val accumulates the loss; must be zero-initialized
    graph::tensor::clear(val_data);

    TensorGraph::TensorNode* x_for_accum = x->data();
    if(axis != 0)
    {
        std::vector<Index> x_t_shape(x->ndim());
        for(Index i = 0; i < x->ndim(); ++i)
        {
            x_t_shape[i] = x_shape[(i + axis) % x->ndim()];
        }
        TensorGraph::TensorNode* x_t_data =
            tg.data(x_t_shape, output_name + "_x_t", x->dtype());
        graph::tensor::transpose(1.0, x->data(), x_t_data, axis);
        x_for_accum = x_t_data;
    }
    graph::tensor::total_sum_accum(
        scale, logsumexp_data, x_for_accum, labels->data(),
        val_data, ignore_index);

    NNGraph::TensorNode* loss = graph->tensor(val_data, out_requires_grad);
    outputs_ = {loss};

    // Buffers for backward: maxsumexp (reused), grad_temp, and when axis!=0
    // also grad_temp_t (transposed view for subtract_indexed_outputs)
    NNGraph::TensorNode* grad_temp = graph->tensor(
        x_shape, output_name + "_gt", x->dtype(), false);
    if(axis == 0)
    {
        buffers_ = {grad_temp};
    }
    else
    {
        std::vector<Index> grad_temp_t_shape(x->ndim());
        for(Index i = 0; i < x->ndim(); ++i)
        {
            grad_temp_t_shape[i] = x_shape[(i + axis) % x->ndim()];
        }
        NNGraph::TensorNode* grad_temp_t = graph->tensor(
            grad_temp_t_shape, output_name + "_gt_t", x->dtype(), false);
        buffers_ = {grad_temp, grad_temp_t};
    }

    return loss;
}

void NNCrossEntropyOp::backward() const
{
    NNGraph::TensorNode* out = output();
    if(out == nullptr)
    {
        return;
    }
    NNGraph* graph = out->graph();
    NNGraph::TensorNode* grad_out = out->grad();
    if(grad_out == nullptr)
    {
        return;
    }
    if(x == nullptr || !x->requires_grad())
    {
        return;
    }

    if(buffers_.empty())
    {
        throw std::runtime_error(
            "NNCrossEntropyOp::backward: buffers are missing");
    }
    if(maxsumexp_data_ == nullptr)
    {
        throw std::runtime_error(
            "NNCrossEntropyOp::backward: maxsumexp_data_ is null");
    }
    NNGraph::TensorNode* grad_temp = buffers_[0];

    auto [grad_x, is_first] =
        graph->get_or_create_grad(x, x->name() + "_grad");

    // Recompute maxsumexp for backward (needed for softmax)
    graph::tensor::clear(maxsumexp_data_);
    graph::tensor::maxsumexp(x->data(), maxsumexp_data_, axis, redux);

    // grad_temp = scale * (softmax(x) - one_hot(labels))
    graph::tensor::softmax(
        maxsumexp_data_, x->data(), grad_temp->data(),
        scale, axis);

    // subtract_indexed_outputs requires class dimension at axis 0
    if(axis == 0)
    {
        graph::tensor::subtract_indexed_outputs(
            scale, labels->data(), grad_temp->data(), ignore_index);
    }
    else
    {
        NNGraph::TensorNode* grad_temp_t = buffers_[1];
        graph::tensor::transpose(1.0, grad_temp->data(), grad_temp_t->data(),
                                axis);
        graph::tensor::subtract_indexed_outputs(
            scale, labels->data(), grad_temp_t->data(), ignore_index);
        graph::tensor::transpose(1.0, grad_temp_t->data(), grad_temp->data(),
                                axis);
    }

    // grad_x += grad_out * grad_temp
    // For scalar loss, grad_out is typically 1.0. We add grad_temp to grad_x.
    Scalar grad_beta = is_first ? grad_overwrite : grad_accumulate;
    graph::tensor::add_inplace(1.0, grad_temp->data(), grad_beta,
                              grad_x->data());
}

NNGraph::TensorNode* cross_entropy(
    NNGraph::TensorNode* x,
    NNGraph::TensorNode* labels,
    const std::string& output_name,
    Index axis,
    int redux,
    Scalar scale,
    Index ignore_index)
{
    if(x == nullptr)
    {
        throw std::invalid_argument("cross_entropy: x must be non-null");
    }
    if(labels == nullptr)
    {
        throw std::invalid_argument("cross_entropy: labels must be non-null");
    }
    NNGraph* graph = x->graph();
    auto op = std::make_shared<NNCrossEntropyOp>(
        x, labels, axis, redux, scale, ignore_index);
    NNGraph::TensorNode* loss = op->forward(output_name);
    graph->register_op(std::move(op));
    return loss;
}

} // namespace nntile::graph
