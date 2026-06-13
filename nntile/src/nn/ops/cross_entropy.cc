#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/nn_graph/cross_entropy.cc
 * NNGraph cross_entropy autograd implementation.
 *
 * Based on wrappers/python/nntile/loss/crossentropy.py
 *
 * @version 1.1.0
 * */

#include "nntile/nn/ops/cross_entropy.hh"

#include "nntile/nn/graph_data_node.hh"
#include "nntile/nn/nn_grad_slot_name.hh"
#include "nntile/tensor/ops/add_inplace.hh"
#include "nntile/tensor/ops/clear.hh"
#include "nntile/tensor/ops/logsumexp.hh"
#include "nntile/tensor/ops/maxsumexp.hh"
#include "nntile/tensor/ops/softmax.hh"
#include "nntile/tensor/ops/subtract_indexed_outputs.hh"
#include "nntile/tensor/ops/total_sum_accum.hh"

#include <stdexcept>

namespace nntile
{

namespace
{
constexpr Scalar grad_overwrite = 0.0;
constexpr Scalar grad_accumulate = 1.0;
} // anonymous namespace

NNGraph::TensorNode *NNCrossEntropyOp::forward()
{
    if (x == nullptr)
    {
        throw std::invalid_argument(
            "NNCrossEntropyOp::forward: x must be non-null");
    }
    if (labels == nullptr)
    {
        throw std::invalid_argument(
            "NNCrossEntropyOp::forward: labels must be non-null");
    }
    if (labels->dtype() != DataType::INT64)
    {
        throw std::invalid_argument(
            "NNCrossEntropyOp::forward: labels must have INT64 dtype");
    }

    NNGraph *graph = x->graph();
    const auto &x_shape = x->shape();

    const Index class_axis = [&]() {
        if (x->ndim() == 3 && labels->ndim() == 2 &&
            labels->shape()[0] == x_shape[0] &&
            labels->shape()[1] == x_shape[2])
        {
            // Linear GEMM layout: [batch, vocab, seq]
            return Index(1);
        }
        return x->ndim() - 1;
    }();

    // Class dimension is innermost (or vocab axis for Linear logits).
    std::vector<Index> labels_shape;
    labels_shape.reserve(class_axis);
    for (Index i = 0; i < class_axis; ++i)
    {
        labels_shape.push_back(x_shape[i]);
    }
    for (Index i = class_axis + 1; i < x->ndim(); ++i)
    {
        labels_shape.push_back(x_shape[i]);
    }
    if (labels->shape() != labels_shape)
    {
        throw std::invalid_argument(
            "NNCrossEntropyOp::forward: labels shape must match x shape "
            "without axis dimension");
    }

    bool out_requires_grad = any_input_requires_grad({x});

    TensorGraph &tg = graph->tensor_graph();

    // maxsumexp shape: [2] + shape without class axis
    std::vector<Index> maxsumexp_shape;
    maxsumexp_shape.reserve(x->ndim());
    maxsumexp_shape.push_back(2);
    for (Index i = 0; i < x->ndim(); ++i)
    {
        if (i != class_axis)
        {
            maxsumexp_shape.push_back(x_shape[i]);
        }
    }
    maxsumexp_data_ = tg.data(maxsumexp_shape, x->dtype());

    // logsumexp shape: shape without axis
    TensorGraph::TensorNode *logsumexp_data =
        tg.data(labels_shape, x->dtype());

    // val: scalar
    TensorGraph::TensorNode *val_data = tg.data({}, x->dtype());

    // Forward: clear maxsumexp, maxsumexp, logsumexp, total_sum_accum
    tensor::clear(maxsumexp_data_);
    tensor::maxsumexp(x->data(), maxsumexp_data_, class_axis, redux);
    tensor::logsumexp(maxsumexp_data_, logsumexp_data);
    tensor::clear(val_data);
    tensor::total_sum_accum(scale,
        logsumexp_data,
        x->data(),
        labels->data(),
        val_data,
        ignore_index);

    NNGraph::TensorNode *loss = graph->tensor(val_data, out_requires_grad);
    outputs_ = {loss};

    // Buffers for backward: maxsumexp (reused), grad_temp
    NNGraph::TensorNode *grad_temp = graph->tensor(x_shape, x->dtype(), false);
    buffers_ = {grad_temp};

    return loss;
}

void NNCrossEntropyOp::backward() const
{
    NNGraph::TensorNode *out = output();
    if (out == nullptr)
    {
        return;
    }
    NNGraph *graph = out->graph();
    if (out->grad() == nullptr)
    {
        return;
    }
    if (x == nullptr || !x->requires_grad())
    {
        return;
    }

    if (buffers_.empty())
    {
        throw std::runtime_error(
            "NNCrossEntropyOp::backward: buffers are missing");
    }
    if (maxsumexp_data_ == nullptr)
    {
        throw std::runtime_error(
            "NNCrossEntropyOp::backward: maxsumexp_data_ is null");
    }
    NNGraph::TensorNode *grad_temp = buffers_[0];
    const Index class_axis = x->ndim() - 1;

    auto [grad_x, is_first] =
        graph->get_or_create_grad(x, nn_grad_slot_name(x));

    // Recompute maxsumexp for backward (needed for softmax)
    tensor::clear(maxsumexp_data_);
    tensor::maxsumexp(x->data(), maxsumexp_data_, class_axis, redux);

    // grad_temp = scale * (softmax(x) - one_hot(labels))
    tensor::softmax(
        maxsumexp_data_, x->data(), grad_temp->data(), scale, class_axis);
    tensor::subtract_indexed_outputs(
        scale, labels->data(), grad_temp->data(), ignore_index);

    // grad_x += grad_temp (gradient w.r.t. loss is implicitly 1.0, as in
    // scalar loss.backward())
    Scalar grad_beta = is_first ? grad_overwrite : grad_accumulate;
    tensor::add_inplace(
        1.0, grad_temp->data(), grad_beta, grad_x->data());
}

NNGraph::TensorNode *cross_entropy(NNGraph::TensorNode *x,
    NNGraph::TensorNode *labels,
    int redux,
    Scalar scale,
    Index ignore_index)
{
    if (x == nullptr)
    {
        throw std::invalid_argument("cross_entropy: x must be non-null");
    }
    if (labels == nullptr)
    {
        throw std::invalid_argument("cross_entropy: labels must be non-null");
    }
    NNGraph *graph = x->graph();
    auto op = std::make_shared<NNCrossEntropyOp>(
        x, labels, redux, scale, ignore_index);
    NNGraph::TensorNode *loss = op->forward();
    graph->register_op(std::move(op));
    return loss;
}

} // namespace nntile
