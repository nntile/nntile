/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn/ops/layer_norm.cc
 * NNGraph LayerNorm autograd implementation.
 *
 * Forward: y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
 * Backward: grad_x, grad_gamma, grad_beta
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn/ops/layer_norm.hh"

#include "nntile/graph/nn/graph_data_node.hh"
#include "nntile/graph/nn/nn_grad_slot_name.hh"
#include "nntile/graph/tensor/ops/add_fiber_inplace.hh"
#include "nntile/graph/tensor/ops/add_inplace.hh"
#include "nntile/graph/tensor/ops/add_slice.hh"
#include "nntile/graph/tensor/ops/add_slice_inplace.hh"
#include "nntile/graph/tensor/ops/clear.hh"
#include "nntile/graph/tensor/ops/hypot_scalar_inverse.hh"
#include "nntile/graph/tensor/ops/multiply_fiber.hh"
#include "nntile/graph/tensor/ops/multiply_slice.hh"
#include "nntile/graph/tensor/ops/norm_slice_inplace.hh"
#include "nntile/graph/tensor/ops/sum_fiber.hh"
#include "nntile/graph/tensor/ops/sum_slice.hh"
#include "nntile/graph/tensor/ops/sumprod_fiber.hh"
#include "nntile/graph/tensor/ops/sumprod_slice.hh"

#include <cmath>
#include <stdexcept>

namespace nntile::graph
{

namespace
{
constexpr Scalar grad_overwrite = 0.0;
constexpr Scalar grad_accumulate = 1.0;
constexpr Index batch_ndim = 0;
} // anonymous namespace

NNGraph::TensorNode *NNLayerNormOp::forward(const std::string &output_name)
{
    if (x == nullptr || gamma == nullptr || beta == nullptr)
    {
        throw std::invalid_argument(
            "NNLayerNormOp::forward: x, gamma, and beta must be non-null");
    }
    NNGraph *graph = x->graph();
    bool out_requires_grad = any_input_requires_grad({x, gamma, beta});

    const Index l = x->shape()[axis];
    const Scalar inv_l = 1.0 / static_cast<Scalar>(l);
    const Scalar inv_sqrt_l = 1.0 / std::sqrt(static_cast<Scalar>(l));
    const Scalar eps_sqrt = std::sqrt(eps);

    std::vector<Index> reduced_shape;
    reduced_shape.reserve(x->ndim() - 1);
    for (Index i = 0; i < x->ndim(); ++i)
    {
        if (i != axis)
        {
            reduced_shape.push_back(x->shape()[i]);
        }
    }

    NNGraph::TensorNode *mean =
        graph->tensor(reduced_shape, x->dtype(), false);
    graph::tensor::sum_slice(
        x->data(), mean->data(), axis, redux, inv_l, 0.0);

    TensorGraph::TensorNode *tmp_y_data = graph::tensor::add_slice(
        -1.0, mean->data(), 1.0, x->data(), axis);
    NNGraph::TensorNode *tmp_y = graph->tensor(tmp_y_data, false);

    NNGraph::TensorNode *inv_stddev =
        graph->tensor(reduced_shape, x->dtype(), false);
    graph::tensor::norm_slice_inplace(
        inv_sqrt_l, tmp_y->data(), 0.0, inv_stddev->data(), axis, redux);
    graph::tensor::hypot_scalar_inverse(eps_sqrt, 1.0, inv_stddev->data());

    graph::tensor::multiply_slice(
        1.0, inv_stddev->data(), tmp_y->data(), axis);

    TensorGraph::TensorNode *y_data =
        graph::tensor::multiply_fiber(1.0, gamma->data(), tmp_y->data(), axis);
    NNGraph::TensorNode *y = graph->tensor(y_data, out_requires_grad);
    graph::tensor::add_fiber_inplace(
        1.0, beta->data(), 1.0, y->data(), axis, batch_ndim);
    y->set_name(output_name);

    outputs_ = {y};
    buffers_ = {inv_stddev, tmp_y, mean};

    return y;
}

void NNLayerNormOp::backward() const
{
    NNGraph::TensorNode *out = output();
    if (out == nullptr)
    {
        return;
    }
    NNGraph *graph = out->graph();
    NNGraph::TensorNode *grad_out = out->grad();
    if (grad_out == nullptr)
    {
        return;
    }

    if (buffers_.size() < 3)
    {
        throw std::runtime_error(
            "NNLayerNormOp::backward: buffers are missing");
    }
    NNGraph::TensorNode *inv_stddev = buffers_[0];
    NNGraph::TensorNode *tmp_y_value = buffers_[1];
    NNGraph::TensorNode *mean_buf = buffers_[2];

    const Scalar inv_l = 1.0 / static_cast<Scalar>(x->shape()[axis]);

    if (beta != nullptr && beta->requires_grad())
    {
        auto [grad_beta, is_first] =
            graph->get_or_create_grad(beta, nn_grad_slot_name(beta));
        Scalar beta_acc = is_first ? grad_overwrite : grad_accumulate;
        graph::tensor::sum_fiber(grad_out->data(),
            grad_beta->data(),
            axis,
            batch_ndim,
            redux,
            1.0,
            beta_acc);
    }

    if (gamma != nullptr && gamma->requires_grad())
    {
        auto [grad_gamma, is_first] =
            graph->get_or_create_grad(gamma, nn_grad_slot_name(gamma));
        Scalar gamma_acc = is_first ? grad_overwrite : grad_accumulate;
        graph::tensor::sumprod_fiber(grad_out->data(),
            tmp_y_value->data(),
            grad_gamma->data(),
            axis,
            redux,
            1.0,
            gamma_acc);
    }

    if (x != nullptr && x->requires_grad())
    {
        auto [grad_x, is_first] =
            graph->get_or_create_grad(x, nn_grad_slot_name(x));
        if (is_first)
        {
            graph::tensor::clear(grad_x->data());
        }

        TensorGraph::TensorNode *grad_temp_data = graph::tensor::multiply_fiber(
            1.0, gamma->data(), grad_out->data(), axis);
        NNGraph::TensorNode *grad_temp = graph->tensor(grad_temp_data, false);

        graph::tensor::sumprod_slice(grad_temp->data(),
            tmp_y_value->data(),
            mean_buf->data(),
            axis,
            redux,
            -inv_l,
            0.0);
        graph::tensor::multiply_slice(
            1.0, mean_buf->data(), tmp_y_value->data(), axis);
        graph::tensor::add_inplace(
            1.0, grad_temp->data(), 1.0, tmp_y_value->data());
        graph::tensor::sum_slice(
            grad_temp->data(), mean_buf->data(), axis, redux, inv_l, 0.0);
        graph::tensor::add_slice_inplace(
            -1.0, mean_buf->data(), 1.0, tmp_y_value->data(), axis);
        graph::tensor::multiply_slice(
            1.0, inv_stddev->data(), tmp_y_value->data(), axis);
        graph::tensor::add_inplace(
            1.0, tmp_y_value->data(), grad_accumulate, grad_x->data());
    }
}

NNGraph::TensorNode *layer_norm(NNGraph::TensorNode *x,
    NNGraph::TensorNode *gamma,
    NNGraph::TensorNode *beta,
    const std::string &output_name,
    Index axis,
    Scalar eps,
    int redux)
{
    if (x == nullptr || gamma == nullptr || beta == nullptr)
    {
        throw std::invalid_argument(
            "layer_norm: x, gamma, and beta must be non-null");
    }
    if (axis < 0 || axis >= x->ndim())
    {
        throw std::invalid_argument("layer_norm: axis out of range");
    }
    if (gamma->ndim() != 1 || gamma->shape()[0] != x->shape()[axis])
    {
        throw std::invalid_argument(
            "layer_norm: gamma must be 1D with shape matching x.shape[axis]");
    }
    if (beta->ndim() != 1 || beta->shape()[0] != x->shape()[axis])
    {
        throw std::invalid_argument(
            "layer_norm: beta must be 1D with shape matching x.shape[axis]");
    }
    NNGraph *graph = x->graph();
    auto op = std::make_shared<NNLayerNormOp>(x, gamma, beta, axis, eps, redux);
    NNGraph::TensorNode *y = op->forward(output_name);
    graph->register_op(std::move(op));
    return y;
}

} // namespace nntile::graph
