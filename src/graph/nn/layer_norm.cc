/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn/layer_norm.cc
 * NNGraph LayerNorm autograd implementation.
 *
 * Forward: y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
 * Backward: grad_x, grad_gamma, grad_beta
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn/layer_norm.hh"
#include "nntile/graph/nn/graph_data_node.hh"

#include <cmath>
#include <stdexcept>

#include "nntile/graph/tensor/add_fiber_inplace.hh"
#include "nntile/graph/tensor/add_inplace.hh"
#include "nntile/graph/tensor/add_slice.hh"
#include "nntile/graph/tensor/add_slice_inplace.hh"
#include "nntile/graph/tensor/clear.hh"
#include "nntile/graph/tensor/copy.hh"
#include "nntile/graph/tensor/hypot_scalar_inverse.hh"
#include "nntile/graph/tensor/multiply_fiber.hh"
#include "nntile/graph/tensor/multiply_slice.hh"
#include "nntile/graph/tensor/norm_slice_inplace.hh"
#include "nntile/graph/tensor/sum_fiber.hh"
#include "nntile/graph/tensor/sum_slice.hh"
#include "nntile/graph/tensor/sumprod_fiber.hh"
#include "nntile/graph/tensor/sumprod_slice.hh"

namespace nntile::graph
{

namespace
{
constexpr Scalar grad_overwrite = 0.0;
constexpr Scalar grad_accumulate = 1.0;
constexpr Index batch_ndim = 0;
} // anonymous namespace

NNGraph::TensorNode* NNLayerNormOp::forward(const std::string& output_name)
{
    if(x == nullptr || gamma == nullptr || beta == nullptr)
    {
        throw std::invalid_argument(
            "NNLayerNormOp::forward: x, gamma, and beta must be non-null");
    }
    NNGraph* graph = x->graph();
    bool out_requires_grad = any_input_requires_grad({x, gamma, beta});

    const Index l = x->shape()[axis];
    const Scalar inv_l = 1.0 / static_cast<Scalar>(l);
    const Scalar inv_sqrt_l = 1.0 / std::sqrt(static_cast<Scalar>(l));
    const Scalar eps_sqrt = std::sqrt(eps);

    // Reduced shape (without axis)
    std::vector<Index> reduced_shape;
    reduced_shape.reserve(x->ndim() - 1);
    for(Index i = 0; i < x->ndim(); ++i)
    {
        if(i != axis)
        {
            reduced_shape.push_back(x->shape()[i]);
        }
    }

    // mean = (1/l) * sum_slice(x)
    NNGraph::TensorNode* mean = graph->tensor(
        reduced_shape, output_name + "_mean", x->dtype(), false);
    graph::tensor::sum_slice(
        x->data(), mean->data(), axis, redux, inv_l, 0.0);

    // tmp_y = x - mean
    TensorGraph::TensorNode* tmp_y_data = graph::tensor::add_slice(
        -1.0, mean->data(), 1.0, x->data(),
        output_name + "_tmp_y", axis);
    NNGraph::TensorNode* tmp_y =
        graph->tensor(tmp_y_data, false);

    // inv_stddev = 1/sqrt(var + eps), where var = (1/l)*sum((x-mean)^2)
    NNGraph::TensorNode* inv_stddev = graph->tensor(
        reduced_shape, output_name + "_inv_stddev", x->dtype(), false);
    graph::tensor::norm_slice_inplace(
        inv_sqrt_l, tmp_y->data(), 0.0, inv_stddev->data(), axis, redux);
    graph::tensor::hypot_scalar_inverse(eps_sqrt, 1.0, inv_stddev->data());

    // tmp_y *= inv_stddev (normalize)
    graph::tensor::multiply_slice(
        1.0, inv_stddev->data(), tmp_y->data(), axis);

    // y = gamma * tmp_y + beta
    TensorGraph::TensorNode* y_data = graph::tensor::multiply_fiber(
        1.0, gamma->data(), tmp_y->data(), output_name, axis);
    NNGraph::TensorNode* y = graph->tensor(y_data, out_requires_grad);
    graph::tensor::add_fiber_inplace(
        1.0, beta->data(), 1.0, y->data(), axis, batch_ndim);

    outputs_ = {y};

    // Buffers for backward: inv_stddev, tmp_y (normalized x), mean
    buffers_ = {inv_stddev, tmp_y, mean};

    return y;
}

void NNLayerNormOp::backward() const
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

    if(buffers_.size() < 3)
    {
        throw std::runtime_error(
            "NNLayerNormOp::backward: buffers are missing");
    }
    NNGraph::TensorNode* inv_stddev = buffers_[0];
    NNGraph::TensorNode* tmp_y_value = buffers_[1];
    NNGraph::TensorNode* mean_buf = buffers_[2];

    const Index l = x->shape()[axis];
    const Scalar inv_l = 1.0 / static_cast<Scalar>(l);

    // grad_beta = sum_fiber(grad_out)
    if(beta != nullptr && beta->requires_grad())
    {
        auto [grad_beta, is_first] =
            graph->get_or_create_grad(beta, beta->name() + "_grad");
        Scalar beta_acc = is_first ? grad_overwrite : grad_accumulate;
        graph::tensor::sum_fiber(
            grad_out->data(), grad_beta->data(),
            axis, batch_ndim, redux, 1.0, beta_acc);
    }

    // grad_gamma = sumprod_fiber(grad_out, tmp_y_value)
    if(gamma != nullptr && gamma->requires_grad())
    {
        auto [grad_gamma, is_first] =
            graph->get_or_create_grad(gamma, gamma->name() + "_grad");
        Scalar gamma_acc = is_first ? grad_overwrite : grad_accumulate;
        graph::tensor::sumprod_fiber(
            grad_out->data(), tmp_y_value->data(), grad_gamma->data(),
            axis, redux, 1.0, gamma_acc);
    }

    // grad_x
    if(x != nullptr && x->requires_grad())
    {
        auto [grad_x, is_first] =
            graph->get_or_create_grad(x, x->name() + "_grad");
        if(is_first)
        {
            graph::tensor::clear(grad_x->data());
        }

        // tmp_y_grad = gamma * grad_out (use tmp_y_value as workspace)
        // We need a separate buffer - tmp_y_value is needed for sumprod above.
        // Create grad_temp for gamma * grad_out
        TensorGraph::TensorNode* grad_temp_data = graph::tensor::multiply_fiber(
            1.0, gamma->data(), grad_out->data(),
            output()->name() + "_grad_temp", axis);
        NNGraph::TensorNode* grad_temp = graph->tensor(grad_temp_data, false);

        // mean = -1/l * sumprod_slice(grad_temp, tmp_y_value)
        graph::tensor::sumprod_slice(
            grad_temp->data(), tmp_y_value->data(), mean_buf->data(),
            axis, redux, -inv_l, 0.0);
        // tmp_y_value *= mean (in-place)
        graph::tensor::multiply_slice(
            1.0, mean_buf->data(), tmp_y_value->data(), axis);
        // tmp_y_value += grad_temp
        graph::tensor::add_inplace(
            1.0, grad_temp->data(), 1.0, tmp_y_value->data());
        // mean = 1/l * sum_slice(grad_temp)
        graph::tensor::sum_slice(
            grad_temp->data(), mean_buf->data(),
            axis, redux, inv_l, 0.0);
        // tmp_y_value -= mean (add_slice_inplace: dst = alpha*src + beta*dst)
        graph::tensor::add_slice_inplace(
            -1.0, mean_buf->data(), 1.0, tmp_y_value->data(), axis);
        // tmp_y_value *= inv_stddev
        graph::tensor::multiply_slice(
            1.0, inv_stddev->data(), tmp_y_value->data(), axis);
        // grad_x += tmp_y_value
        graph::tensor::add_inplace(
            1.0, tmp_y_value->data(), grad_accumulate, grad_x->data());
    }
}

NNGraph::TensorNode* layer_norm(
    NNGraph::TensorNode* x,
    NNGraph::TensorNode* gamma,
    NNGraph::TensorNode* beta,
    const std::string& output_name,
    Index axis,
    Scalar eps,
    int redux)
{
    if(x == nullptr || gamma == nullptr || beta == nullptr)
    {
        throw std::invalid_argument(
            "layer_norm: x, gamma, and beta must be non-null");
    }
    if(axis < 0 || axis >= x->ndim())
    {
        throw std::invalid_argument("layer_norm: axis out of range");
    }
    if(gamma->ndim() != 1 || gamma->shape()[0] != x->shape()[axis])
    {
        throw std::invalid_argument(
            "layer_norm: gamma must be 1D with shape matching x.shape[axis]");
    }
    if(beta->ndim() != 1 || beta->shape()[0] != x->shape()[axis])
    {
        throw std::invalid_argument(
            "layer_norm: beta must be 1D with shape matching x.shape[axis]");
    }
    NNGraph* graph = x->graph();
    auto op = std::make_shared<NNLayerNormOp>(
        x, gamma, beta, axis, eps, redux);
    NNGraph::TensorNode* y = op->forward(output_name);
    graph->register_op(std::move(op));
    return y;
}

} // namespace nntile::graph
