/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn/rms_norm.cc
 * NNGraph RMSNorm autograd implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn/rms_norm.hh"
#include "nntile/graph/nn/graph_data_node.hh"

#include <cmath>
#include <stdexcept>

#include "nntile/graph/tensor/add_inplace.hh"
#include "nntile/graph/tensor/clear.hh"
#include "nntile/graph/tensor/copy.hh"
#include "nntile/graph/tensor/hypot_scalar_inverse.hh"
#include "nntile/graph/tensor/multiply_fiber.hh"
#include "nntile/graph/tensor/multiply_slice.hh"
#include "nntile/graph/tensor/norm_slice_inplace.hh"
#include "nntile/graph/tensor/sumprod_fiber.hh"
#include "nntile/graph/tensor/sumprod_slice.hh"

namespace nntile::graph
{

namespace
{
constexpr Scalar grad_overwrite = 0.0;
constexpr Scalar grad_accumulate = 1.0;
} // anonymous namespace

NNGraph::TensorNode* NNRmsNormOp::forward(const std::string& output_name)
{
    if(x == nullptr || gamma == nullptr)
    {
        throw std::invalid_argument(
            "NNRmsNormOp::forward: x and gamma must be non-null");
    }
    NNGraph* graph = x->graph();
    bool out_requires_grad = any_input_requires_grad({x, gamma});

    const Index l = x->shape()[axis];
    const Scalar inv_sqrt_l = 1.0 / std::sqrt(static_cast<Scalar>(l));
    const Scalar eps_sqrt = std::sqrt(eps);

    // inv_stddev shape: reduced along axis
    std::vector<Index> inv_stddev_shape;
    inv_stddev_shape.reserve(x->ndim() - 1);
    for(Index i = 0; i < x->ndim(); ++i)
    {
        if(i != axis)
        {
            inv_stddev_shape.push_back(x->shape()[i]);
        }
    }

    NNGraph::TensorNode* inv_stddev = graph->tensor(
        inv_stddev_shape, output_name + "_inv_stddev", x->dtype(), false);

    graph::tensor::norm_slice_inplace(
        inv_sqrt_l, x->data(), 0.0, inv_stddev->data(), axis, redux);
    graph::tensor::hypot_scalar_inverse(eps_sqrt, 1.0, inv_stddev->data());

    TensorGraph::TensorNode* tmp_y_data =
        graph::tensor::copy(x->data(), output_name + "_tmp_y");
    NNGraph::TensorNode* tmp_y =
        graph->tensor(tmp_y_data, false);

    graph::tensor::multiply_slice(
        1.0, inv_stddev->data(), tmp_y->data(), axis);

    TensorGraph::TensorNode* y_data = graph::tensor::multiply_fiber(
        1.0, gamma->data(), tmp_y->data(), output_name, axis);
    NNGraph::TensorNode* y = graph->tensor(y_data, out_requires_grad);
    outputs_ = {y};

    // Buffers for backward: inv_stddev, tmp_y (normalized x), mean, grad_temp
    std::vector<Index> mean_shape = inv_stddev_shape;
    NNGraph::TensorNode* mean_buf = graph->tensor(
        mean_shape, output_name + "_mean", x->dtype(), false);
    NNGraph::TensorNode* grad_temp = graph->tensor(
        x->shape(), output_name + "_gt", x->dtype(), false);
    buffers_ = {inv_stddev, tmp_y, mean_buf, grad_temp};

    return y;
}

void NNRmsNormOp::backward() const
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

    if(buffers_.size() < 4)
    {
        throw std::runtime_error(
            "NNRmsNormOp::backward: buffers are missing");
    }
    NNGraph::TensorNode* inv_stddev = buffers_[0];
    NNGraph::TensorNode* tmp_y_value = buffers_[1];
    NNGraph::TensorNode* mean_buf = buffers_[2];
    NNGraph::TensorNode* grad_temp = buffers_[3];

    const Index l = x->shape()[axis];
    const Scalar inv_l = -1.0 / static_cast<Scalar>(l);

    if(gamma != nullptr && gamma->requires_grad())
    {
        auto [grad_gamma, is_first] =
            graph->get_or_create_grad(gamma, gamma->name() + "_grad");
        graph::tensor::sumprod_fiber(
            grad_out->data(), tmp_y_value->data(), grad_gamma->data(),
            axis, redux, 1.0, is_first ? grad_overwrite : grad_accumulate);
    }

    if(x != nullptr && x->requires_grad())
    {
        auto [grad_x, is_first] =
            graph->get_or_create_grad(x, x->name() + "_grad");
        if(is_first)
        {
            graph::tensor::clear(grad_x->data());
        }

        // grad_temp = gamma * grad_out
        graph::tensor::multiply_fiber(
            1.0, gamma->data(), grad_out->data(), grad_temp->data(), axis);
        // mean = -1/l * sumprod_slice(grad_temp, tmp_y_value)
        graph::tensor::sumprod_slice(
            grad_temp->data(), tmp_y_value->data(), mean_buf->data(),
            axis, redux, inv_l, 0.0);
        // tmp_y_value *= mean (in-place)
        graph::tensor::multiply_slice(
            1.0, mean_buf->data(), tmp_y_value->data(), axis);
        // tmp_y_value += grad_temp
        graph::tensor::add_inplace(
            1.0, grad_temp->data(), 1.0, tmp_y_value->data());
        // tmp_y_value *= inv_stddev
        graph::tensor::multiply_slice(
            1.0, inv_stddev->data(), tmp_y_value->data(), axis);
        // grad_x += tmp_y_value
        graph::tensor::add_inplace(
            1.0, tmp_y_value->data(), grad_accumulate, grad_x->data());
    }
}

NNGraph::TensorNode* rms_norm(
    NNGraph::TensorNode* x,
    NNGraph::TensorNode* gamma,
    const std::string& output_name,
    Index axis,
    Scalar eps,
    int redux)
{
    if(x == nullptr || gamma == nullptr)
    {
        throw std::invalid_argument(
            "rms_norm: x and gamma must be non-null");
    }
    if(axis < 0 || axis >= x->ndim())
    {
        throw std::invalid_argument("rms_norm: axis out of range");
    }
    if(gamma->ndim() != 1 || gamma->shape()[0] != x->shape()[axis])
    {
        throw std::invalid_argument(
            "rms_norm: gamma must be 1D with shape matching x.shape[axis]");
    }
    NNGraph* graph = x->graph();
    auto op = std::make_shared<NNRmsNormOp>(x, gamma, axis, eps, redux);
    NNGraph::TensorNode* y = op->forward(output_name);
    graph->register_op(std::move(op));
    return y;
}

} // namespace nntile::graph
