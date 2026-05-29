#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/nn_graph/multiply_slice.cc
 * NNGraph multiply_slice autograd implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/nn_graph/ops/multiply_slice.hh"

#include "nntile/nn_graph/nn_grad_slot_name.hh"
#include "nntile/tensor_graph/ops/add_inplace.hh"
#include "nntile/tensor_graph/ops/copy.hh"
#include "nntile/tensor_graph/ops/multiply.hh"
#include "nntile/tensor_graph/ops/multiply_slice.hh"
#include "nntile/tensor_graph/ops/sum_slice.hh"

#include <stdexcept>
#include <utility>

namespace nntile
{

namespace
{
constexpr Scalar grad_overwrite = 0.0;
constexpr Scalar grad_accumulate = 1.0;
constexpr int sum_slice_redux = 0;
} // anonymous namespace

NNGraph::TensorNode *NNMultiplySliceOp::forward()
{
    if (slice == nullptr || tensor == nullptr)
    {
        throw std::invalid_argument(
            "NNMultiplySliceOp::forward: slice and tensor must be non-null");
    }
    NNGraph *graph = slice->graph();
    TensorGraph::TensorNode *slice_data = slice->data();
    TensorGraph::TensorNode *tensor_data = tensor->data();
    bool out_requires_grad = any_input_requires_grad({slice, tensor});

    TensorGraph::TensorNode *dst = tensor_graph::copy(tensor_data);
    tensor_graph::multiply_slice(alpha, slice_data, dst, axis);

    NNGraph::TensorNode *output = graph->tensor(dst, out_requires_grad);
    outputs_ = {output};
    return output;
}

void NNMultiplySliceOp::backward() const
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
    if (slice != nullptr && slice->requires_grad())
    {
        auto [grad_slice, is_first] =
            graph->get_or_create_grad(slice, nn_grad_slot_name(slice));
        Scalar grad_beta = is_first ? grad_overwrite : grad_accumulate;
        TensorGraph::TensorNode *buf =
            tensor_graph::multiply(grad_out->data(), tensor->data(), 1.0);
        tensor_graph::sum_slice(
            buf, grad_slice->data(), axis, sum_slice_redux, alpha, grad_beta);
    }
    if (tensor != nullptr && tensor->requires_grad())
    {
        auto [grad_tensor, is_first] =
            graph->get_or_create_grad(tensor, nn_grad_slot_name(tensor));
        Scalar grad_beta = is_first ? grad_overwrite : grad_accumulate;
        TensorGraph::TensorNode *buf = tensor_graph::copy(grad_out->data());
        tensor_graph::multiply_slice(alpha, slice->data(), buf, axis);
        tensor_graph::add_inplace(1.0, buf, grad_beta, grad_tensor->data());
    }
}

NNGraph::TensorNode *multiply_slice(Scalar alpha,
    NNGraph::TensorNode *slice,
    NNGraph::TensorNode *tensor,
    Index axis)
{
    if (slice == nullptr || tensor == nullptr)
    {
        throw std::invalid_argument(
            "multiply_slice: slice and tensor must be non-null");
    }
    NNGraph *graph = slice->graph();
    auto op = std::make_shared<NNMultiplySliceOp>(slice, tensor, alpha, axis);
    NNGraph::TensorNode *output = op->forward();
    graph->register_op(std::move(op));
    return output;
}

} // namespace nntile
