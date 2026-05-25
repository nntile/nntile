/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn/multiply_fiber.cc
 * NNGraph multiply_fiber autograd implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn/ops/multiply_fiber.hh"

#include "nntile/graph/nn/nn_grad_slot_name.hh"
#include "nntile/graph/tensor/ops/add_inplace.hh"
#include "nntile/graph/tensor/ops/multiply.hh"
#include "nntile/graph/tensor/ops/multiply_fiber.hh"
#include "nntile/graph/tensor/ops/sum_fiber.hh"

#include <stdexcept>

namespace nntile::graph
{

namespace
{
constexpr Scalar grad_overwrite = 0.0;
constexpr Scalar grad_accumulate = 1.0;
constexpr Index batch_ndim_fiber = 0;
constexpr int sum_fiber_redux = 0;
} // anonymous namespace

NNGraph::TensorNode *NNMultiplyFiberOp::forward()
{
    if (src1 == nullptr || src2 == nullptr)
    {
        throw std::invalid_argument(
            "NNMultiplyFiberOp::forward: src1, src2 must be non-null");
    }
    NNGraph *graph = src1->graph();
    bool out_requires_grad = any_input_requires_grad({src1, src2});
    TensorGraph::TensorNode *output_data =
        graph::tensor::multiply_fiber(alpha, src1->data(), src2->data(), axis);
    NNGraph::TensorNode *output =
        graph->tensor(output_data, out_requires_grad);
    outputs_ = {output};
    return output;
}

void NNMultiplyFiberOp::backward() const
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
    if (src1 != nullptr && src1->requires_grad())
    {
        auto [grad_src1, is_first] =
            graph->get_or_create_grad(src1, nn_grad_slot_name(src1));
        Scalar grad_beta = is_first ? grad_overwrite : grad_accumulate;
        TensorGraph::TensorNode *grad_src1_buf =
            graph::tensor::multiply(grad_out->data(), src2->data(), 1.0);
        graph::tensor::sum_fiber(grad_src1_buf,
            grad_src1->data(),
            axis,
            batch_ndim_fiber,
            sum_fiber_redux,
            alpha,
            grad_beta);
    }
    if (src2 != nullptr && src2->requires_grad())
    {
        auto [grad_src2, is_first] =
            graph->get_or_create_grad(src2, nn_grad_slot_name(src2));
        Scalar grad_beta = is_first ? grad_overwrite : grad_accumulate;
        TensorGraph::TensorNode *grad_src2_buf = graph::tensor::multiply_fiber(
            alpha, src1->data(), grad_out->data(), axis);
        graph::tensor::add_inplace(
            1.0, grad_src2_buf, grad_beta, grad_src2->data());
    }
}

NNGraph::TensorNode *multiply_fiber(Scalar alpha,
    NNGraph::TensorNode *src1,
    NNGraph::TensorNode *src2,
    Index axis)
{
    if (src1 == nullptr || src2 == nullptr)
    {
        throw std::invalid_argument(
            "multiply_fiber: src1 and src2 must be non-null");
    }
    NNGraph *graph = src1->graph();
    auto op = std::make_shared<NNMultiplyFiberOp>(src1, src2, alpha, axis);
    NNGraph::TensorNode *output = op->forward();
    graph->register_op(std::move(op));
    return output;
}

} // namespace nntile::graph
