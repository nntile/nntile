/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn/sum_fiber.cc
 * NNGraph sum_fiber autograd implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn/sum_fiber.hh"

#include <stdexcept>

#include "nntile/graph/tensor/add_fiber_inplace.hh"
#include "nntile/graph/tensor/sum_fiber.hh"

namespace nntile::graph
{

namespace
{
constexpr Scalar grad_overwrite = 0.0;
constexpr Scalar grad_accumulate = 1.0;
} // anonymous namespace

NNGraph::TensorNode* NNSumFiberOp::forward(const std::string& output_name)
{
    if(x == nullptr)
    {
        throw std::invalid_argument(
            "NNSumFiberOp::forward: x must be non-null");
    }
    NNGraph* graph = x->graph();
    bool out_requires_grad = any_input_requires_grad({x});
    TensorGraph::TensorNode* y_data = graph::tensor::sum_fiber(
        x->data(), output_name, axis, batch_ndim, redux, alpha, beta);
    NNGraph::TensorNode* y = graph->tensor(y_data, out_requires_grad);
    outputs_ = {y};
    return y;
}

void NNSumFiberOp::backward() const
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
    if(x != nullptr && x->requires_grad())
    {
        auto [grad_x, is_first] =
            graph->get_or_create_grad(x, x->name() + "_grad");
        Scalar grad_beta = is_first ? grad_overwrite : grad_accumulate;
        graph::tensor::add_fiber_inplace(alpha, grad_out->data(), grad_beta,
                                grad_x->data(), axis, batch_ndim);
    }
}

NNGraph::TensorNode* sum_fiber(
    NNGraph::TensorNode* x,
    const std::string& output_name,
    Index axis,
    Index batch_ndim,
    int redux,
    Scalar alpha,
    Scalar beta)
{
    if(x == nullptr)
    {
        throw std::invalid_argument("sum_fiber: x must be non-null");
    }
    NNGraph* graph = x->graph();
    auto op = std::make_shared<NNSumFiberOp>(
        x, axis, batch_ndim, redux, alpha, beta);
    NNGraph::TensorNode* y = op->forward(output_name);
    graph->register_op(std::move(op));
    return y;
}

} // namespace nntile::graph
