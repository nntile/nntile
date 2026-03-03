/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn/add_fiber.cc
 * NNGraph add_fiber autograd implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn/add_fiber.hh"

#include <stdexcept>

#include "nntile/graph/tensor/add_fiber.hh"
#include "nntile/graph/tensor/add_fiber_inplace.hh"
#include "nntile/graph/tensor/add_inplace.hh"
#include "nntile/graph/tensor/sum_fiber.hh"

namespace nntile::graph
{

namespace
{
constexpr Scalar grad_overwrite = 0.0;
constexpr Scalar grad_accumulate = 1.0;
constexpr int sum_fiber_redux = 0;
} // anonymous namespace

NNGraph::TensorNode* NNAddFiberOp::forward(const std::string& output_name)
{
    if(fiber == nullptr || tensor == nullptr)
    {
        throw std::invalid_argument(
            "NNAddFiberOp::forward: fiber, tensor must be non-null");
    }
    NNGraph* graph = fiber->graph();
    bool out_requires_grad = any_input_requires_grad({fiber, tensor});
    NNGraph::TensorNode* output = graph->tensor(
        tensor->shape(), output_name, tensor->dtype(), out_requires_grad);
    outputs_ = {output};
    graph::add_fiber(alpha, fiber->data(), beta, tensor->data(),
                    output->data(), axis, batch_ndim);
    return output;
}

void NNAddFiberOp::backward() const
{
    NNGraph* graph = fiber->graph();
    NNGraph::TensorNode* grad_out = output()->grad();
    if(grad_out == nullptr)
    {
        return;
    }
    if(fiber != nullptr && fiber->requires_grad())
    {
        bool first = graph->is_first_grad(fiber);
        NNGraph::TensorNode* grad_fiber =
            graph->get_or_create_grad(fiber, fiber->name() + "_grad");
        graph::sum_fiber(grad_out->data(), grad_fiber->data(), axis,
                        batch_ndim, sum_fiber_redux, alpha,
                        first ? grad_overwrite : grad_accumulate);
    }
    if(tensor != nullptr && tensor->requires_grad())
    {
        bool first = graph->is_first_grad(tensor);
        NNGraph::TensorNode* grad_tensor =
            graph->get_or_create_grad(tensor, tensor->name() + "_grad");
        Scalar grad_beta = first ? grad_overwrite : grad_accumulate;
        graph::add_inplace(beta, grad_out->data(), grad_beta,
                          grad_tensor->data());
    }
}

NNGraph::TensorNode* add_fiber(
    Scalar alpha,
    NNGraph::TensorNode* fiber,
    Scalar beta,
    NNGraph::TensorNode* tensor,
    const std::string& output_name,
    Index axis,
    Index batch_ndim)
{
    if(fiber == nullptr || tensor == nullptr)
    {
        throw std::invalid_argument(
            "add_fiber: fiber and tensor must be non-null");
    }
    NNGraph* graph = fiber->graph();
    auto op = std::make_shared<NNAddFiberOp>(
        fiber, tensor, alpha, beta, axis, batch_ndim);
    NNGraph::TensorNode* output = op->forward(output_name);
    register_op(*graph, std::move(op));
    return output;
}

} // namespace nntile::graph
