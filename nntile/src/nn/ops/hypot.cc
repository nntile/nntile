/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/nn_graph/hypot.cc
 * NNGraph hypot autograd implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/nn/ops/hypot.hh"

#include "nntile/nn/nn_grad_slot_name.hh"
#include "nntile/tensor/ops/hypot.hh"

#include <stdexcept>

namespace nntile
{

NNGraph::TensorNode *NNHypotOp::forward()
{
    if (x == nullptr || y == nullptr)
    {
        throw std::invalid_argument(
            "NNHypotOp::forward: x, y must be non-null");
    }
    NNGraph *graph = x->graph();
    bool out_requires_grad = any_input_requires_grad({x, y});
    TensorGraph::TensorNode *z_data =
        tensor::hypot(alpha, x->data(), beta, y->data());
    NNGraph::TensorNode *z = graph->tensor(z_data, out_requires_grad);
    outputs_ = {z};
    return z;
}

void NNHypotOp::backward() const
{
    if ((x != nullptr && x->requires_grad()) ||
        (y != nullptr && y->requires_grad()))
    {
        throw std::runtime_error("hypot backward is not implemented");
    }
}

NNGraph::TensorNode *hypot(NNGraph::TensorNode *x,
    NNGraph::TensorNode *y,
    Scalar alpha,
    Scalar beta)
{
    if (x == nullptr || y == nullptr)
    {
        throw std::invalid_argument("hypot: x and y must be non-null");
    }
    NNGraph *graph = x->graph();
    auto op = std::make_shared<NNHypotOp>(x, y, alpha, beta);
    NNGraph::TensorNode *z = op->forward();
    graph->register_op(std::move(op));
    return z;
}

} // namespace nntile
