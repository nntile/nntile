/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/nn_graph/silu.cc
 * NNGraph SiLU autograd implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/nn/ops/silu.hh"

#include "nntile/nn/graph_data_node.hh"
#include "nntile/nn/nn_grad_slot_name.hh"
#include "nntile/tensor/ops/silu.hh"
#include "nntile/tensor/ops/silu_backward.hh"

#include <stdexcept>

namespace nntile
{

NNGraph::TensorNode *NNSiluOp::forward()
{
    if (x == nullptr)
    {
        throw std::invalid_argument("NNSiluOp::forward: x must be non-null");
    }
    NNGraph *graph = x->graph();
    bool out_requires_grad = any_input_requires_grad({x});
    TensorGraph::TensorNode *y_data = tensor::silu(x->data());
    NNGraph::TensorNode *y = graph->tensor(y_data, out_requires_grad);
    outputs_ = {y};
    return y;
}

void NNSiluOp::backward() const
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
    if (x != nullptr && x->requires_grad())
    {
        auto [grad_x, is_first] =
            graph->get_or_create_grad(x, nn_grad_slot_name(x));
        tensor::silu_backward(
            Scalar{1.0}, x->data(), grad_out->data(),
            is_first ? Scalar{0.0} : Scalar{1.0},
            grad_x->data());
    }
}

NNGraph::TensorNode *silu(NNGraph::TensorNode *x)
{
    if (x == nullptr)
    {
        throw std::invalid_argument("silu: x must be non-null");
    }
    NNGraph *graph = x->graph();
    auto op = std::make_shared<NNSiluOp>(x);
    NNGraph::TensorNode *y = op->forward();
    graph->register_op(std::move(op));
    return y;
}

} // namespace nntile
