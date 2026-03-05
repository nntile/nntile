/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn/norm.cc
 * NNGraph norm autograd implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn/norm.hh"

#include <stdexcept>

#include "nntile/graph/tensor/clear.hh"
#include "nntile/graph/tensor/norm.hh"

namespace nntile::graph
{

NNGraph::TensorNode* NNNormOp::forward(const std::string& output_name)
{
    if(x == nullptr)
    {
        throw std::invalid_argument(
            "NNNormOp::forward: x must be non-null");
    }
    NNGraph* graph = x->graph();
    bool out_requires_grad = any_input_requires_grad({x});
    NNGraph::TensorNode* y = graph->tensor({}, output_name, x->dtype(),
                                           out_requires_grad);
    graph::tensor::clear(y->data());
    constexpr Scalar beta_fresh = 0.0;  // NNGraph always outputs fresh data
    graph::tensor::norm(x->data(), y->data(), alpha, beta_fresh);
    outputs_ = {y};
    return y;
}

void NNNormOp::backward() const
{
    if(x != nullptr && x->requires_grad())
    {
        throw std::runtime_error(
            "norm backward is not implemented");
    }
}

NNGraph::TensorNode* norm(
    NNGraph::TensorNode* x,
    const std::string& output_name,
    Scalar alpha)
{
    if(x == nullptr)
    {
        throw std::invalid_argument("norm: x must be non-null");
    }
    NNGraph* graph = x->graph();
    auto op = std::make_shared<NNNormOp>(x, alpha);
    NNGraph::TensorNode* y = op->forward(output_name);
    graph->register_op(std::move(op));
    return y;
}

} // namespace nntile::graph
