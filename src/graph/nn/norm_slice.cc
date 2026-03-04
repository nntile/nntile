/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn/norm_slice.cc
 * NNGraph norm_slice autograd implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn/norm_slice.hh"

#include <stdexcept>
#include <utility>
#include <vector>

#include "nntile/graph/tensor/clear.hh"
#include "nntile/graph/tensor/norm_slice.hh"

namespace nntile::graph
{

namespace
{

std::vector<Index> norm_slice_output_shape(
    const std::vector<Index>& x_shape,
    Index axis)
{
    std::vector<Index> out_shape;
    out_shape.reserve(x_shape.size() - 1);
    for(Index i = 0; i < static_cast<Index>(x_shape.size()); ++i)
    {
        if(i != axis)
        {
            out_shape.push_back(x_shape[i]);
        }
    }
    return out_shape;
}

} // anonymous namespace

NNGraph::TensorNode* NNNormSliceOp::forward(const std::string& output_name)
{
    if(x == nullptr)
    {
        throw std::invalid_argument(
            "NNNormSliceOp::forward: x must be non-null");
    }
    NNGraph* graph = x->graph();
    bool out_requires_grad = any_input_requires_grad({x});
    std::vector<Index> base_shape =
        norm_slice_output_shape(x->shape(), axis);
    NNGraph::TensorNode* base = graph->tensor(
        std::move(base_shape), output_name + "_base", x->dtype(), false);
    graph::tensor::clear(base->data());
    constexpr Scalar beta_fresh = 0.0;  // NNGraph always outputs fresh data
    TensorGraph::TensorNode* y_data = graph::tensor::norm_slice(
        alpha, x->data(), beta_fresh, base->data(),
        output_name, axis, redux);
    NNGraph::TensorNode* y = graph->tensor(y_data, out_requires_grad);
    outputs_ = {y};
    return y;
}

void NNNormSliceOp::backward() const
{
    if(x != nullptr && x->requires_grad())
    {
        throw std::runtime_error(
            "norm_slice backward is not implemented");
    }
}

NNGraph::TensorNode* norm_slice(
    NNGraph::TensorNode* x,
    const std::string& output_name,
    Index axis,
    int redux,
    Scalar alpha)
{
    if(x == nullptr)
    {
        throw std::invalid_argument("norm_slice: x must be non-null");
    }
    NNGraph* graph = x->graph();
    auto op = std::make_shared<NNNormSliceOp>(
        x, axis, redux, alpha);
    NNGraph::TensorNode* y = op->forward(output_name);
    graph->register_op(std::move(op));
    return y;
}

} // namespace nntile::graph
