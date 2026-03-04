/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn/rope.cc
 * NNGraph RoPE autograd implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn/rope.hh"
#include "nntile/graph/nn/graph_data_node.hh"

#include <stdexcept>

#include "nntile/graph/tensor/add_inplace.hh"
#include "nntile/graph/tensor/rope.hh"
#include "nntile/graph/tensor/rope_backward.hh"

namespace nntile::graph
{

namespace
{
constexpr Scalar grad_overwrite = 0.0;
constexpr Scalar grad_accumulate = 1.0;
} // anonymous namespace

NNGraph::TensorNode* NNRopeOp::forward(const std::string& output_name)
{
    if(sin == nullptr || cos == nullptr || x == nullptr)
    {
        throw std::invalid_argument(
            "NNRopeOp::forward: sin, cos, x must be non-null");
    }
    NNGraph* graph = x->graph();
    bool out_requires_grad = any_input_requires_grad({x});
    TensorGraph::TensorNode* y_data = graph::rope(
        sin->data(), cos->data(), x->data(), output_name);
    NNGraph::TensorNode* y = graph->tensor(y_data, out_requires_grad);
    outputs_ = {y};
    if(x->requires_grad())
    {
        NNGraph::TensorNode* grad_buf =
            graph->tensor(x->shape(), output_name + "_gb", x->dtype(), false);
        buffers_ = {grad_buf};
    }
    return y;
}

void NNRopeOp::backward() const
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
    if(x == nullptr || !x->requires_grad())
    {
        return;
    }

    auto [grad_x, is_first] =
        graph->get_or_create_grad(x, x->name() + "_grad");

    if(is_first)
    {
        graph::rope_backward(sin->data(), cos->data(), grad_out->data(),
                            grad_x->data());
    }
    else
    {
        NNGraph::TensorNode* grad_buf = buffers_.empty() ? nullptr : buffers_[0];
        if(grad_buf == nullptr)
        {
            throw std::runtime_error(
                "NNRopeOp::backward: gradient buffer is missing");
        }
        graph::rope_backward(sin->data(), cos->data(), grad_out->data(),
                            grad_buf->data());
        graph::add_inplace(1.0, grad_buf->data(), grad_accumulate,
                          grad_x->data());
    }
}

NNGraph::TensorNode* rope(
    NNGraph::TensorNode* sin,
    NNGraph::TensorNode* cos,
    NNGraph::TensorNode* x,
    const std::string& output_name)
{
    if(sin == nullptr || cos == nullptr || x == nullptr)
    {
        throw std::invalid_argument("rope: sin, cos, x must be non-null");
    }
    NNGraph* graph = x->graph();
    auto op = std::make_shared<NNRopeOp>(sin, cos, x);
    NNGraph::TensorNode* y = op->forward(output_name);
    register_op(*graph, std::move(op));
    return y;
}

} // namespace nntile::graph
