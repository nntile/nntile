/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn/softmax.cc
 * NNGraph softmax autograd implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn/softmax.hh"
#include "nntile/graph/nn/graph_data_node.hh"

#include <stdexcept>

#include "nntile/graph/tensor/add_inplace.hh"
#include "nntile/graph/tensor/add_slice.hh"
#include "nntile/graph/tensor/add_slice_inplace.hh"
#include "nntile/graph/tensor/copy.hh"
#include "nntile/graph/tensor/maxsumexp.hh"
#include "nntile/graph/tensor/multiply_inplace.hh"
#include "nntile/graph/tensor/softmax_inplace.hh"
#include "nntile/graph/tensor/sumprod_slice.hh"

namespace nntile::graph
{

namespace
{
constexpr Scalar grad_overwrite = 0.0;
constexpr Scalar grad_accumulate = 1.0;
} // anonymous namespace

NNGraph::TensorNode* NNSoftmaxOp::forward(const std::string& output_name)
{
    if(x == nullptr)
    {
        throw std::invalid_argument(
            "NNSoftmaxOp::forward: x must be non-null");
    }
    NNGraph* graph = x->graph();
    bool out_requires_grad = any_input_requires_grad({x});

    TensorGraph::TensorNode* y_data = graph::copy(x->data(), output_name);

    std::string mse_name = output_name + "_mse";
    TensorGraph::TensorNode* maxsumexp_buf =
        graph::maxsumexp(y_data, mse_name, axis, redux);

    graph::softmax_inplace(maxsumexp_buf, y_data, 1.0, axis);

    NNGraph::TensorNode* y = graph->tensor(y_data, out_requires_grad);
    outputs_ = {y};

    std::vector<Index> sumprod_shape;
    sumprod_shape.reserve(x->ndim() - 1);
    for(Index i = 0; i < x->ndim(); ++i)
    {
        if(i != axis)
        {
            sumprod_shape.push_back(x->shape()[i]);
        }
    }
    NNGraph::TensorNode* sumprod_buf = graph->tensor(
        sumprod_shape, output_name + "_sps", x->dtype(), false);
    NNGraph::TensorNode* grad_temp = graph->tensor(
        x->shape(), output_name + "_gt", x->dtype(), false);
    buffers_ = {sumprod_buf, grad_temp};

    return y;
}

void NNSoftmaxOp::backward() const
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

    if(buffers_.size() < 2)
    {
        throw std::runtime_error(
            "NNSoftmaxOp::backward: buffers are missing");
    }
    NNGraph::TensorNode* sumprod_buf = buffers_[0];
    NNGraph::TensorNode* grad_temp = buffers_[1];

    auto [grad_x, is_first] =
        graph->get_or_create_grad(x, x->name() + "_grad");

    graph::sumprod_slice(
        out->data(), grad_out->data(), sumprod_buf->data(),
        axis, redux, 1.0, 0.0);
    graph::add_slice(-1.0, sumprod_buf->data(), 1.0, grad_out->data(),
                    grad_temp->data(), axis);
    graph::multiply_inplace(1.0, out->data(), grad_temp->data());
    graph::add_inplace(1.0, grad_temp->data(),
                      is_first ? grad_overwrite : grad_accumulate,
                      grad_x->data());
}

NNGraph::TensorNode* softmax(
    NNGraph::TensorNode* x,
    const std::string& output_name,
    Index axis,
    int redux)
{
    if(x == nullptr)
    {
        throw std::invalid_argument("softmax: x must be non-null");
    }
    if(axis < 0 || axis >= x->ndim())
    {
        throw std::invalid_argument("softmax: axis out of range");
    }
    NNGraph* graph = x->graph();
    auto op = std::make_shared<NNSoftmaxOp>(x, axis, redux);
    NNGraph::TensorNode* y = op->forward(output_name);
    register_op(*graph, std::move(op));
    return y;
}

} // namespace nntile::graph
