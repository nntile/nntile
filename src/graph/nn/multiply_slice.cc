/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn/multiply_slice.cc
 * NNGraph multiply_slice autograd implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn/multiply_slice.hh"

#include <stdexcept>
#include <utility>

#include "nntile/graph/tensor/add_inplace.hh"
#include "nntile/graph/tensor/copy.hh"
#include "nntile/graph/tensor/multiply.hh"
#include "nntile/graph/tensor/multiply_slice.hh"
#include "nntile/graph/tensor/sum_slice.hh"

namespace nntile::graph
{

namespace
{
constexpr Scalar grad_overwrite = 0.0;
constexpr Scalar grad_accumulate = 1.0;
constexpr int sum_slice_redux = 0;
} // anonymous namespace

NNGraph::TensorNode* NNMultiplySliceOp::forward(const std::string& output_name)
{
    if(slice == nullptr || tensor == nullptr)
    {
        throw std::invalid_argument(
            "NNMultiplySliceOp::forward: slice and tensor must be non-null");
    }
    NNGraph* graph = slice->graph();
    TensorGraph::TensorNode* slice_data = slice->data();
    TensorGraph::TensorNode* tensor_data = tensor->data();
    bool out_requires_grad = any_input_requires_grad({slice, tensor});

    TensorGraph::TensorNode* dst = graph::tensor::copy(tensor_data, output_name);
    graph::tensor::multiply_slice(alpha, slice_data, dst, axis);

    NNGraph::TensorNode* output = graph->tensor(dst, out_requires_grad);
    outputs_ = {output};
    return output;
}

void NNMultiplySliceOp::backward() const
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
    if(slice != nullptr && slice->requires_grad())
    {
        auto [grad_slice, is_first] =
            graph->get_or_create_grad(slice, slice->name() + "_grad");
        Scalar grad_beta = is_first ? grad_overwrite : grad_accumulate;
        TensorGraph::TensorNode* buf = graph::tensor::multiply(
            grad_out->data(), tensor->data(),
            out->name() + "_grad_slice_buf", 1.0);
        graph::tensor::sum_slice(buf, grad_slice->data(),
                        axis, sum_slice_redux, alpha, grad_beta);
    }
    if(tensor != nullptr && tensor->requires_grad())
    {
        auto [grad_tensor, is_first] =
            graph->get_or_create_grad(tensor, tensor->name() + "_grad");
        Scalar grad_beta = is_first ? grad_overwrite : grad_accumulate;
        TensorGraph::TensorNode* buf = graph::tensor::copy(
            grad_out->data(), out->name() + "_grad_tensor_buf");
        graph::tensor::multiply_slice(alpha, slice->data(), buf, axis);
        graph::tensor::add_inplace(1.0, buf, grad_beta, grad_tensor->data());
    }
}

NNGraph::TensorNode* multiply_slice(
    Scalar alpha,
    NNGraph::TensorNode* slice,
    NNGraph::TensorNode* tensor,
    const std::string& output_name,
    Index axis)
{
    if(slice == nullptr || tensor == nullptr)
    {
        throw std::invalid_argument(
            "multiply_slice: slice and tensor must be non-null");
    }
    NNGraph* graph = slice->graph();
    auto op = std::make_shared<NNMultiplySliceOp>(slice, tensor, alpha, axis);
    NNGraph::TensorNode* output = op->forward(output_name);
    graph->register_op(std::move(op));
    return output;
}

} // namespace nntile::graph
