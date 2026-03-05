/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn/scale_slice.cc
 * NNGraph scale_slice autograd implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn/scale_slice.hh"

#include <stdexcept>

#include "nntile/graph/tensor/scale_slice.hh"
#include "nntile/graph/tensor/sum_slice.hh"

namespace nntile::graph
{

namespace
{
constexpr Scalar grad_overwrite = 0.0;
constexpr Scalar grad_accumulate = 1.0;
constexpr int sum_slice_redux = 0;
} // anonymous namespace

NNGraph::TensorNode* NNScaleSliceOp::forward(const std::string& output_name)
{
    if(src == nullptr)
    {
        throw std::invalid_argument(
            "NNScaleSliceOp::forward: src must be non-null");
    }
    NNGraph* graph = src->graph();
    bool out_requires_grad = any_input_requires_grad({src});
    TensorGraph::TensorNode* output_data = graph::tensor::scale_slice(
        alpha, src->data(), output_name, axis, axis_size);
    NNGraph::TensorNode* output = graph->tensor(output_data, out_requires_grad);
    outputs_ = {output};
    return output;
}

void NNScaleSliceOp::backward() const
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
    if(src != nullptr && src->requires_grad())
    {
        auto [grad_src, is_first] =
            graph->get_or_create_grad(src, src->name() + "_grad");
        Scalar grad_beta = is_first ? grad_overwrite : grad_accumulate;
        graph::tensor::sum_slice(grad_out->data(), grad_src->data(),
                        axis, sum_slice_redux, alpha, grad_beta);
    }
}

NNGraph::TensorNode* scale_slice(
    Scalar alpha,
    NNGraph::TensorNode* src,
    const std::string& output_name,
    Index axis,
    Index axis_size)
{
    if(src == nullptr)
    {
        throw std::invalid_argument("scale_slice: src must be non-null");
    }
    NNGraph* graph = src->graph();
    auto op = std::make_shared<NNScaleSliceOp>(src, alpha, axis, axis_size);
    NNGraph::TensorNode* output = op->forward(output_name);
    graph->register_op(std::move(op));
    return output;
}

} // namespace nntile::graph
