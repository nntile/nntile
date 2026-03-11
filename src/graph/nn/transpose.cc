/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn/transpose.cc
 * NNGraph transpose autograd implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn/transpose.hh"
#include "nntile/graph/nn/graph_data_node.hh"

#include <stdexcept>

#include "nntile/graph/tensor/add_inplace.hh"
#include "nntile/graph/tensor/transpose.hh"

namespace nntile::graph
{

namespace
{
constexpr Scalar grad_overwrite = 0.0;
constexpr Scalar grad_accumulate = 1.0;
} // anonymous namespace

NNGraph::TensorNode* NNTransposeOp::forward(const std::string& output_name)
{
    if(src == nullptr)
    {
        throw std::invalid_argument(
            "NNTransposeOp::forward: src must be non-null");
    }
    if(ndim <= 0 || ndim >= src->ndim())
    {
        throw std::invalid_argument(
            "NNTransposeOp::forward: ndim must be in (0, src.ndim)");
    }
    NNGraph* graph = src->graph();
    bool out_requires_grad = any_input_requires_grad({src});
    TensorGraph::TensorNode* output_data = graph::tensor::transpose(
        1.0, src->data(), output_name, ndim);
    NNGraph::TensorNode* output = graph->tensor(output_data, out_requires_grad);
    outputs_ = {output};
    if(src->requires_grad())
    {
        NNGraph::TensorNode* grad_buf = graph->tensor(
            src->shape(), output_name + "_gb", src->dtype(), false);
        buffers_ = {grad_buf};
    }
    return output;
}

void NNTransposeOp::backward() const
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
        Index inv_ndim = src->ndim() - ndim;
        if(inv_ndim <= 0)
        {
            inv_ndim += src->ndim();
        }
        if(is_first)
        {
            graph::tensor::transpose(1.0, grad_out->data(), grad_src->data(),
                                    inv_ndim);
        }
        else
        {
            NNGraph::TensorNode* grad_buf =
                buffers_.empty() ? nullptr : buffers_[0];
            if(grad_buf == nullptr)
            {
                throw std::runtime_error(
                    "NNTransposeOp::backward: gradient buffer is missing");
            }
            graph::tensor::transpose(1.0, grad_out->data(), grad_buf->data(),
                                    inv_ndim);
            graph::tensor::add_inplace(1.0, grad_buf->data(), grad_accumulate,
                                      grad_src->data());
        }
    }
}

NNGraph::TensorNode* transpose(
    NNGraph::TensorNode* src,
    const std::string& output_name,
    Index ndim)
{
    if(src == nullptr)
    {
        throw std::invalid_argument("transpose: src must be non-null");
    }
    NNGraph* graph = src->graph();
    auto op = std::make_shared<NNTransposeOp>(src, ndim);
    NNGraph::TensorNode* output = op->forward(output_name);
    graph->register_op(std::move(op));
    return output;
}

} // namespace nntile::graph
