#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/nn_graph/gemm.cc
 * NNGraph GEMM autograd implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/nn/ops/gemm.hh"

#include "nntile/nn/graph_data_node.hh"
#include "nntile/nn/nn_grad_slot_name.hh"
#include "nntile/nn/shape_layout.hh"
#include "nntile/tensor/ops/gemm.hh"

#include <stdexcept>

namespace nntile
{

namespace
{
constexpr Scalar gemm_new_output_beta = 0.0;
constexpr Scalar grad_overwrite = 0.0;
constexpr Scalar grad_accumulate = 1.0;
} // anonymous namespace

NNGraph::TensorNode *NNGemmOp::forward()
{
    if (x == nullptr || w == nullptr)
    {
        throw std::invalid_argument(
            "NNGemmOp::forward: x, w must be non-null");
    }
    NNGraph *graph = x->graph();
    bool out_requires_grad = any_input_requires_grad({x, w});
    TensorGraph::TensorNode *c_data = tensor::gemm(
        w->data(), x->data(), alpha, trans_w, trans_b, ndim, batch_ndim);
    NNGraph::TensorNode *c = graph->tensor(c_data, out_requires_grad);
    outputs_ = {c};
    return c;
}

void NNGemmOp::backward() const
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
    // Forward: tensor::gemm(w, x, trans_w, trans_b, ...).  Gradients below use
    // tensor-level formulas with (a=w, b=x).
    if (x != nullptr && x->requires_grad())
    {
        auto [grad_x, is_first] =
            graph->get_or_create_grad(x, nn_grad_slot_name(x));
        Scalar beta = is_first ? grad_overwrite : grad_accumulate;
        if (!trans_b)
        {
            tensor::gemm(w->data(),
                grad_out->data(),
                grad_x->data(),
                alpha,
                beta,
                !trans_w,
                false,
                w->ndim() - batch_ndim - ndim,
                batch_ndim);
        }
        else
        {
            tensor::gemm(grad_out->data(),
                w->data(),
                grad_x->data(),
                alpha,
                beta,
                true,
                trans_w,
                w->ndim() - batch_ndim - ndim,
                batch_ndim);
        }
    }
    if (w != nullptr && w->requires_grad())
    {
        auto [grad_w, is_first] =
            graph->get_or_create_grad(w, nn_grad_slot_name(w));
        Scalar beta = is_first ? grad_overwrite : grad_accumulate;
        if (!trans_w)
        {
            tensor::gemm(grad_out->data(),
                x->data(),
                grad_w->data(),
                alpha,
                beta,
                false,
                !trans_b,
                x->ndim() - batch_ndim - ndim,
                batch_ndim);
        }
        else
        {
            tensor::gemm(x->data(),
                grad_out->data(),
                grad_w->data(),
                alpha,
                beta,
                trans_b,
                true,
                x->ndim() - batch_ndim - ndim,
                batch_ndim);
        }
    }
}

NNGraph::TensorNode *gemm(NNGraph::TensorNode *x,
    NNGraph::TensorNode *w,
    Scalar alpha,
    bool trans_w,
    bool trans_b,
    Index ndim,
    Index batch_ndim)
{
    if (x == nullptr || w == nullptr)
    {
        throw std::invalid_argument("gemm: x and w must be non-null");
    }
    NNGraph *graph = x->graph();
    auto op = std::make_shared<NNGemmOp>(
        x, w, alpha, trans_w, trans_b, ndim, batch_ndim);
    NNGraph::TensorNode *c = op->forward();
    graph->register_op(std::move(op));
    return c;
}

} // namespace nntile
