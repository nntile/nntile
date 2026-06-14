#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/nn_graph/sum_slice.cc
 * NNGraph sum_slice autograd implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/nn/ops/sum_slice.hh"

#include "nntile/nn/nn_grad_slot_name.hh"
#include "nntile/nn/shape_layout.hh"
#include "nntile/tensor/ops/add_slice_inplace.hh"
#include "nntile/tensor/ops/sum_slice.hh"

#include <stdexcept>

namespace nntile
{

namespace
{
constexpr Scalar grad_overwrite = 0.0;
constexpr Scalar grad_accumulate = 1.0;
constexpr Scalar beta_fresh = 0.0; // NNGraph always outputs fresh data
} // anonymous namespace

NNGraph::TensorNode *NNSumSliceOp::forward()
{
    if (x == nullptr)
    {
        throw std::invalid_argument(
            "NNSumSliceOp::forward: x must be non-null");
    }
    NNGraph *graph = x->graph();
    bool out_requires_grad = any_input_requires_grad({x});
    const Index f_axis = nn::c_axis_to_fortran(axis, x->ndim());
    TensorGraph::TensorNode *y_data =
        tensor::sum_slice(x->data(), f_axis, redux, alpha, beta_fresh);
    NNGraph::TensorNode *y = graph->tensor(y_data, out_requires_grad);
    outputs_ = {y};
    return y;
}

void NNSumSliceOp::backward() const
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
        Scalar grad_beta = is_first ? grad_overwrite : grad_accumulate;
        const Index f_axis = nn::c_axis_to_fortran(axis, x->ndim());
        tensor::add_slice_inplace(
            alpha, grad_out->data(), grad_beta, grad_x->data(), f_axis);
    }
}

NNGraph::TensorNode *sum_slice(
    NNGraph::TensorNode *x, Index axis, int redux, Scalar alpha)
{
    if (x == nullptr)
    {
        throw std::invalid_argument("sum_slice: x must be non-null");
    }
    NNGraph *graph = x->graph();
    auto op = std::make_shared<NNSumSliceOp>(x, axis, redux, alpha);
    NNGraph::TensorNode *y = op->forward();
    graph->register_op(std::move(op));
    return y;
}

} // namespace nntile
