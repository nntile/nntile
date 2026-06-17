#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/nn_graph/sdpa_eager.cc
 * NNGraph SDPA eager autograd implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/nn/ops/sdpa_eager.hh"

#include "nntile/nn/graph_data_node.hh"
#include "nntile/nn/nn_grad_slot_name.hh"
#include "nntile/nn/ops/clear.hh"
#include "nntile/tensor/ops/add_inplace.hh"
#include "nntile/tensor/ops/add_slice.hh"
#include "nntile/tensor/ops/add_slice_inplace.hh"
#include "nntile/tensor/ops/gemm.hh"
#include "nntile/tensor/ops/mask_scalar.hh"
#include "nntile/tensor/ops/maxsumexp.hh"
#include "nntile/tensor/ops/multiply_inplace.hh"
#include "nntile/tensor/ops/softmax_inplace.hh"
#include "nntile/tensor/ops/sumprod_slice.hh"

#include <cmath>
#include <stdexcept>

namespace nntile
{

namespace
{
constexpr Scalar grad_overwrite = 0.0;
constexpr Scalar grad_accumulate = 1.0;
constexpr Scalar mask_val = -std::numeric_limits<Scalar>::infinity();
} // anonymous namespace

NNGraph::TensorNode *NNSdpaEagerOp::forward()
{
    if (q == nullptr || k == nullptr || v == nullptr)
    {
        throw std::invalid_argument(
            "NNSdpaEagerOp::forward: Q, K, V must be non-null");
    }
    NNGraph *graph = q->graph();
    bool out_requires_grad = any_input_requires_grad({q, k, v});

    const auto &q_shape = q->shape();
    const auto &k_shape = k->shape();
    const Index q_ndim = static_cast<Index>(q_shape.size());
    Index q_seq = q_shape[q_ndim - 2];
    Index k_seq = k_shape[q_ndim - 2];

    std::vector<Index> batch_shape(q_shape.begin(),
        q_shape.begin() + static_cast<ptrdiff_t>(batch_ndim));

    std::vector<Index> attn_shape = batch_shape;
    attn_shape.push_back(q_seq);
    attn_shape.push_back(k_seq);

    NNGraph::TensorNode *attn =
        graph->tensor(attn_shape, q->dtype(), out_requires_grad);
    tensor::gemm(q->data(),
        k->data(),
        attn->data(),
        scale,
        0.0,
        false,
        true,
        1,
        batch_ndim);

    if (mask != nullptr)
    {
        tensor::mask_scalar(
            mask->data(), mask_val, attn->data(), batch_ndim);
    }

    std::vector<Index> attn_max_shape = {2};
    attn_max_shape.insert(
        attn_max_shape.end(), batch_shape.begin(), batch_shape.end());
    attn_max_shape.push_back(q_seq);
    NNGraph::TensorNode *maxsumexp_buf =
        graph->tensor(attn_max_shape, q->dtype(), false);
    clear(maxsumexp_buf);
    const Index attn_axis = q_ndim - 1;
    tensor::maxsumexp(attn->data(), maxsumexp_buf->data(), attn_axis, redux);
    tensor::softmax_inplace(
        maxsumexp_buf->data(), attn->data(), 1.0, attn_axis);

    std::vector<Index> sumprod_shape = batch_shape;
    sumprod_shape.push_back(q_seq);
    NNGraph::TensorNode *sumprod_buf =
        graph->tensor(sumprod_shape, q->dtype(), false);
    NNGraph::TensorNode *grad_temp =
        graph->tensor(attn_shape, q->dtype(), false);
    buffers_ = {attn, sumprod_buf, grad_temp};

    std::vector<Index> y_shape = q_shape;
    NNGraph::TensorNode *out =
        graph->tensor(y_shape, q->dtype(), out_requires_grad);
    tensor::gemm(attn->data(),
        v->data(),
        out->data(),
        1.0,
        0.0,
        false,
        false,
        1,
        batch_ndim);
    outputs_ = {out};

    return out;
}

void NNSdpaEagerOp::backward() const
{
    NNGraph::TensorNode *out = output();
    if (out == nullptr)
        return;
    NNGraph *graph = out->graph();
    NNGraph::TensorNode *grad_out = out->grad();
    if (grad_out == nullptr)
        return;

    if (buffers_.size() < 3)
    {
        throw std::runtime_error(
            "NNSdpaEagerOp::backward: buffers are missing");
    }
    NNGraph::TensorNode *attn = buffers_[0];
    NNGraph::TensorNode *sumprod_buf = buffers_[1];
    NNGraph::TensorNode *grad_temp = buffers_[2];

    Index ndim_contraction = 1;
    Index q_ndim = static_cast<Index>(q->shape().size());

    if (v != nullptr && v->requires_grad())
    {
        auto [grad_v, is_first] =
            graph->get_or_create_grad(v, nn_grad_slot_name(v));
        Scalar beta = is_first ? grad_overwrite : grad_accumulate;
        tensor::gemm(attn->data(),
            grad_out->data(),
            grad_v->data(),
            1.0,
            beta,
            true,
            false,
            ndim_contraction,
            batch_ndim);
    }

    // d_attn = grad_out @ V^T, stored in grad_temp buffer
    tensor::gemm(grad_out->data(),
        v->data(),
        grad_temp->data(),
        1.0,
        0.0,
        false,
        true,
        ndim_contraction,
        batch_ndim);

    // grad_temp = (grad_temp - sumprod(attn, grad_temp)) * attn
    const Index attn_axis = q_ndim - 1;
    tensor::sumprod_slice(attn->data(),
        grad_temp->data(),
        sumprod_buf->data(),
        attn_axis,
        redux,
        1.0,
        0.0);
    tensor::add_slice_inplace(
        -1.0, sumprod_buf->data(), 1.0, grad_temp->data(), attn_axis);
    tensor::multiply_inplace(1.0, attn->data(), grad_temp->data());

    if (q != nullptr && q->requires_grad())
    {
        auto [grad_q, is_first] =
            graph->get_or_create_grad(q, nn_grad_slot_name(q));
        Scalar beta = is_first ? grad_overwrite : grad_accumulate;
        tensor::gemm(grad_temp->data(),
            k->data(),
            grad_q->data(),
            scale,
            beta,
            false,
            false,
            ndim_contraction,
            batch_ndim);
    }

    if (k != nullptr && k->requires_grad())
    {
        auto [grad_k, is_first] =
            graph->get_or_create_grad(k, nn_grad_slot_name(k));
        Scalar beta = is_first ? grad_overwrite : grad_accumulate;
        tensor::gemm(grad_temp->data(),
            q->data(),
            grad_k->data(),
            scale,
            beta,
            true,
            false,
            ndim_contraction,
            batch_ndim);
    }
}

NNGraph::TensorNode *sdpa_eager(NNGraph::TensorNode *q,
    NNGraph::TensorNode *k,
    NNGraph::TensorNode *v,
    NNGraph::TensorNode *mask,
    Index batch_ndim,
    int redux)
{
    if (q == nullptr || k == nullptr || v == nullptr)
    {
        throw std::invalid_argument("sdpa_eager: Q, K, V must be non-null");
    }
    const auto &q_shape = q->shape();
    const auto &k_shape = k->shape();
    const auto &v_shape = v->shape();

    if (q_shape.size() != k_shape.size() || q_shape.size() != v_shape.size())
    {
        throw std::invalid_argument("sdpa_eager: Q, K, V must have same ndim");
    }
    const Index q_ndim = static_cast<Index>(q_shape.size());
    if (q_shape[q_ndim - 1] != k_shape[q_ndim - 1] ||
        q_shape[q_ndim - 1] != v_shape[q_ndim - 1])
    {
        throw std::invalid_argument(
            "sdpa_eager: Q, K, V head_size must match");
    }
    if (k_shape[q_ndim - 2] != v_shape[q_ndim - 2])
    {
        throw std::invalid_argument(
            "sdpa_eager: K and V seq length must match");
    }
    Index head_size = q_shape[q_ndim - 1];
    if (head_size <= 0)
    {
        throw std::invalid_argument("sdpa_eager: head_size must be positive");
    }

    Scalar scale = 1.0 / std::sqrt(static_cast<Scalar>(head_size));
    NNGraph *graph = q->graph();
    auto op = std::make_shared<NNSdpaEagerOp>(
        q, k, v, scale, batch_ndim, redux, mask);
    NNGraph::TensorNode *out = op->forward();
    graph->register_op(std::move(op));
    return out;
}

} // namespace nntile
