/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn/sdpa_eager.cc
 * NNGraph SDPA eager autograd implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn/sdpa_eager.hh"
#include "nntile/graph/nn/clear.hh"
#include "nntile/graph/nn/graph_data_node.hh"

#include <cmath>
#include <stdexcept>

#include "nntile/graph/tensor/add_inplace.hh"
#include "nntile/graph/tensor/add_slice.hh"
#include "nntile/graph/tensor/gemm.hh"
#include "nntile/graph/tensor/mask_scalar.hh"
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
constexpr Scalar mask_val = -std::numeric_limits<Scalar>::infinity();
} // anonymous namespace

NNGraph::TensorNode* NNSdpaEagerOp::forward(const std::string& output_name)
{
    if(q == nullptr || k == nullptr || v == nullptr)
    {
        throw std::invalid_argument(
            "NNSdpaEagerOp::forward: Q, K, V must be non-null");
    }
    NNGraph* graph = q->graph();
    bool out_requires_grad = any_input_requires_grad({q, k, v});

    const auto& q_shape = q->shape();
    const auto& k_shape = k->shape();
    Index q_seq = q_shape[1];
    Index k_seq = k_shape[1];

    std::vector<Index> batch_shape(
        q_shape.begin() + 2,
        q_shape.begin() + 2 + static_cast<ptrdiff_t>(batch_ndim));

    std::vector<Index> attn_shape = {k_seq, q_seq};
    attn_shape.insert(attn_shape.end(), batch_shape.begin(), batch_shape.end());

    std::string attn_name = output_name + "_attn";
    NNGraph::TensorNode* attn = graph->tensor(
        attn_shape, attn_name, q->dtype(), out_requires_grad);
    graph::tensor::gemm(
        k->data(), q->data(), attn->data(),
        scale, 0.0, true, false, 1, batch_ndim);

    if(mask != nullptr)
    {
        graph::tensor::mask_scalar(
            mask->data(), mask_val, attn->data(), batch_ndim);
    }

    std::vector<Index> attn_max_shape = {2, q_seq};
    attn_max_shape.insert(
        attn_max_shape.end(), batch_shape.begin(), batch_shape.end());
    std::string mse_name = output_name + "_mse";
    NNGraph::TensorNode* maxsumexp_buf = graph->tensor(
        attn_max_shape, mse_name, q->dtype(), false);
    graph::clear(maxsumexp_buf);
    graph::tensor::maxsumexp(
        attn->data(), maxsumexp_buf->data(), 0, redux);
    graph::tensor::softmax_inplace(
        maxsumexp_buf->data(), attn->data(), 1.0, 0);

    std::vector<Index> sumprod_shape = {q_seq};
    sumprod_shape.insert(
        sumprod_shape.end(), batch_shape.begin(), batch_shape.end());
    NNGraph::TensorNode* sumprod_buf = graph->tensor(
        sumprod_shape, output_name + "_sps", q->dtype(), false);
    NNGraph::TensorNode* grad_temp = graph->tensor(
        attn_shape, output_name + "_gt", q->dtype(), false);
    buffers_ = {attn, sumprod_buf, grad_temp};

    std::vector<Index> y_shape = q_shape;
    NNGraph::TensorNode* out = graph->tensor(
        y_shape, output_name, q->dtype(), out_requires_grad);
    graph::tensor::gemm(
        v->data(), attn->data(), out->data(),
        1.0, 0.0, false, false, 1, batch_ndim);
    outputs_ = {out};

    return out;
}

void NNSdpaEagerOp::backward() const
{
    NNGraph::TensorNode* out = output();
    if(out == nullptr)
        return;
    NNGraph* graph = out->graph();
    NNGraph::TensorNode* grad_out = out->grad();
    if(grad_out == nullptr)
        return;

    if(buffers_.size() < 3)
    {
        throw std::runtime_error(
            "NNSdpaEagerOp::backward: buffers are missing");
    }
    NNGraph::TensorNode* attn = buffers_[0];
    NNGraph::TensorNode* sumprod_buf = buffers_[1];
    NNGraph::TensorNode* grad_temp = buffers_[2];

    Index ndim_contraction = 1;
    Index q_ndim = static_cast<Index>(q->shape().size());

    if(v != nullptr && v->requires_grad())
    {
        auto [grad_v, is_first] =
            graph->get_or_create_grad(v, v->name() + "_grad");
        Scalar beta = is_first ? grad_overwrite : grad_accumulate;
        graph::tensor::gemm(
            grad_out->data(),
            attn->data(),
            grad_v->data(),
            1.0, beta,
            false, true,
            q_ndim - batch_ndim - ndim_contraction,
            batch_ndim);
    }

    TensorGraph::TensorNode* d_attn_data = grad_temp->data();
    graph::tensor::gemm(
        v->data(),
        grad_out->data(),
        d_attn_data,
        1.0, 0.0,
        true, false,
        q_ndim - batch_ndim - ndim_contraction,
        batch_ndim);

    graph::tensor::sumprod_slice(
        attn->data(), d_attn_data, sumprod_buf->data(),
        0, redux, 1.0, 0.0);
    graph::tensor::add_slice(
        -1.0, sumprod_buf->data(), 1.0, d_attn_data,
        grad_temp->data(), 0);
    graph::tensor::multiply_inplace(1.0, attn->data(), grad_temp->data());

    if(q != nullptr && q->requires_grad())
    {
        auto [grad_q, is_first] =
            graph->get_or_create_grad(q, q->name() + "_grad");
        Scalar beta = is_first ? grad_overwrite : grad_accumulate;
        graph::tensor::gemm(
            k->data(),
            grad_temp->data(),
            grad_q->data(),
            scale, beta,
            false, false,
            q_ndim - batch_ndim - ndim_contraction,
            batch_ndim);
    }

    if(k != nullptr && k->requires_grad())
    {
        auto [grad_k, is_first] =
            graph->get_or_create_grad(k, k->name() + "_grad");
        Scalar beta = is_first ? grad_overwrite : grad_accumulate;
        graph::tensor::gemm(
            q->data(),
            grad_temp->data(),
            grad_k->data(),
            scale, beta,
            false, true,
            q_ndim - batch_ndim - ndim_contraction,
            batch_ndim);
    }
}

NNGraph::TensorNode* sdpa_eager(
    NNGraph::TensorNode* q,
    NNGraph::TensorNode* k,
    NNGraph::TensorNode* v,
    const std::string& output_name,
    NNGraph::TensorNode* mask,
    Index batch_ndim,
    int redux)
{
    if(q == nullptr || k == nullptr || v == nullptr)
    {
        throw std::invalid_argument(
            "sdpa_eager: Q, K, V must be non-null");
    }
    const auto& q_shape = q->shape();
    const auto& k_shape = k->shape();
    const auto& v_shape = v->shape();

    if(q_shape.size() != k_shape.size() || q_shape.size() != v_shape.size())
    {
        throw std::invalid_argument(
            "sdpa_eager: Q, K, V must have same ndim");
    }
    if(q_shape[0] != k_shape[0] || q_shape[0] != v_shape[0])
    {
        throw std::invalid_argument(
            "sdpa_eager: Q, K, V head_size must match");
    }
    if(k_shape[1] != v_shape[1])
    {
        throw std::invalid_argument(
            "sdpa_eager: K and V seq length must match");
    }
    Index head_size = q_shape[0];
    if(head_size <= 0)
    {
        throw std::invalid_argument(
            "sdpa_eager: head_size must be positive");
    }

    Scalar scale = 1.0 / std::sqrt(static_cast<Scalar>(head_size));
    NNGraph* graph = q->graph();
    auto op = std::make_shared<NNSdpaEagerOp>(
        q, k, v, scale, batch_ndim, redux, mask);
    NNGraph::TensorNode* out = op->forward(output_name);
    graph->register_op(std::move(op));
    return out;
}

} // namespace nntile::graph
