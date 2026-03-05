/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/module/sdpa.cc
 * SDPA module implementation using NNTile graph API.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/module/sdpa.hh"

// Include NNTile headers
#include "nntile/graph/nn/clear.hh"
#include "nntile/graph/nn/fill.hh"
#include "nntile/graph/nn/gemm.hh"
#include "nntile/graph/nn/softmax.hh"
#include "nntile/graph/tensor/flash_sdpa_fwd_cudnn.hh"

// Include standard headers
#include <cmath>
#include <stdexcept>

// Include graph tensor operations
#include "nntile/graph/nn/clear.hh"
#include "nntile/graph/tensor/gemm.hh"
#include "nntile/graph/tensor/mask_scalar.hh"
#include "nntile/graph/tensor/maxsumexp.hh"
#include "nntile/graph/tensor/softmax_inplace.hh"

namespace nntile::module
{

//! Constructor: vanilla SDPA
Sdpa::Sdpa(
    graph::NNGraph* graph,
    const std::string& name,
    Index head_size,
    Index batch_ndim,
    graph::DataType dtype,
    int redux)
    : Module(graph, name)
    , flash_attention_(false)
    , head_size_(head_size)
    , batch_ndim_(batch_ndim)
    , scale_(1.0 / std::sqrt(static_cast<Scalar>(head_size)))
    , mask_val_(-std::numeric_limits<Scalar>::infinity())
    , redux_(redux)
{
    if(head_size <= 0)
    {
        throw std::invalid_argument(
            "Sdpa: head_size must be positive, got " +
            std::to_string(head_size));
    }
}

//! Constructor: Flash SDPA (placeholder - throws if flash_attention=true)
Sdpa::Sdpa(
    graph::NNGraph* graph,
    const std::string& name,
    Index head_size,
    Index batch_ndim,
    bool flash_attention,
    graph::DataType dtype,
    int redux)
    : Module(graph, name)
    , flash_attention_(flash_attention)
    , head_size_(head_size)
    , batch_ndim_(batch_ndim)
    , scale_(1.0 / std::sqrt(static_cast<Scalar>(head_size)))
    , mask_val_(-std::numeric_limits<Scalar>::infinity())
    , redux_(redux)
{
    if(!flash_attention)
    {
        throw std::invalid_argument(
            "Sdpa: this constructor is for Flash SDPA (flash_attention=true)");
    }
    if(head_size <= 0)
    {
        throw std::invalid_argument(
            "Sdpa: head_size must be positive, got " +
            std::to_string(head_size));
    }
    if(dtype != graph::DataType::FP16 && dtype != graph::DataType::BF16)
    {
        throw std::invalid_argument(
            "Sdpa: Flash SDPA requires FP16 or BF16 dtype");
    }
    // Flash attention is not implemented - throw exception
    throw std::runtime_error(
        "Sdpa: Flash attention (flash_attention=true) is not supported");
}

graph::NNGraph::TensorNode* Sdpa::forward(
    graph::NNGraph::TensorNode* q,
    graph::NNGraph::TensorNode* k,
    graph::NNGraph::TensorNode* v,
    graph::NNGraph::TensorNode* mask)
{
    if(flash_attention_)
    {
        throw std::runtime_error(
            "Sdpa::build_forward: Flash attention is not supported");
    }

    if(q == nullptr || k == nullptr || v == nullptr)
    {
        throw std::invalid_argument(
            "Sdpa::forward: Q, K, V must be non-null");
    }
  
    const auto& q_shape = q->shape();
    const auto& k_shape = k->shape();
    const auto& v_shape = v->shape();

    if(q_shape.size() != k_shape.size() || q_shape.size() != v_shape.size())
    {
        throw std::invalid_argument(
            "Sdpa::forward: Q, K, V must have same ndim");
    }
    if(q_shape[0] != head_size_)
    {
        throw std::invalid_argument(
            "Sdpa::forward: Q head_size mismatch");
    }
    if(k_shape[0] != head_size_ || v_shape[0] != head_size_)
    {
        throw std::invalid_argument(
            "Sdpa::forward: K and V head_size must match Q");
    }

    Index q_seq = q_shape[1];
    Index k_seq = k_shape[1];

    q_tensor_ = q;
    k_tensor_ = k;
    v_tensor_ = v;
    mask_tensor_ = mask;

    bool output_requires_grad = graph_->requires_grad(q) ||
        graph_->requires_grad(k) || graph_->requires_grad(v);

    // Vanilla SDPA: attn = scale * K^T @ Q, mask, maxsumexp, softmax, y = V @ attn
    std::vector<Index> batch_shape(
        q_shape.begin() + 2,
        q_shape.begin() + 2 + static_cast<ptrdiff_t>(batch_ndim_));

    std::vector<Index> attn_shape = {k_seq, q_seq};
    attn_shape.insert(attn_shape.end(), batch_shape.begin(), batch_shape.end());

    attn_tensor_ = graph_.tensor(
        attn_shape,
        tensor_name("attn"),
        q.dtype(),
        output_requires_grad);

    std::vector<Index> attn_max_shape = {2, q_seq};
    attn_max_shape.insert(
        attn_max_shape.end(), batch_shape.begin(), batch_shape.end());
    attn_maxsumexp_tensor_ = graph_.tensor(
        attn_max_shape,
        tensor_name("attn_maxsumexp"),
        q.dtype(),
        false);
    register_buffer("attn_maxsumexp", attn_maxsumexp_tensor_);

    std::vector<Index> attn_sum_shape = {q_seq};
    attn_sum_shape.insert(
        attn_sum_shape.end(), batch_shape.begin(), batch_shape.end());
    attn_sumprod_slice_tensor_ = graph_.tensor(
        attn_sum_shape,
        tensor_name("attn_sumprod_slice"),
        q.dtype(),
        false);
    register_buffer("attn_sumprod_slice", attn_sumprod_slice_tensor_);

    // Create output y
    std::vector<Index> y_shape = q_shape;
    output_tensor_ = graph_.tensor(
        y_shape,
        tensor_name("output"),
        q.dtype(),
        output_requires_grad);

    // attn = scale * K^T @ Q (ndim=1, batch_ndim)
    graph::tensor::gemm(
        k_tensor_->data(),
        q.data(),
        attn_tensor_->data(),
        scale_,
        0.0,
        true,
        false,
        1,
        batch_ndim_);

    // Clear attn_maxsumexp
    graph::clear(attn_maxsumexp_tensor_);

    // Optional mask: A[mask] = val for masked positions
    if(mask_tensor_ != nullptr)
    {
        graph::tensor::mask_scalar(
            mask_tensor_->data(),
            mask_val_,
            attn_tensor_->data(),
            batch_ndim_);
    }

    // maxsumexp along axis 0
    graph::tensor::maxsumexp(
        attn_tensor_->data(),
        attn_maxsumexp_tensor_->data(),
        0,
        redux_);

    // softmax_inplace: attn = softmax(attn) using maxsumexp
    graph::tensor::softmax_inplace(
        attn_maxsumexp_tensor_->data(),
        attn_tensor_->data(),
        1.0,
        0);

    // y = V @ attn
    graph::tensor::gemm(
        v_tensor_->data(),
        attn_tensor_->data(),
        output_tensor_->data(),
        1.0,
        0.0,
        false,
        false,
        1,
        batch_ndim_);

    return output_tensor_;
}

std::string Sdpa::repr() const
{
    std::string result = "Sdpa(head_size=" + std::to_string(head_size_) +
                         ", batch_ndim=" + std::to_string(batch_ndim_);
    if(flash_attention_)
    {
        result += ", flash=True";
    }
    result += ")";
    return result;
}

} // namespace nntile::module
