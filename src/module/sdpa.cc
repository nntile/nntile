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

//! Constructor: Flash SDPA
Sdpa::Sdpa(
    graph::NNGraph* graph,
    const std::string& name,
    Index head_size,
    Index batch_ndim,
    bool flash_attention,
    graph::DataType dtype,
    int redux)
    : Module(graph, name)
    , flash_attention_(true)
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
}

graph::NNGraph::TensorNode* Sdpa::forward(
    graph::NNGraph::TensorNode* q,
    graph::NNGraph::TensorNode* k,
    graph::NNGraph::TensorNode* v,
    graph::NNGraph::TensorNode* mask)
{
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

    if(flash_attention_)
    {
        // Flash path: validate dtypes
        if(q->dtype() != graph::DataType::FP16 && q->dtype() != graph::DataType::BF16)
        {
            throw std::invalid_argument(
                "Sdpa::forward: Flash SDPA requires FP16 or BF16");
        }

        // Create flash_logsumexp [q_seq, batch...] fp32
        std::vector<Index> logsumexp_shape(q_shape.begin() + 1, q_shape.end());
        flash_logsumexp_tensor_ = graph_->tensor(
            logsumexp_shape,
            tensor_name("flash_logsumexp"),
            graph::DataType::FP32,
            false);
        register_buffer("flash_logsumexp", flash_logsumexp_tensor_);

        // Create output y [head_size, q_seq, batch...]
        std::vector<Index> y_shape = q_shape;
        output_tensor_ = graph_->tensor(
            y_shape,
            tensor_name("output"),
            q->dtype(),
            output_requires_grad);

        // Create default mask if needed: [q_seq, q_seq] or [k_seq, q_seq]
        graph::NNGraph::TensorNode* mask_to_use = mask;
        graph::NNGraph::TensorNode* default_mask = nullptr;
        if(mask_to_use == nullptr)
        {
            std::vector<Index> mask_shape = {k_seq, q_seq};
            default_mask = graph_->tensor(
                mask_shape,
                tensor_name("default_mask"),
                q->dtype(),
                false);
            register_buffer("default_mask", default_mask);
            mask_to_use = default_mask;
            graph::clear(default_mask);
        }

        // Fill logsumexp with -inf, clear y
        graph::fill(mask_val_, flash_logsumexp_tensor_);
        graph::clear(output_tensor_);

        // Flash SDPA forward: K, Q, mask, logsumexp, V -> A (output)
        graph::tensor::flash_sdpa_fwd_cudnn(
            k_tensor_->data(),
            q->data(),
            mask_to_use->data(),
            flash_logsumexp_tensor_->data(),
            v_tensor_->data(),
            output_tensor_->data());
    }
    else
    {
        // Vanilla path: attn = scale * K^T @ Q, softmax(attn), y = V @ attn
        attn_tensor_ = graph::gemm(
            k_tensor_,
            q,
            tensor_name("attn"),
            scale_,
            true,   // trans_a (K^T)
            false,  // trans_b
            1,
            batch_ndim_);

        // Optional mask (TODO: implement mask application)
        if(mask_tensor_ != nullptr)
        {
            (void)mask_tensor_;  // unused until mask is implemented
        }

        // Softmax along axis 0 (over k_seq dimension)
        graph::NNGraph::TensorNode* softmax_out = graph::softmax(
            attn_tensor_,
            tensor_name("attn_softmax"),
            0,
            redux_);
        attn_tensor_ = softmax_out;

        // y = V @ attn
        output_tensor_ = graph::gemm(
            v_tensor_,
            attn_tensor_,
            tensor_name("output"),
            1.0,
            false,  // trans_a
            false,  // trans_b
            1,
            batch_ndim_);
    }

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
