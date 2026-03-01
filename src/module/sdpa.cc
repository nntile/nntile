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

// Include standard headers
#include <cmath>
#include <stdexcept>

namespace nntile::module
{

//! Constructor: vanilla SDPA
Sdpa::Sdpa(
    graph::NNGraph& graph,
    const std::string& name,
    Index head_size,
    Index batch_ndim,
    graph::DataType dtype,
    int redux)
    : ModuleBase(graph, name)
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
    graph::NNGraph& graph,
    const std::string& name,
    Index head_size,
    Index batch_ndim,
    bool flash_attention,
    graph::DataType dtype,
    int redux)
    : ModuleBase(graph, name)
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

graph::NNGraph::TensorNode& Sdpa::build_forward(
    graph::NNGraph::TensorNode& q,
    graph::NNGraph::TensorNode& k,
    graph::NNGraph::TensorNode& v,
    graph::NNGraph::TensorNode* mask)
{
    const auto& q_shape = q.shape();
    const auto& k_shape = k.shape();
    const auto& v_shape = v.shape();

    if(q_shape.size() != k_shape.size() || q_shape.size() != v_shape.size())
    {
        throw std::invalid_argument(
            "Sdpa::build_forward: Q, K, V must have same ndim");
    }
    if(q_shape[0] != head_size_)
    {
        throw std::invalid_argument(
            "Sdpa::build_forward: Q head_size mismatch");
    }
    if(k_shape[0] != head_size_ || v_shape[0] != head_size_)
    {
        throw std::invalid_argument(
            "Sdpa::build_forward: K and V head_size must match Q");
    }

    Index q_seq = q_shape[1];
    Index k_seq = k_shape[1];

    q_tensor_ = &q;
    k_tensor_ = &k;
    v_tensor_ = &v;
    mask_tensor_ = mask;

    bool output_requires_grad = graph_.requires_grad(&q) ||
        graph_.requires_grad(&k) || graph_.requires_grad(&v);

    if(flash_attention_)
    {
        // Flash path: validate dtypes
        if(q.dtype() != graph::DataType::FP16 && q.dtype() != graph::DataType::BF16)
        {
            throw std::invalid_argument(
                "Sdpa::build_forward: Flash SDPA requires FP16 or BF16");
        }

        // Create flash_logsumexp [q_seq, batch...] fp32
        std::vector<Index> logsumexp_shape(q_shape.begin() + 1, q_shape.end());
        flash_logsumexp_tensor_ = graph_.tensor(
            logsumexp_shape,
            tensor_name("flash_logsumexp"),
            graph::DataType::FP32,
            false);
        register_buffer("flash_logsumexp", flash_logsumexp_tensor_);

        // Create output y [head_size, q_seq, batch...]
        std::vector<Index> y_shape = q_shape;
        output_tensor_ = graph_.tensor(
            y_shape,
            tensor_name("output"),
            q.dtype(),
            output_requires_grad);

        // Create default mask if needed: [q_seq, q_seq] or [k_seq, q_seq]
        graph::NNGraph::TensorNode* mask_to_use = mask;
        graph::NNGraph::TensorNode* default_mask = nullptr;
        if(mask_to_use == nullptr)
        {
            std::vector<Index> mask_shape = {k_seq, q_seq};
            default_mask = graph_.tensor(
                mask_shape,
                tensor_name("default_mask"),
                q.dtype(),
                false);
            register_buffer("default_mask", default_mask);
            mask_to_use = default_mask;
            graph_.add_op(
                graph::OpType::CLEAR,
                graph::OpAttrs{graph::ClearAttrs{}},
                {},
                {default_mask});
        }

        // Fill logsumexp with -inf, clear y
        graph_.add_op(
            graph::OpType::FILL,
            graph::OpAttrs{graph::FillAttrs{mask_val_}},
            {},
            {flash_logsumexp_tensor_});
        graph_.add_op(
            graph::OpType::CLEAR,
            graph::OpAttrs{graph::ClearAttrs{}},
            {},
            {output_tensor_});

        // Flash SDPA forward: K, Q, mask, logsumexp, V -> A (output)
        graph_.add_op(
            graph::OpType::FLASH_SDPA_FWD_CUDNN,
            graph::OpAttrs{graph::ClearAttrs{}},
            {k_tensor_, &q, mask_to_use, flash_logsumexp_tensor_, v_tensor_},
            {output_tensor_});
    }
    else
    {
        // Vanilla path: create buffers
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
        graph_.add_op(
            graph::OpType::GEMM,
            graph::OpAttrs{graph::GemmAttrs{
                true, false, scale_, 0.0, 1, batch_ndim_}},
            {k_tensor_, &q},
            {attn_tensor_});

        // Clear attn_maxsumexp
        graph_.add_op(
            graph::OpType::CLEAR,
            graph::OpAttrs{graph::ClearAttrs{}},
            {},
            {attn_maxsumexp_tensor_});

        // Optional mask
        if(mask_tensor_ != nullptr)
        {
            graph_.add_op(
                graph::OpType::MASK_SCALAR,
                graph::OpAttrs{graph::MaskScalarAttrs{mask_val_, batch_ndim_}},
                {mask_tensor_, attn_tensor_},
                {attn_tensor_});
        }

        // maxsumexp along axis 0
        graph_.add_op(
            graph::OpType::MAXSUMEXP,
            graph::OpAttrs{graph::LogSumExpAttrs{1.0, 0.0, 0}},
            {attn_tensor_},
            {attn_maxsumexp_tensor_});

        // softmax_inplace
        graph_.add_op(
            graph::OpType::SOFTMAX_INPLACE,
            graph::OpAttrs{graph::LogSumExpAttrs{1.0, 1.0, 0}},
            {attn_maxsumexp_tensor_, attn_tensor_},
            {attn_tensor_});

        // y = V @ attn
        graph_.add_op(
            graph::OpType::GEMM,
            graph::OpAttrs{graph::GemmAttrs{false, false, 1.0, 0.0, 1, batch_ndim_}},
            {v_tensor_, attn_tensor_},
            {output_tensor_});
    }

    return *output_tensor_;
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
