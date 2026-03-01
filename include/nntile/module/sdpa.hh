/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/module/sdpa.hh
 * Scaled dot-product attention (SDPA) module using NNTile graph API.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// Include NNTile headers
#include <nntile/graph.hh>
#include <nntile/module/module.hh>

namespace nntile::module
{

//! Scaled dot-product attention module using graph API
//!
//! Supports two implementations controlled by flash_attention flag:
//! - Vanilla: GEMM + maxsumexp + softmax + GEMM (any dtype)
//! - Flash: cuDNN FlashAttention (FP16/BF16 only, CUDA)
//!
//! Tensor layout: [head_size, n_seq, batch_dims...]
//! - Q: [head_size, q_seq, batch...]
//! - K: [head_size, k_seq, batch...]
//! - V: [head_size, k_seq, batch...]
//! - Y: [head_size, q_seq, batch...]
class Sdpa : public Module<Sdpa>
{
private:
    bool flash_attention_;
    Index head_size_;
    Index batch_ndim_;
    Scalar scale_;
    Scalar mask_val_;
    int redux_;

    // Input tensors from last build_forward
    graph::NNGraph::TensorNode* q_tensor_ = nullptr;
    graph::NNGraph::TensorNode* k_tensor_ = nullptr;
    graph::NNGraph::TensorNode* v_tensor_ = nullptr;
    graph::NNGraph::TensorNode* mask_tensor_ = nullptr;

    // Vanilla buffers (created when flash_attention=false)
    graph::NNGraph::TensorNode* attn_tensor_ = nullptr;
    graph::NNGraph::TensorNode* attn_maxsumexp_tensor_ = nullptr;
    graph::NNGraph::TensorNode* attn_sumprod_slice_tensor_ = nullptr;

    // Flash buffer (created when flash_attention=true)
    graph::NNGraph::TensorNode* flash_logsumexp_tensor_ = nullptr;

    graph::NNGraph::TensorNode* output_tensor_ = nullptr;

public:
    static constexpr bool has_custom_backward = false;

    //! Constructor: vanilla SDPA (creates attn, attn_maxsumexp, attn_sumprod_slice)
    //! @param graph The neural network graph
    //! @param name Module name
    //! @param head_size Head dimension (scale = 1/sqrt(head_size))
    //! @param batch_ndim Number of trailing batch dimensions
    //! @param dtype Data type for tensors
    //! @param redux Reduction mode for distributed training (default: 0)
    Sdpa(
        graph::NNGraph& graph,
        const std::string& name,
        Index head_size,
        Index batch_ndim,
        graph::DataType dtype = graph::DataType::FP32,
        int redux = 0
    );

    //! Constructor: Flash SDPA (creates flash_logsumexp, requires FP16/BF16)
    //! @param graph The neural network graph
    //! @param name Module name
    //! @param head_size Head dimension
    //! @param batch_ndim Number of trailing batch dimensions
    //! @param flash_attention Must be true
    //! @param dtype Must be FP16 or BF16
    //! @param redux Reduction mode (default: 0)
    Sdpa(
        graph::NNGraph& graph,
        const std::string& name,
        Index head_size,
        Index batch_ndim,
        bool flash_attention,
        graph::DataType dtype,
        int redux = 0
    );

    //! Build forward operations for SDPA
    //! @param q Query tensor [head_size, q_seq, batch...]
    //! @param k Key tensor [head_size, k_seq, batch...]
    //! @param v Value tensor [head_size, k_seq, batch...]
    //! @param mask Optional boolean mask [k_seq, q_seq] (nullptr = no mask)
    //! @return Output tensor [head_size, q_seq, batch...]
    graph::NNGraph::TensorNode& build_forward(
        graph::NNGraph::TensorNode& q,
        graph::NNGraph::TensorNode& k,
        graph::NNGraph::TensorNode& v,
        graph::NNGraph::TensorNode* mask = nullptr
    );

    //! Build backward operations
    void build_backward();

    //! Get string representation
    std::string repr() const override;

    // Accessors
    bool flash_attention() const { return flash_attention_; }
    Index head_size() const { return head_size_; }
    Index batch_ndim() const { return batch_ndim_; }
    Scalar scale() const { return scale_; }
};

} // namespace nntile::module
