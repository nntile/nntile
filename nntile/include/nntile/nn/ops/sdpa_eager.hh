/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/nn/ops/sdpa_eager.hh
 * NNGraph SDPA eager autograd operation.
 *
 * Forward: attn = softmax(scale * K^T @ Q + mask), out = V @ attn
 * Backward: grad_Q, grad_K, grad_V via chain rule.
 *
 * Layout: [batch..., seq, head_size]. Scale = 1/sqrt(head_size).
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/nn/graph_op_node.hh>

namespace nntile
{

//! SDPA eager op: out = V @ softmax(scale * K^T @ Q + mask)
struct NNSdpaEagerOp : NNGraph::OpNode
{
    Scalar scale;
    Index batch_ndim;
    int redux;
    NNGraph::TensorNode *q = nullptr;
    NNGraph::TensorNode *k = nullptr;
    NNGraph::TensorNode *v = nullptr;
    NNGraph::TensorNode *mask = nullptr;

    NNSdpaEagerOp() = default;
    NNSdpaEagerOp(NNGraph::TensorNode *q_,
        NNGraph::TensorNode *k_,
        NNGraph::TensorNode *v_,
        Scalar scale_,
        Index batch_ndim_,
        int redux_,
        NNGraph::TensorNode *mask_ = nullptr) :
        scale(scale_),
        batch_ndim(batch_ndim_),
        redux(redux_),
        q(q_),
        k(k_),
        v(v_),
        mask(mask_)
    {
        inputs_ = {q, k, v};
        if (mask)
            inputs_.push_back(mask);
    }

    NNGraph::TensorNode *forward();
    void backward() const override;
};

//! SDPA eager: out = V @ softmax(scale * K^T @ Q + mask)
//! @param q Query [batch..., q_seq, head_size]
//! @param k Key [batch..., k_seq, head_size]
//! @param v Value [batch..., k_seq, head_size]
//! @param output_name Name for output tensor
//! @param mask Optional boolean mask [q_seq, k_seq] (nullptr = no mask)
//! @param batch_ndim Number of leading batch dimensions
//! @param redux Reduction mode for distributed training
//! @return Output [batch..., q_seq, head_size]
NNGraph::TensorNode *sdpa_eager(NNGraph::TensorNode *q,
    NNGraph::TensorNode *k,
    NNGraph::TensorNode *v,
    NNGraph::TensorNode *mask = nullptr,
    Index batch_ndim = 2,
    int redux = 0);

} // namespace nntile
