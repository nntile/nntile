/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/nn/ops/gemm.hh
 * NNGraph GEMM autograd operation (graph API).
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>

// NNTile headers
#include <nntile/nn/graph_op_node.hh>
#include <nntile/tensor/ops/gemm.hh>

namespace nntile
{

//! Generic GEMM on graph shapes.
//!
//! ``trans_a`` / ``trans_b`` transpose the first ``ndim`` axes of operands
//! ``a`` / ``b``.  Lowers to ``tensor::gemm(b, a, trans_b, trans_a, ndim,
//! batch_ndim)`` (operands and transpose flags swapped for graph labels).
struct NNGemmOp : NNGraph::OpNode
{
    Scalar alpha;
    bool trans_a;
    bool trans_b;
    Index ndim;
    Index batch_ndim;
    NNGraph::TensorNode *a = nullptr;
    NNGraph::TensorNode *b = nullptr;

    NNGemmOp(NNGraph::TensorNode *a_,
        NNGraph::TensorNode *b_,
        Scalar alpha_,
        bool trans_a_,
        bool trans_b_,
        Index ndim_,
        Index batch_ndim_) :
        alpha(alpha_),
        trans_a(trans_a_),
        trans_b(trans_b_),
        ndim(ndim_),
        batch_ndim(batch_ndim_),
        a(a_),
        b(b_)
    {
        inputs_ = {a, b};
    }

    NNGraph::TensorNode *forward();
    void backward() const override;
};

//! Generic GEMM: ``y = alpha * op(a) @ op(b)`` on graph shapes.
NNGraph::TensorNode *gemm(NNGraph::TensorNode *a,
    NNGraph::TensorNode *b,
    Scalar alpha,
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim);

} // namespace nntile
