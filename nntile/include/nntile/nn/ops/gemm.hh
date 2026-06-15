/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/nn/ops/gemm.hh
 * NNGraph GEMM autograd operation (virtual C-order API).
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

//! GEMM op: PyTorch-style ``y = x @ w.T`` on virtual C-order shapes.
//!
//! ``x`` is the activation (``[..., k]`` trailing contraction axes),
//! ``w`` is the weight (``[m..., k]`` or ``[k, m...]`` depending on layout;
//! use ``trans_w`` when the contraction axis is leading in ``w``).
//! Lowers to ``tensor::gemm(w, x, trans_w, trans_b, ndim, batch_ndim)``.
struct NNGemmOp : NNGraph::OpNode
{
    Scalar alpha;
    bool trans_w;
    bool trans_b;
    Index ndim;
    Index batch_ndim;
    NNGraph::TensorNode *x = nullptr;
    NNGraph::TensorNode *w = nullptr;

    NNGemmOp(NNGraph::TensorNode *x_,
        NNGraph::TensorNode *w_,
        Scalar alpha_,
        bool trans_w_,
        bool trans_b_,
        Index ndim_,
        Index batch_ndim_) :
        alpha(alpha_),
        trans_w(trans_w_),
        trans_b(trans_b_),
        ndim(ndim_),
        batch_ndim(batch_ndim_),
        x(x_),
        w(w_)
    {
        inputs_ = {x, w};
    }

    NNGraph::TensorNode *forward();
    void backward() const override;
};

//! ``gemm(x, w)`` with virtual C-order shapes: ``y = alpha * x @ op(w).T``
//! (PyTorch ``linear`` semantics when ``w`` is ``[out, in]`` and
//! ``trans_w=true``).
NNGraph::TensorNode *gemm(NNGraph::TensorNode *x,
    NNGraph::TensorNode *w,
    Scalar alpha,
    bool trans_w,
    bool trans_b,
    Index ndim,
    Index batch_ndim);

} // namespace nntile
