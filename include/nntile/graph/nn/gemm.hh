/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/gemm.hh
 * NNGraph GEMM autograd operation.
 *
 * Forward: C = alpha * op(A) @ op(B)
 * Backward: grad_A = alpha * grad_C @ B^T, grad_B = alpha * A^T @ grad_C
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/graph/nn/graph_op_node.hh>
#include <nntile/graph/tensor/gemm.hh>

namespace nntile::graph
{

//! GEMM op: C = alpha * op(A) @ op(B). PyTorch-style: outputs created in forward().
struct NNGemmOp : NNGraph::OpNode
{
    Scalar alpha;
    bool trans_a;
    bool trans_b;
    Index ndim;
    Index batch_ndim;
    NNGraph::TensorNode* a = nullptr;
    NNGraph::TensorNode* b = nullptr;

    NNGemmOp(NNGraph::TensorNode* a_,
            NNGraph::TensorNode* b_,
            Scalar alpha_,
            bool trans_a_,
            bool trans_b_,
            Index ndim_,
            Index batch_ndim_)
        : alpha(alpha_), trans_a(trans_a_), trans_b(trans_b_)
        , ndim(ndim_), batch_ndim(batch_ndim_)
        , a(a_), b(b_)
    {
        inputs_ = {a, b};
    }

    NNGraph::TensorNode* forward(const std::string& output_name);
    void backward() const override;
};

NNGraph::TensorNode* gemm(
    NNGraph::TensorNode* a,
    NNGraph::TensorNode* b,
    const std::string& output_name,
    Scalar alpha,
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim);

} // namespace nntile::graph
