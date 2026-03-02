/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn_graph/gemm.hh
 * NNGraph GEMM autograd operation.
 *
 * Forward: C = alpha * op(A) @ op(B)
 * Backward: grad_A = alpha * grad_C @ B^T, grad_B = alpha * A^T @ grad_C
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/graph/tensor/gemm.hh>
#include <nntile/graph/nn_graph.hh>

namespace nntile::graph
{

//! GEMM op: C = alpha * op(A) @ op(B). Self-contained: holds params and tensors.
struct NNGemmOp : NNBaseOpNode
{
    Scalar alpha = 1.0;
    bool trans_a = false;
    bool trans_b = false;
    Index ndim = 1;
    Index batch_ndim = 0;
    NNGraph::TensorNode* a = nullptr;
    NNGraph::TensorNode* b = nullptr;
    NNGraph::TensorNode* c = nullptr;

    NNGemmOp() = default;
    NNGemmOp(NNGraph::TensorNode* a_,
            NNGraph::TensorNode* b_,
            NNGraph::TensorNode* c_,
            Scalar alpha_ = 1.0,
            bool trans_a_ = false,
            bool trans_b_ = false,
            Index ndim_ = 1,
            Index batch_ndim_ = 0)
        : alpha(alpha_), trans_a(trans_a_), trans_b(trans_b_)
        , ndim(ndim_), batch_ndim(batch_ndim_)
        , a(a_), b(b_), c(c_)
    {
        inputs_ = {a, b};
        outputs_ = {c};
    }

    void add_forward_to_tensor_graph(NNGraph& graph) override;
    void backward() override;
};

NNGraph::TensorNode* gemm(
    NNGraph::TensorNode* a,
    NNGraph::TensorNode* b,
    const std::string& output_name,
    Scalar alpha = 1.0,
    bool trans_a = false,
    bool trans_b = false,
    Index ndim = 1,
    Index batch_ndim = 0);

} // namespace nntile::graph
