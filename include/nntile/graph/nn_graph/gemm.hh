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
 * Backward: grad_A = alpha * grad_C @ B^T, grad_B = alpha * A^T @ grad_C (2 gemms)
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/graph/autograd_function.hh>
#include <nntile/graph/logical/gemm.hh>
#include <nntile/graph/nn_graph.hh>

namespace nntile::graph
{

//! Gemm functor: operator() forwards to build_forward
struct Gemm : AutogradFunction<Gemm>
{
    //! Forward: C = alpha * op(A) @ op(B), creates new output
    static NNGraph::TensorNode* build_forward(
        NNGraph::TensorNode* a,
        NNGraph::TensorNode* b,
        const std::string& output_name,
        Scalar alpha = 1.0,
        bool trans_a = false,
        bool trans_b = false,
        Index ndim = 1,
        Index batch_ndim = 0);

    //! Backward: grad_A = alpha * grad_C @ B^T, grad_B = alpha * A^T @ grad_C
    static void build_backward(const NNGraph::OpNode* op);
};

//! Convenience free function
inline NNGraph::TensorNode* gemm(
    NNGraph::TensorNode* a,
    NNGraph::TensorNode* b,
    const std::string& output_name,
    Scalar alpha = 1.0,
    bool trans_a = false,
    bool trans_b = false,
    Index ndim = 1,
    Index batch_ndim = 0)
{
    return Gemm::build_forward(a, b, output_name, alpha, trans_a, trans_b,
                               ndim, batch_ndim);
}

} // namespace nntile::graph
