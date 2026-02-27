/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn_graph/gemm.hh
 * NNGraph GEMM operation overload.
 *
 * @version 1.1.0
 * */

#pragma once

// Include other NNTile headers
#include <nntile/graph/logical/gemm.hh>
#include <nntile/graph/nn_graph.hh>

namespace nntile::graph
{

//! Tensor contraction creating new output: C = alpha * op(A) @ op(B)
//! Overload for NNGraph::TensorNode
inline NNGraph::TensorNode& gemm(
    NNGraph& graph,
    NNGraph::TensorNode& a,
    NNGraph::TensorNode& b,
    const std::string& output_name,
    Scalar alpha = 1.0,
    bool trans_a = false,
    bool trans_b = false,
    Index ndim = 1,
    Index batch_ndim = 0)
{
    LogicalGraph::TensorNode& c_data = gemm(
        a.data(), b.data(), output_name, alpha, trans_a, trans_b, ndim,
        batch_ndim);
    return *graph.tensor(c_data);
}

//! Tensor contraction with accumulation: C = alpha * op(A) @ op(B) + beta * C
//! Overload for NNGraph::TensorNode
inline void gemm(
    NNGraph::TensorNode& a,
    NNGraph::TensorNode& b,
    NNGraph::TensorNode& c,
    Scalar alpha = 1.0,
    Scalar beta = 1.0,
    bool trans_a = false,
    bool trans_b = false,
    Index ndim = 1,
    Index batch_ndim = 0)
{
    gemm(a.data(), b.data(), c.data(), alpha, beta, trans_a, trans_b, ndim,
         batch_ndim);
}

} // namespace nntile::graph
