/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/gemm.hh
 * Logical graph GEMM operation.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>
#include <vector>

// Include other NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! GEMM: C = alpha * A @ B + beta * C
struct GemmAttrs
{
    bool trans_a = false;
    bool trans_b = false;
    Scalar alpha = 1.0;
    Scalar beta = 0.0;
    Index ndim = 1;
    Index batch_ndim = 0;
};

//! Compute output shape for gemm: C = alpha * op(A) @ op(B)
//! @param a_shape Shape of first input tensor
//! @param b_shape Shape of second input tensor
//! @param trans_a Swap first ndim dimensions in A
//! @param trans_b Swap first ndim dimensions in B
//! @param ndim Number of contraction dimensions (default: 1)
//! @param batch_ndim Number of trailing batch dimensions (default: 0)
//! @return Output shape for the gemm result
std::vector<Index> gemm_output_shape(
    const std::vector<Index>& a_shape,
    const std::vector<Index>& b_shape,
    bool trans_a = false,
    bool trans_b = false,
    Index ndim = 1,
    Index batch_ndim = 0);

//! Tensor contraction creating new output: C = alpha * op(A) @ op(B)
//! @param a First input tensor
//! @param b Second input tensor
//! @param output_name Name for the output tensor
//! @param alpha Scalar multiplier for A @ B (default: 1.0)
//! @param trans_a Swap M and K dimensions in A (default: false)
//! @param trans_b Swap K and N dimensions in B (default: false)
//! @param ndim Number of contraction dimensions (K) (default: 1)
//! @param batch_ndim Number of trailing batch dimensions (default: 0)
//! @return Reference to the created output tensor
LogicalGraph::TensorNode& gemm(
    LogicalGraph::TensorNode& a,
    LogicalGraph::TensorNode& b,
    const std::string& output_name,
    Scalar alpha = 1.0,
    bool trans_a = false,
    bool trans_b = false,
    Index ndim = 1,
    Index batch_ndim = 0
);

//! Tensor contraction with accumulation: C = alpha * op(A) @ op(B) + beta * C
//! @param a First input tensor
//! @param b Second input tensor
//! @param c Existing tensor to accumulate into (modified in-place)
//! @param alpha Scalar multiplier for A @ B (default: 1.0)
//! @param beta Scalar multiplier for existing C (default: 1.0)
//! @param trans_a Swap M and K dimensions in A (default: false)
//! @param trans_b Swap K and N dimensions in B (default: false)
//! @param ndim Number of contraction dimensions (K) (default: 1)
//! @param batch_ndim Number of trailing batch dimensions (default: 0)
void gemm(
    LogicalGraph::TensorNode& a,
    LogicalGraph::TensorNode& b,
    LogicalGraph::TensorNode& c,
    Scalar alpha = 1.0,
    Scalar beta = 1.0,
    bool trans_a = false,
    bool trans_b = false,
    Index ndim = 1,
    Index batch_ndim = 0
);

} // namespace nntile::graph
