/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/gemm.hh
 * GEMM operation for logical graph.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// Include third-party headers

// Include other NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tensor_node.hh>

namespace nntile::graph
{

//! Tensor contraction (generalized matrix multiplication):
//!   C = alpha * op(A) @ op(B) + beta * C
//!
//! Tensor layout (column-major, dimensions listed from inner to outer):
//!
//!   A (no trans): [M_dims..., K_dims..., batch_dims...]
//!   A (trans):    [K_dims..., M_dims..., batch_dims...]
//!
//!   B (no trans): [K_dims..., N_dims..., batch_dims...]
//!   B (trans):    [N_dims..., K_dims..., batch_dims...]
//!
//!   C:            [M_dims..., N_dims..., batch_dims...]
//!
//! @param a First input tensor
//! @param b Second input tensor
//! @param output_name Name for the output tensor
//! @param alpha Scalar multiplier for A @ B (default: 1.0)
//! @param beta Scalar multiplier for C (default: 0.0)
//! @param trans_a Swap M and K dimensions in A (default: false)
//! @param trans_b Swap K and N dimensions in B (default: false)
//! @param ndim Number of contraction dimensions (K) (default: 1)
//! @param batch_ndim Number of trailing batch dimensions (default: 0)
//! @return Reference to the output tensor
TensorNode& gemm(
    TensorNode& a,
    TensorNode& b,
    const std::string& output_name,
    Scalar alpha = 1.0,
    Scalar beta = 0.0,
    bool trans_a = false,
    bool trans_b = false,
    Index ndim = 1,
    Index batch_ndim = 0
);

} // namespace nntile::graph
