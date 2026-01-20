/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/gemm.cc
 * GEMM operation implementation for logical graph.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/gemm.hh"

// Include standard headers
#include <stdexcept>

// Include other NNTile headers
#include "nntile/graph/logical_graph.hh"
#include "nntile/graph/op_node.hh"

namespace nntile::graph
{

namespace
{

//! Compute output shape for gemm (tensor contraction) operation
//!
//! Tensor layout (column-major, dimensions listed from inner to outer):
//!
//! A (no trans): [M_dims..., K_dims..., batch_dims...]
//! A (trans):    [K_dims..., M_dims..., batch_dims...]
//!
//! B (no trans): [K_dims..., N_dims..., batch_dims...]
//! B (trans):    [N_dims..., K_dims..., batch_dims...]
//!
//! C:            [M_dims..., N_dims..., batch_dims...]
//!
//! Where:
//!   - ndim = number of contraction dimensions (K)
//!   - batch_ndim = number of batch dimensions (must match between A and B)
//!
TensorSpec compute_gemm_output_spec(
    const TensorSpec& a,
    const TensorSpec& b,
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim)
{
    // Validate ndim and batch_ndim are non-negative
    if(ndim < 1)
    {
        throw std::invalid_argument(
            "gemm: ndim must be at least 1");
    }
    if(batch_ndim < 0)
    {
        throw std::invalid_argument(
            "gemm: batch_ndim must be non-negative");
    }

    // Validate tensor dimensions are sufficient
    // A needs at least ndim + batch_ndim dimensions (M can be empty for vector)
    // But typically A needs ndim + batch_ndim + (M_ndim >= 1)
    if(a.ndim() < ndim + batch_ndim)
    {
        throw std::invalid_argument(
            "gemm: tensor A has insufficient dimensions for given ndim and batch_ndim");
    }
    if(b.ndim() < ndim + batch_ndim)
    {
        throw std::invalid_argument(
            "gemm: tensor B has insufficient dimensions for given ndim and batch_ndim");
    }

    // Compute dimension counts
    Index a_m_ndim = a.ndim() - ndim - batch_ndim;  // Number of M dimensions
    Index b_n_ndim = b.ndim() - ndim - batch_ndim;  // Number of N dimensions

    // Extract shapes based on transpose flags
    // A: [M..., K..., batch...] or [K..., M..., batch...] if trans_a
    // B: [K..., N..., batch...] or [N..., K..., batch...] if trans_b

    // Get K dimensions from A
    std::vector<Index> k_dims_a(ndim);
    Index k_start_a = trans_a ? 0 : a_m_ndim;
    for(Index i = 0; i < ndim; ++i)
    {
        k_dims_a[i] = a.dim(k_start_a + i);
    }

    // Get K dimensions from B
    std::vector<Index> k_dims_b(ndim);
    Index k_start_b = trans_b ? b_n_ndim : 0;
    for(Index i = 0; i < ndim; ++i)
    {
        k_dims_b[i] = b.dim(k_start_b + i);
    }

    // Validate K dimensions match
    for(Index i = 0; i < ndim; ++i)
    {
        if(k_dims_a[i] != k_dims_b[i])
        {
            throw std::invalid_argument(
                "gemm: contraction dimension " + std::to_string(i) +
                " mismatch: A has " + std::to_string(k_dims_a[i]) +
                ", B has " + std::to_string(k_dims_b[i]));
        }
    }

    // Get batch dimensions from A and B
    std::vector<Index> batch_dims_a(batch_ndim);
    std::vector<Index> batch_dims_b(batch_ndim);
    for(Index i = 0; i < batch_ndim; ++i)
    {
        batch_dims_a[i] = a.dim(a.ndim() - batch_ndim + i);
        batch_dims_b[i] = b.dim(b.ndim() - batch_ndim + i);
    }

    // Validate batch dimensions match
    for(Index i = 0; i < batch_ndim; ++i)
    {
        if(batch_dims_a[i] != batch_dims_b[i])
        {
            throw std::invalid_argument(
                "gemm: batch dimension " + std::to_string(i) +
                " mismatch: A has " + std::to_string(batch_dims_a[i]) +
                ", B has " + std::to_string(batch_dims_b[i]));
        }
    }

    // Build output shape: [M..., N..., batch...]
    std::vector<Index> output_shape;
    output_shape.reserve(a_m_ndim + b_n_ndim + batch_ndim);

    // Add M dimensions from A
    Index m_start_a = trans_a ? ndim : 0;
    for(Index i = 0; i < a_m_ndim; ++i)
    {
        output_shape.push_back(a.dim(m_start_a + i));
    }

    // Add N dimensions from B
    Index n_start_b = trans_b ? 0 : ndim;
    for(Index i = 0; i < b_n_ndim; ++i)
    {
        output_shape.push_back(b.dim(n_start_b + i));
    }

    // Add batch dimensions
    for(Index i = 0; i < batch_ndim; ++i)
    {
        output_shape.push_back(batch_dims_a[i]);
    }

    return TensorSpec(output_shape, a.dtype());
}

} // namespace

//! Validate inputs for gemm operation
void validate_gemm_inputs(
    TensorNode& a,
    TensorNode& b,
    LogicalGraph& expected_graph)
{
    // Validate inputs belong to the same graph
    if(&a.graph() != &expected_graph || &b.graph() != &expected_graph)
    {
        throw std::invalid_argument(
            "gemm: input tensors must belong to the same graph");
    }

    // Validate input dtypes match
    if(a.dtype() != b.dtype())
    {
        throw std::invalid_argument(
            "gemm: input tensors must have the same dtype");
    }
}

//! Tensor contraction creating new output: C = alpha * op(A) @ op(B)
TensorNode& gemm(
    TensorNode& a,
    TensorNode& b,
    const std::string& output_name,
    Scalar alpha,
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim)
{
    validate_gemm_inputs(a, b, a.graph());

    // Compute output specification
    TensorSpec output_spec = compute_gemm_output_spec(
        a.spec(), b.spec(), trans_a, trans_b, ndim, batch_ndim
    );

    // Create output tensor
    TensorNode& output = a.graph().tensor(output_spec, output_name);

    // Create operation attributes (beta = 0 for new output)
    OpAttrs attrs = GemmAttrs{trans_a, trans_b, alpha, 0.0, ndim, batch_ndim};

    // Add operation to graph using public builder API
    a.graph().add_op(
        OpType::GEMM,
        attrs,
        {&a, &b},
        {&output}
    );

    return output;
}

//! Tensor contraction with accumulation: C = alpha * op(A) @ op(B) + beta * C
void gemm(
    TensorNode& a,
    TensorNode& b,
    TensorNode& c,
    Scalar alpha,
    Scalar beta,
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim)
{
    validate_gemm_inputs(a, b, a.graph());

    // Validate c belongs to the same graph
    if(&c.graph() != &a.graph())
    {
        throw std::invalid_argument(
            "gemm: tensor c must belong to the same graph as a and b");
    }

    // Validate c dtype matches
    if(c.dtype() != a.dtype())
    {
        throw std::invalid_argument(
            "gemm: tensor c must have the same dtype as a and b");
    }

    // Compute expected output shape and validate against c
    TensorSpec expected_spec = compute_gemm_output_spec(
        a.spec(), b.spec(), trans_a, trans_b, ndim, batch_ndim
    );

    if(c.shape() != expected_spec.shape())
    {
        throw std::invalid_argument(
            "gemm: tensor c has incompatible shape for accumulation");
    }

    // Create operation attributes
    OpAttrs attrs = GemmAttrs{trans_a, trans_b, alpha, beta, ndim, batch_ndim};

    // Add operation to graph
    a.graph().add_op(
        OpType::GEMM,
        attrs,
        {&a, &b},
        {&c}
    );
}

} // namespace nntile::graph
