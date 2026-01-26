/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/gemm.cc
 * Logical graph GEMM operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/gemm.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Tensor contraction creating new output: C = alpha * op(A) @ op(B)
LogicalGraph::TensorNode& gemm(
    LogicalGraph::TensorNode& a,
    LogicalGraph::TensorNode& b,
    const std::string& output_name,
    Scalar alpha,
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim)
{
    // Validate inputs belong to the same graph
    if(&a.graph() != &b.graph())
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

    // Compute output specification
    std::vector<Index> output_shape = a.shape();

    // Handle transpose for A
    if(trans_a)
    {
        // Swap first ndim dimensions for transpose
        for(Index i = 0; i < ndim/2; ++i)
        {
            std::swap(output_shape[i], output_shape[ndim-1-i]);
        }
    }

    // Remove contraction dimensions from A and B, add result dimensions
    output_shape.erase(output_shape.begin() + (trans_a ? ndim - ndim : 0),
                      output_shape.begin() + (trans_a ? ndim : ndim));

    std::vector<Index> b_shape = b.shape();
    if(trans_b)
    {
        // Swap first ndim dimensions for transpose
        for(Index i = 0; i < ndim/2; ++i)
        {
            std::swap(b_shape[i], b_shape[ndim-1-i]);
        }
    }

    // Add dimensions from B (excluding contraction dimensions)
    output_shape.insert(output_shape.begin(),
                       b_shape.begin() + (trans_b ? 0 : ndim),
                       b_shape.begin() + (trans_b ? ndim - ndim : ndim));

    // Add batch dimensions
    if(batch_ndim > 0)
    {
        Index total_ndim = a.ndim() + b.ndim() - 2*ndim;
        for(Index i = 0; i < batch_ndim; ++i)
        {
            Index batch_dim_a = a.ndim() - batch_ndim + i;
            Index batch_dim_b = b.ndim() - batch_ndim + i;
            if(a.shape()[batch_dim_a] != b.shape()[batch_dim_b])
            {
                throw std::invalid_argument(
                    "gemm: batch dimensions must match");
            }
            output_shape.push_back(a.shape()[batch_dim_a]);
        }
    }

    // Create output tensor
    LogicalGraph::TensorNode& output = a.graph().tensor(
        std::move(output_shape),
        output_name,
        a.dtype());

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
    LogicalGraph::TensorNode& a,
    LogicalGraph::TensorNode& b,
    LogicalGraph::TensorNode& c,
    Scalar alpha,
    Scalar beta,
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim)
{
    // Validate inputs belong to the same graph
    if(&a.graph() != &b.graph())
    {
        throw std::invalid_argument(
            "gemm: input tensors must belong to the same graph");
    }

    // Validate c belongs to the same graph
    if(&c.graph() != &a.graph())
    {
        throw std::invalid_argument(
            "gemm: tensor c must belong to the same graph as a and b");
    }

    // Validate input dtypes match
    if(a.dtype() != b.dtype())
    {
        throw std::invalid_argument(
            "gemm: input tensors must have the same dtype");
    }

    // Validate c dtype matches
    if(c.dtype() != a.dtype())
    {
        throw std::invalid_argument(
            "gemm: tensor c must have the same dtype as a and b");
    }

    // Compute expected output shape
    std::vector<Index> expected_shape = a.shape();

    // Handle transpose for A
    if(trans_a)
    {
        // Swap first ndim dimensions for transpose
        for(Index i = 0; i < ndim/2; ++i)
        {
            std::swap(expected_shape[i], expected_shape[ndim-1-i]);
        }
    }

    // Remove contraction dimensions from A and B, add result dimensions
    expected_shape.erase(expected_shape.begin() + (trans_a ? ndim - ndim : 0),
                        expected_shape.begin() + (trans_a ? ndim : ndim));

    std::vector<Index> b_shape = b.shape();
    if(trans_b)
    {
        // Swap first ndim dimensions for transpose
        for(Index i = 0; i < ndim/2; ++i)
        {
            std::swap(b_shape[i], b_shape[ndim-1-i]);
        }
    }

    // Add dimensions from B (excluding contraction dimensions)
    expected_shape.insert(expected_shape.begin(),
                         b_shape.begin() + (trans_b ? 0 : ndim),
                         b_shape.begin() + (trans_b ? ndim - ndim : ndim));

    // Add batch dimensions
    if(batch_ndim > 0)
    {
        Index total_ndim = a.ndim() + b.ndim() - 2*ndim;
        for(Index i = 0; i < batch_ndim; ++i)
        {
            Index batch_dim_a = a.ndim() - batch_ndim + i;
            Index batch_dim_b = b.ndim() - batch_ndim + i;
            if(a.shape()[batch_dim_a] != b.shape()[batch_dim_b])
            {
                throw std::invalid_argument(
                    "gemm: batch dimensions must match");
            }
            expected_shape.push_back(a.shape()[batch_dim_a]);
        }
    }

    if(c.shape() != expected_shape)
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
        {&a, &b, &c},
        {&c}
    );
}

} // namespace nntile::graph