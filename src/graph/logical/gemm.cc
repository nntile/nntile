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

//! Compute output shape for gemm: C = alpha * op(A) @ op(B)
std::vector<Index> gemm_output_shape(
    const std::vector<Index>& a_shape,
    const std::vector<Index>& b_shape,
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim)
{
    Index a_ndim = static_cast<Index>(a_shape.size());
    Index b_ndim = static_cast<Index>(b_shape.size());

    std::vector<Index> output_shape;
    output_shape.reserve(a_ndim + b_ndim - 2 * ndim);

    Index a_batch_start = a_ndim - batch_ndim;
    Index b_batch_start = b_ndim - batch_ndim;

    Index a_m_begin = trans_a ? ndim : 0;
    Index a_m_end = trans_a ? a_batch_start : a_batch_start - ndim;
    Index b_n_begin = trans_b ? 0 : ndim;
    Index b_n_end = trans_b ? b_batch_start - ndim : b_batch_start;

    // Add M dimensions from A and N dimensions from B
    output_shape.insert(output_shape.end(),
                        a_shape.begin() + a_m_begin,
                        a_shape.begin() + a_m_end);
    output_shape.insert(output_shape.end(),
                        b_shape.begin() + b_n_begin,
                        b_shape.begin() + b_n_end);

    // Add batch dimensions
    output_shape.insert(output_shape.end(),
                        a_shape.begin() + a_batch_start,
                        a_shape.end());

    return output_shape;
}

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

    // Validate batch dimensions match
    if(batch_ndim > 0)
    {
        for(Index i = 0; i < batch_ndim; ++i)
        {
            Index batch_dim_a = a.ndim() - batch_ndim + i;
            Index batch_dim_b = b.ndim() - batch_ndim + i;
            if(a.shape()[batch_dim_a] != b.shape()[batch_dim_b])
            {
                throw std::invalid_argument(
                    "gemm: batch dimensions must match");
            }
        }
    }

    std::vector<Index> output_shape = gemm_output_shape(
        a.shape(), b.shape(), trans_a, trans_b, ndim, batch_ndim);

    // Create output tensor
    LogicalGraph::TensorNode& output = a.graph().tensor(
        std::move(output_shape),
        output_name,
        a.dtype());

    // Create operation attributes (beta = 0 for new output)
    auto attrs = std::make_shared<GemmAttrs>(GemmAttrs{trans_a, trans_b, alpha, 0.0, ndim, batch_ndim});

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

    // Validate batch dimensions match
    if(batch_ndim > 0)
    {
        for(Index i = 0; i < batch_ndim; ++i)
        {
            Index batch_dim_a = a.ndim() - batch_ndim + i;
            Index batch_dim_b = b.ndim() - batch_ndim + i;
            if(a.shape()[batch_dim_a] != b.shape()[batch_dim_b])
            {
                throw std::invalid_argument(
                    "gemm: batch dimensions must match");
            }
        }
    }

    // Compute expected output shape
    std::vector<Index> expected_shape = gemm_output_shape(
        a.shape(), b.shape(), trans_a, trans_b, ndim, batch_ndim);

    if(c.shape() != expected_shape)
    {
        throw std::invalid_argument(
            "gemm: tensor c has incompatible shape for accumulation");
    }

    // Create operation attributes
    auto attrs = std::make_shared<GemmAttrs>(GemmAttrs{trans_a, trans_b, alpha, beta, ndim, batch_ndim});

    // Add operation to graph
    a.graph().add_op(
        OpType::GEMM,
        attrs,
        {&a, &b, &c},
        {&c}
    );
}

} // namespace nntile::graph
