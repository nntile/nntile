/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/gemm.hh
 * TensorGraph GEMM operation: C = alpha * A @ B + beta * C
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>
#include <vector>

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{
struct LoweringContext;
}

namespace nntile::graph::tensor
{

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
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim);

//! GEMM operation at tensor level: C = alpha * op(A) @ op(B) + beta * C
struct TensorGemmOp : TensorGraph::OpNode
{
    bool trans_a;
    bool trans_b;
    Scalar alpha;
    Scalar beta;
    Index ndim;
    Index batch_ndim;
    TensorGraph::TensorNode* a = nullptr;
    TensorGraph::TensorNode* b = nullptr;
    TensorGraph::TensorNode* c = nullptr;

    TensorGemmOp() = default;
    TensorGemmOp(
        TensorGraph::TensorNode* a_,
        TensorGraph::TensorNode* b_,
        TensorGraph::TensorNode* c_,
        Scalar alpha_, Scalar beta_,
        bool trans_a_, bool trans_b_,
        Index ndim_, Index batch_ndim_)
        : trans_a(trans_a_), trans_b(trans_b_)
        , alpha(alpha_), beta(beta_)
        , ndim(ndim_), batch_ndim(batch_ndim_)
        , a(a_), b(b_), c(c_)
    {
        inputs_ = {a, b, c};
        outputs_ = {c};
    }

    std::string op_name() const override { return "GEMM"; }


    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorGemmOp>(*this);
    }
    void lower_to_tile(const LoweringContext& ctx) const override;

};

//! GEMM creating new output: C = alpha * op(A) @ op(B)
//! @param a First input tensor
//! @param b Second input tensor
//! @param output_name Name for the output tensor
//! @param alpha Scalar multiplier for A @ B (default: 1.0)
//! @param trans_a Transpose A (default: false)
//! @param trans_b Transpose B (default: false)
//! @param ndim Number of contraction dimensions (default: 1)
//! @param batch_ndim Number of trailing batch dimensions (default: 0)
//! @return Pointer to the output tensor
TensorGraph::TensorNode* gemm(
    TensorGraph::TensorNode* a,
    TensorGraph::TensorNode* b,
    const std::string& output_name,
    Scalar alpha,
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim);

//! GEMM with accumulation: C = alpha * op(A) @ op(B) + beta * C
//! @param a First input tensor
//! @param b Second input tensor
//! @param c Tensor to accumulate into (modified in-place)
//! @param alpha Scalar multiplier for A @ B (default: 1.0)
//! @param beta Scalar multiplier for existing C (default: 1.0)
//! @param trans_a Transpose A (default: false)
//! @param trans_b Transpose B (default: false)
//! @param ndim Number of contraction dimensions (default: 1)
//! @param batch_ndim Number of trailing batch dimensions (default: 0)
void gemm(
    TensorGraph::TensorNode* a,
    TensorGraph::TensorNode* b,
    TensorGraph::TensorNode* c,
    Scalar alpha,
    Scalar beta,
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim);

} // namespace nntile::graph::tensor
