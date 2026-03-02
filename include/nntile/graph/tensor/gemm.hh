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

#include <string>
#include <vector>

#include <nntile/base_types.hh>
#include <nntile/graph/tensor_graph.hh>
#include <nntile/graph/base_op_node.hh>

namespace nntile::graph
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
    bool trans_a = false,
    bool trans_b = false,
    Index ndim = 1,
    Index batch_ndim = 0);

//! GEMM operation at tensor level: C = alpha * op(A) @ op(B) + beta * C
struct TensorGemmOp : BaseOpNode<TensorGraph, TensorGraphNode>
{
    bool trans_a = false;
    bool trans_b = false;
    Scalar alpha = 1.0;
    Scalar beta = 0.0;
    Index ndim = 1;
    Index batch_ndim = 0;
    TensorGraph::DataNode* a = nullptr;
    TensorGraph::DataNode* b = nullptr;
    TensorGraph::DataNode* c = nullptr;

    TensorGemmOp() = default;
    TensorGemmOp(
        TensorGraph::DataNode* a_,
        TensorGraph::DataNode* b_,
        TensorGraph::DataNode* c_,
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

    void execute(ExecutionContext<TensorGraph::DataNode>& ctx) const override;

    std::shared_ptr<BaseOpNode<TensorGraph, TensorGraphNode>> clone() const override
    {
        return std::make_shared<TensorGemmOp>(*this);
    }
};

//! GEMM creating new output: C = alpha * op(A) @ op(B)
//! @param a First input tensor
//! @param b Second input tensor
//! @param output_name Name for the output tensor
//! @param alpha Scalar multiplier (default: 1.0)
//! @param trans_a Transpose A (default: false)
//! @param trans_b Transpose B (default: false)
//! @param ndim Number of contraction dimensions (default: 1)
//! @param batch_ndim Number of trailing batch dimensions (default: 0)
//! @return Pointer to the output tensor
TensorGraph::DataNode* gemm(
    TensorGraph::DataNode* a,
    TensorGraph::DataNode* b,
    const std::string& output_name,
    Scalar alpha = 1.0,
    bool trans_a = false,
    bool trans_b = false,
    Index ndim = 1,
    Index batch_ndim = 0);

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
    TensorGraph::DataNode* a,
    TensorGraph::DataNode* b,
    TensorGraph::DataNode* c,
    Scalar alpha = 1.0,
    Scalar beta = 1.0,
    bool trans_a = false,
    bool trans_b = false,
    Index ndim = 1,
    Index batch_ndim = 0);

} // namespace nntile::graph
