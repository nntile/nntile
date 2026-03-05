/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/transpose.hh
 * NNGraph transpose autograd operation (cyclic shift of dimensions).
 *
 * Forward: output[i] = src[(i+ndim) % ndim]
 * Backward: grad_src gets transpose of grad_out with inverse permutation
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/nn/graph_op_node.hh>

namespace nntile::graph
{

//! Transpose op: cyclic shift of dimensions. PyTorch-style.
struct NNTransposeOp : NNGraph::OpNode
{
    Index ndim;
    NNGraph::TensorNode* src = nullptr;

    NNTransposeOp(NNGraph::TensorNode* src_, Index ndim_)
        : ndim(ndim_), src(src_)
    {
        inputs_ = {src};
    }

    NNGraph::TensorNode* forward(const std::string& output_name);
    void backward() const override;
};

//! Transpose: cyclic shift by ndim dimensions
NNGraph::TensorNode* transpose(
    NNGraph::TensorNode* src,
    const std::string& output_name,
    Index ndim);

} // namespace nntile::graph
