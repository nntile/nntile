/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/nn/ops/transpose.hh
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
#include <nntile/nn/graph_op_node.hh>

namespace nntile
{

//! Transpose op: cyclic shift of dimensions.
//!
//! ``ndim`` uses **storage-order** axis semantics (matching graph model code):
//! NN ``forward`` calls ``tensor::transpose(..., src->ndim() - ndim)``; backward
//! uses ``tensor::transpose(..., ndim)`` as the inverse.
struct NNTransposeOp : NNGraph::OpNode
{
    Index ndim;
    NNGraph::TensorNode *src = nullptr;

    NNTransposeOp(NNGraph::TensorNode *src_, Index ndim_) :
        ndim(ndim_), src(src_)
    {
        inputs_ = {src};
    }

    NNGraph::TensorNode *forward();
    void backward() const override;
};

//! Transpose: cyclic shift by ``ndim`` storage-order axes (see ``NNTransposeOp``).
NNGraph::TensorNode *transpose(NNGraph::TensorNode *src, Index ndim);

} // namespace nntile
