/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/scale.hh
 * NNGraph scale autograd operation.
 *
 * Forward: output = alpha * src
 * Backward: grad_src += alpha * grad_out
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>

// NNTile headers
#include <nntile/graph/nn/graph_op_node.hh>
#include <nntile/graph/tensor/scale.hh>

namespace nntile::graph
{

//! Scale op: output = alpha*src. PyTorch-style.
struct NNScaleOp : NNGraph::OpNode
{
    Scalar alpha;
    NNGraph::TensorNode* src = nullptr;

    NNScaleOp(NNGraph::TensorNode* src_,
              Scalar alpha_)
        : alpha(alpha_), src(src_)
    {
        inputs_ = {src};
    }

    NNGraph::TensorNode* forward(const std::string& output_name);
    void backward() const override;
};

NNGraph::TensorNode* scale(
    Scalar alpha,
    NNGraph::TensorNode* src,
    const std::string& output_name);

} // namespace nntile::graph
