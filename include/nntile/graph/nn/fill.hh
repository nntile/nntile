/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/fill.hh
 * NNGraph fill: x = val (forward-only, no backward).
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <stdexcept>

// NNTile headers
#include <nntile/graph/nn/graph.hh>
#include <nntile/graph/tensor/fill.hh>

namespace nntile::graph
{

//! Fill NNGraph tensor: x = val. Adds TensorFillOp to tensor graph.
inline void fill(Scalar val, NNGraph::TensorNode* x)
{
    if(x == nullptr)
    {
        throw std::invalid_argument("fill: input tensor must be non-null");
    }
    graph::tensor::fill(val, x->data());
}

} // namespace nntile::graph
