/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn_graph/fill.hh
 * NNGraph fill: x = val (forward-only, no backward).
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/nn_graph/nn_graph.hh>
#include <nntile/graph/tensor/fill.hh>

namespace nntile::graph
{

//! Fill NNGraph tensor: x = val. Adds TensorFillOp to tensor graph.
inline void fill(Scalar val, NNGraph::TensorNode* x)
{
    if(x != nullptr)
    {
        graph::fill(val, x->data());
    }
}

} // namespace nntile::graph
