/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/clear.hh
 * NNGraph clear: x = 0 (forward-only, no backward).
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/nn/nn_graph.hh>
#include <nntile/graph/tensor/clear.hh>

namespace nntile::graph
{

//! Clear NNGraph tensor: x = 0. Adds TensorClearOp to tensor graph.
inline void clear(NNGraph::TensorNode* x)
{
    if(x != nullptr)
    {
        graph::clear(x->data());
    }
}

} // namespace nntile::graph
