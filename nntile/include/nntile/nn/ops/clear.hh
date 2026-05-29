/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/nn/ops/clear.hh
 * NNGraph clear: x = 0 (forward-only, no backward).
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <stdexcept>

// NNTile headers
#include <nntile/nn/graph.hh>
#include <nntile/tensor/ops/clear.hh>

namespace nntile
{

//! Clear NNGraph tensor: x = 0. Adds TensorClearOp to tensor graph.
inline void clear(NNGraph::TensorNode* x)
{
    if(x == nullptr)
    {
        throw std::invalid_argument("clear: input tensor must be non-null");
    }
    tensor::clear(x->data());
}

} // namespace nntile
