/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/nn_grad_slot_name.hh
 * Stable gradient-slot names for ``get_or_create_grad`` when tensors may be
 * unnamed (identity is the tensor node pointer / id).
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/nn/graph_data_node.hh>
#include <string>

namespace nntile::graph
{

//! Slot name for gradient tensors tied to \p t (non-empty name → name_grad).
inline std::string nn_grad_slot_name(NNGraph::TensorNode const *t)
{
    if (t == nullptr || t->data() == nullptr)
    {
        return {"grad"};
    }
    if (!t->name().empty())
    {
        return t->name() + "_grad";
    }
    return std::string{"g_"} + std::to_string(t->data()->id());
}

} // namespace nntile::graph
