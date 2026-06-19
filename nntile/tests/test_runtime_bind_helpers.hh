/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/tests/test_runtime_bind_helpers.hh
 * Helpers for explicit runtime binding in C++ tests.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/module/module.hh>
#include <nntile/runtime.hh>
#include <nntile/tensor/graph.hh>

namespace nntile::test
{

//! Bind every tensor in ``tg`` that carries a non-empty ``bind_hint``.
inline void bind_hints_from_tensor_graph(
    Runtime &rt, TensorGraph const &tg)
{
    for (auto const &node : tg.tensor_nodes())
    {
        TensorGraph::TensorNode const *tensor = node.get();
        const std::vector<std::uint8_t> *hint = tensor->get_bind_hint();
        if (hint != nullptr && !hint->empty())
        {
            rt.bind_data_from_hint(tensor);
        }
    }
}

//! Bind module parameters from their ``bind_hint`` (e.g. after ``load()``).
inline void bind_parameter_hints(Runtime &rt, module::Module &mod)
{
    for (NNGraph::TensorNode *tensor : mod.parameters_recursive())
    {
        rt.bind_data_from_hint(tensor);
    }
}

} // namespace nntile::test
