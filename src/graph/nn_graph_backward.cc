/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file src/graph/nn_graph_backward.cc
 * Backward registry implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn_graph_backward.hh"

#include <map>
#include <mutex>

namespace nntile::graph
{

namespace
{
std::map<OpType, BackwardFn>& get_registry()
{
    static std::map<OpType, BackwardFn> registry;
    return registry;
}

std::mutex& get_registry_mutex()
{
    static std::mutex m;
    return m;
}
} // namespace

void register_backward(OpType type, BackwardFn fn)
{
    std::lock_guard<std::mutex> lock(get_registry_mutex());
    get_registry()[type] = std::move(fn);
}

BackwardFn get_backward(OpType type)
{
    std::lock_guard<std::mutex> lock(get_registry_mutex());
    auto it = get_registry().find(type);
    return it != get_registry().end() ? it->second : BackwardFn{};
}

} // namespace nntile::graph
