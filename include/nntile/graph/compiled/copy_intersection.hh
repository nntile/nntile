#pragma once

#include <nntile/graph/compiled_graph.hh>

namespace nntile::graph
{

void execute_copy_intersection(CompiledGraph& graph, const OpExecutionInfo& op_info);

} // namespace nntile::graph
