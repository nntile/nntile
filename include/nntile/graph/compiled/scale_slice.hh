#pragma once

#include <nntile/graph/compiled_graph.hh>

namespace nntile::graph
{

void execute_scale_slice(CompiledGraph& graph, const OpExecutionInfo& op_info);

} // namespace nntile::graph
