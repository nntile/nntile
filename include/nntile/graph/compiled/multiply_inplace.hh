#pragma once

#include <nntile/graph/compiled_graph.hh>

namespace nntile::graph
{

void execute_multiply_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info);

} // namespace nntile::graph
