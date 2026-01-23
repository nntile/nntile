#pragma once

#include <nntile/graph/compiled_graph.hh>

namespace nntile::graph
{

void execute_mask_scalar(CompiledGraph& graph, const OpExecutionInfo& op_info);

} // namespace nntile::graph
