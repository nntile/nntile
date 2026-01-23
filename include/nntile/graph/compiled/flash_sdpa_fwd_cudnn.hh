#pragma once

#include <nntile/graph/compiled_graph.hh>

namespace nntile::graph
{

void execute_flash_sdpa_fwd_cudnn(CompiledGraph& graph, const OpExecutionInfo& op_info);

} // namespace nntile::graph
