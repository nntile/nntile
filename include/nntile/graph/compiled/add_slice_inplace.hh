/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/compiled/add_slice_inplace.hh
 * Compiled graph add_slice_inplace operation.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/compiled_graph.hh>

namespace nntile::graph
{

//! Execute add_slice_inplace operation on compiled graph
void execute_add_slice_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info);

} // namespace nntile::graph
