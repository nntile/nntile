/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/compiled/gelutanh_backward.hh
 * Compiled graph gelutanh_backward operation.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/compiled_graph.hh>

namespace nntile::graph
{

//! Execute gelutanh_backward operation on compiled graph
//! Note: dx tensor accumulates gradients (is both input and output)
void execute_gelutanh_backward(CompiledGraph& graph, const OpExecutionInfo& op_info);

} // namespace nntile::graph
