/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/compiled_graph_ops.hh
 * Compiled graph operations.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/compiled_graph.hh>

namespace nntile::graph
{

//! Execute clear operation on compiled graph
void execute_clear(CompiledGraph& graph, const OpExecutionInfo& op_info);

//! Execute gelu operation on compiled graph
void execute_gelu(CompiledGraph& graph, const OpExecutionInfo& op_info);

//! Execute gelu_backward operation on compiled graph
//! Note: dx tensor accumulates gradients (is both input and output)
void execute_gelu_backward(CompiledGraph& graph, const OpExecutionInfo& op_info);

//! Execute gemm operation on compiled graph
void execute_gemm(CompiledGraph& graph, const OpExecutionInfo& op_info);

//! Execute element-wise unary operations
void execute_gelu_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info);
void execute_gelutanh(CompiledGraph& graph, const OpExecutionInfo& op_info);
void execute_gelutanh_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info);
void execute_gelutanh_backward(CompiledGraph& graph, const OpExecutionInfo& op_info);
void execute_relu(CompiledGraph& graph, const OpExecutionInfo& op_info);
void execute_relu_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info);
void execute_relu_backward(CompiledGraph& graph, const OpExecutionInfo& op_info);
void execute_silu(CompiledGraph& graph, const OpExecutionInfo& op_info);
void execute_silu_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info);
void execute_silu_backward(CompiledGraph& graph, const OpExecutionInfo& op_info);
void execute_sqrt(CompiledGraph& graph, const OpExecutionInfo& op_info);
void execute_sqrt_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info);

//! Execute binary operations
void execute_add(CompiledGraph& graph, const OpExecutionInfo& op_info);
void execute_add_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info);
void execute_multiply(CompiledGraph& graph, const OpExecutionInfo& op_info);
void execute_multiply_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info);

//! Execute reduction operations
void execute_sum(CompiledGraph& graph, const OpExecutionInfo& op_info);
void execute_sum_fiber(CompiledGraph& graph, const OpExecutionInfo& op_info);

//! Execute scale operations
void execute_scale(CompiledGraph& graph, const OpExecutionInfo& op_info);
void execute_scale_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info);

//! Execute embedding operations
void execute_embedding(CompiledGraph& graph, const OpExecutionInfo& op_info);
void execute_embedding_backward(CompiledGraph& graph, const OpExecutionInfo& op_info);

//! Execute additional element-wise operations
void execute_hypot(CompiledGraph& graph, const OpExecutionInfo& op_info);
void execute_hypot_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info);
void execute_hypot_scalar_inverse(CompiledGraph& graph, const OpExecutionInfo& op_info);
void execute_pow(CompiledGraph& graph, const OpExecutionInfo& op_info);
void execute_pow_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info);
void execute_log_scalar(CompiledGraph& graph, const OpExecutionInfo& op_info);
void execute_mask_scalar(CompiledGraph& graph, const OpExecutionInfo& op_info);
void execute_subtract_indexed_outputs(CompiledGraph& graph, const OpExecutionInfo& op_info);

//! Execute additional reduction operations
void execute_sum_slice(CompiledGraph& graph, const OpExecutionInfo& op_info);
void execute_norm(CompiledGraph& graph, const OpExecutionInfo& op_info);
void execute_logsumexp(CompiledGraph& graph, const OpExecutionInfo& op_info);
void execute_maxsumexp(CompiledGraph& graph, const OpExecutionInfo& op_info);

//! Execute utility operations
void execute_fill(CompiledGraph& graph, const OpExecutionInfo& op_info);
void execute_copy(CompiledGraph& graph, const OpExecutionInfo& op_info);
void execute_transpose(CompiledGraph& graph, const OpExecutionInfo& op_info);
void execute_gather(CompiledGraph& graph, const OpExecutionInfo& op_info);

} // namespace nntile::graph
