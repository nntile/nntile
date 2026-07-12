/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_graph_recorder.h
 */

#pragma once

#include "nntile_tensor_gc.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace at
{
class Tensor;
}

namespace torch_nntile
{

bool has_pending_graph();

void require_no_pending_graph(const char *op_name);

void execute_pending_graph();

//! Lower and compile the pending TensorGraph (synchronous CPU work).
void compile_graph();

//! Submit the compiled graph to StarPU (asynchronous; does not wait).
void run_graph();

//! Block until submitted run() tasks finish; then reclaim / release pin holds
//! and compact the incremental session (tile reset + drop sealed ops).
//! Prefer ``torch_nntile.wait()`` / ``wait_for_all()`` which call this.
void wait_graph_session();

void reset_graph_session();

void shutdown_recorder();

bool has_graph_session();

void set_axis_group_name(
    const at::Tensor &tensor,
    const std::unordered_map<int, std::string> &names);

bool is_tensor_graph_output(const at::Tensor &tensor);

void stage_tensor_for_axis_group_compile(const at::Tensor &tensor);

void set_axis_group_tiling(
    const std::string &name,
    const std::vector<std::int64_t> &tile_sizes);

std::string format_axis_groups();

void print_axis_groups();

void print_info();

void copy_nntile_tensor_to_cpu(const at::Tensor &src, at::Tensor &dst);

} // namespace torch_nntile
