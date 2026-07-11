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

//! Enqueue seal/tiling/append/Runtime::compile on a background thread
//! (non-blocking). Prefer ``torch_nntile.compile_graph()``.
void compile_graph();

//! Request StarPU submit (non-blocking). Chains after in-flight compile.
void run_graph();

//! Join async compile/run, drain StarPU, then reclaim / release pin holds.
//! Prefer ``torch_nntile.wait()`` / ``wait_for_all()`` which call this.
//! Only this path may block on compile completion and StarPU.
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
