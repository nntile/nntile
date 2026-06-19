/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_graph_recorder.h
 */

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace torch_nntile
{

bool has_pending_graph();

void require_no_pending_graph(const char *op_name);

void execute_pending_graph();

void compile_graph();

void run_graph();

void reset_graph_session();

void shutdown_recorder();

bool has_graph_session();

void sync_nntile_storage_to_runtime(void *data_ptr);

void sync_runtime_to_nntile_storage(void *data_ptr);

void maybe_execute_after_record();

void set_axis_group_name(
    void *data_ptr,
    int ndim,
    const std::unordered_map<int, std::string> &names);

void set_axis_group_tiling(
    const std::string &name,
    const std::vector<std::int64_t> &tile_sizes);

std::string format_axis_groups();

void print_axis_groups();

} // namespace torch_nntile
