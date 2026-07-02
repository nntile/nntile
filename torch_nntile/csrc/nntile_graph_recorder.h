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
    TensorImplKey impl_key,
    int ndim,
    const std::unordered_map<int, std::string> &names);

void set_axis_group_tiling(
    const std::string &name,
    const std::vector<std::int64_t> &tile_sizes);

std::string format_axis_groups();

void print_axis_groups();

//! Snapshot recorder state for tensor GC investigation.
struct GcDebugStats
{
    std::int64_t pinned_tensors = 0;
    std::int64_t tensor_nodes = 0;
    std::int64_t tile_pool = 0;
    std::int64_t pending_ops = 0;
    std::int64_t pending_data = 0;
    bool has_session = false;
};

GcDebugStats debug_gc_stats();

void copy_nntile_tensor_to_cpu(const at::Tensor &src, at::Tensor &dst);

} // namespace torch_nntile
