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

void maybe_execute_after_record();

void set_axis_group_name(
    void *data_ptr,
    int ndim,
    const std::unordered_map<int, std::string> &names);

void set_axis_group_tiling(
    const std::string &name,
    const std::vector<std::int64_t> &tile_sizes);

} // namespace torch_nntile
