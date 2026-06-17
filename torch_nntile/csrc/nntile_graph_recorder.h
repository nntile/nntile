/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_graph_recorder.h
 */

#pragma once

namespace torch_nntile
{

bool has_pending_graph();

void require_no_pending_graph(const char *op_name);

void execute_pending_graph();

void maybe_execute_after_record();

} // namespace torch_nntile
