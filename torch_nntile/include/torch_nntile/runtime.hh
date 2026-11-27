/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/include/torch_nntile/runtime.hh
 * Public runtime API for libtorch_nntile (context + TensorGraph).
 */

#pragma once

#include <cstddef>

namespace torch_nntile
{

//! Whether linked libnntile was compiled with ``NNTILE_USE_CUDA``.
bool built_with_cuda();

//! Unregistered aten ops always raise. ``cpu_fallback`` is kept for source
//! compatibility and must be false; true is rejected.
void init_context(
    int ncpu = -1,
    int ncuda = -1,
    int ooc_enabled = 0,
    char const *ooc_path = "/tmp/nntile_ooc",
    std::size_t ooc_size = 16 * 1024 * 1024,
    int logger = 0,
    int verbose = 0,
    bool cpu_fallback = false,
    std::size_t cpu_cap_bytes = 0,
    std::size_t cuda_cap_bytes_each = 0);

bool is_context_initialized();

bool is_cpu_fallback_enabled();

//! Block until submitted tasks finish.
void wait_for_all();

//! Shut down libnntile (safe to call repeatedly).
void shutdown_context();

bool has_pending_graph();

//! Lower and compile the pending TensorGraph.
void compile_graph();

//! Submit the compiled graph to NNHaul (does not wait).
void run_graph();

void reset_graph_session();

} // namespace torch_nntile
