/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_context.h
 */

#pragma once

#include <cstddef>

namespace torch_nntile
{

void init_context(
    int ncpu = -1,
    int ncuda = -1,
    int ooc_enabled = 0,
    const char *ooc_path = "/tmp/nntile_ooc",
    std::size_t ooc_size = 16 * 1024 * 1024,
    int logger = 0,
    int verbose = 0,
    bool cpu_fallback = true);

bool is_context_initialized();

bool is_context_verbose();

bool is_cpu_fallback_enabled();

void ensure_nntile_context();

void restrict_cpu();

void restrict_cuda();

void restore_where();

//! Block until all submitted StarPU tasks finish (including async unregisters).
void wait_for_all();

//! Shut down libnntile / StarPU and release the context (safe to call repeatedly).
void shutdown_context();

} // namespace torch_nntile
