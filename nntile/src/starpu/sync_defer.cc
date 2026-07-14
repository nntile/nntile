/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/src/starpu/sync_defer.cc
 * Definitions for StarPU sync-defer TLS / wait counters.
 */

#include <nntile/starpu/sync_defer.hh>

namespace nntile
{

thread_local int g_starpu_sync_defer_depth = 0;
std::atomic<std::uint64_t> g_starpu_wait_for_all_count{0};

} // namespace nntile
