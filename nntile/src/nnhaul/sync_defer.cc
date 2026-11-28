/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/src/nnhaul/sync_defer.cc
 * Definitions for the NNHaul sync-defer TLS / wait counters.
 * @version 2.0.0
 */

#include <nntile/nnhaul/sync_defer.hh>

namespace nntile
{

thread_local int g_sync_defer_depth = 0;
std::atomic<std::uint64_t> g_wait_for_all_count{0};

void nnhaul_wait_for_all_drained()
{
    // The raw drain shim (data_access.hh) shares this counted wait so
    // staged scratch handles are released on that path too.
    task_wait_for_all_counted();
}

} // namespace nntile
