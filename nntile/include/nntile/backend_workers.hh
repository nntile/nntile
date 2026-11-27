/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/backend_workers.hh
 * Worker counts recorded by ``Context`` for the NNHaul build.
 */

#pragma once

#include <string>
#include <vector>

namespace nntile
{

enum class ScheduleKindFilter
{
    Any,
    Cpu,
    Cuda
};

extern int g_backend_ncpu;
extern int g_backend_ncuda;
extern bool g_backend_ready;
extern ScheduleKindFilter g_schedule_kind_filter;

struct BackendWorker
{
    int id = 0;
    std::string kind;
    int device = -1;
};

std::vector<BackendWorker> backend_worker_list();

} // namespace nntile
