/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/core/execution_worker.hh
 * Thread-local StarPU worker preference for scheduled tile ops (lightweight).
 *
 * @version 1.1.0
 * */

#pragma once

namespace nntile::sched
{

int count_execution_workers();

int logical_worker_to_starpu_id(int logical_worker, bool use_cuda_workers);

int preferred_starpu_worker_id();
void set_preferred_starpu_worker_id(int starpu_worker_id);

class ScopedPreferredWorker
{
public:
    explicit ScopedPreferredWorker(int logical_worker, bool use_cuda_workers);
    ~ScopedPreferredWorker();

    ScopedPreferredWorker(const ScopedPreferredWorker &) = delete;
    ScopedPreferredWorker &operator=(const ScopedPreferredWorker &) = delete;

private:
    int previous_ = -1;
};

} // namespace nntile::sched
