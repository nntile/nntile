/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/core/log_scalar.cc
 * Log scalar value from Tile<T>
 *
 * @version 2.0.0
 * */

#include "nntile/core/log_scalar.hh"
#include "nntile/nnhaul/ops/log_scalar.hh"
#include "nntile/nnhaul/sync_defer.hh"

namespace nntile::core
{

template<typename T>
void log_scalar_async(int worker_hint, const std::string &name, const Tile<T> &value)
{
    // Check if value is a scalar tile
    if(value.nelems != 1)
    {
        throw std::runtime_error("value must be a scalar tile");
    }
    nnhaul::log_scalar.submit<std::tuple<T>>(worker_hint, name, value);
}

template<typename T>
void log_scalar(int worker_hint, const std::string &name, const Tile<T> &value)
{
    log_scalar_async<T>(worker_hint, name, value);
    nntile::nnhaul_task_wait_for_all_unless_deferred();
}

// Explicit instantiation
template
void log_scalar_async<fp32_t>(int worker_hint, const std::string &name,
        const Tile<fp32_t> &value);

template
void log_scalar_async<fp64_t>(int worker_hint, const std::string &name,
        const Tile<fp64_t> &value);

template
void log_scalar_async<fp32_fast_tf32_t>(int worker_hint, const std::string &name,
        const Tile<fp32_fast_tf32_t> &value);

template
void log_scalar_async<fp32_fast_fp16_t>(int worker_hint, const std::string &name,
        const Tile<fp32_fast_fp16_t> &value);

template
void log_scalar_async<fp32_fast_bf16_t>(int worker_hint, const std::string &name,
        const Tile<fp32_fast_bf16_t> &value);

template
void log_scalar_async<bf16_t>(int worker_hint, const std::string &name,
        const Tile<bf16_t> &value);

template
void log_scalar<fp32_t>(int worker_hint, const std::string &name, const Tile<fp32_t> &value);

template
void log_scalar<fp64_t>(int worker_hint, const std::string &name, const Tile<fp64_t> &value);

template
void log_scalar<fp32_fast_tf32_t>(int worker_hint, const std::string &name,
        const Tile<fp32_fast_tf32_t> &value);

template
void log_scalar<fp32_fast_fp16_t>(int worker_hint, const std::string &name,
        const Tile<fp32_fast_fp16_t> &value);

template
void log_scalar<fp32_fast_bf16_t>(int worker_hint, const std::string &name,
        const Tile<fp32_fast_bf16_t> &value);

template
void log_scalar<bf16_t>(int worker_hint, const std::string &name, const Tile<bf16_t> &value);

} // namespace nntile::core
