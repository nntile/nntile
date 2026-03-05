/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/log_scalar.cc
 * Log scalar value from Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/log_scalar.hh"
#include "nntile/starpu/log_scalar.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tile
{

template<typename T>
void log_scalar_async(const std::string &name, const Tile<T> &value)
{
    // Check if value is a scalar tile
    if(value.nelems != 1)
    {
        throw std::runtime_error("value must be a scalar tile");
    }
    int mpi_rank = starpu_mpi_world_rank();
    int value_rank = value.mpi_get_rank();
    if(mpi_rank == value_rank)
    {
        starpu::log_scalar.submit<std::tuple<T>>(name, value);
    }
}

template<typename T>
void log_scalar(const std::string &name, const Tile<T> &value)
{
    log_scalar_async<T>(name, value);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void log_scalar_async<fp32_t>(const std::string &name,
        const Tile<fp32_t> &value);

template
void log_scalar_async<fp64_t>(const std::string &name,
        const Tile<fp64_t> &value);

template
void log_scalar_async<fp32_fast_tf32_t>(const std::string &name,
        const Tile<fp32_fast_tf32_t> &value);

template
void log_scalar_async<fp32_fast_fp16_t>(const std::string &name,
        const Tile<fp32_fast_fp16_t> &value);

template
void log_scalar_async<fp32_fast_bf16_t>(const std::string &name,
        const Tile<fp32_fast_bf16_t> &value);

template
void log_scalar_async<bf16_t>(const std::string &name,
        const Tile<bf16_t> &value);

template
void log_scalar<fp32_t>(const std::string &name, const Tile<fp32_t> &value);

template
void log_scalar<fp64_t>(const std::string &name, const Tile<fp64_t> &value);

template
void log_scalar<fp32_fast_tf32_t>(const std::string &name,
        const Tile<fp32_fast_tf32_t> &value);

template
void log_scalar<fp32_fast_fp16_t>(const std::string &name,
        const Tile<fp32_fast_fp16_t> &value);

template
void log_scalar<fp32_fast_bf16_t>(const std::string &name,
        const Tile<fp32_fast_bf16_t> &value);

template
void log_scalar<bf16_t>(const std::string &name, const Tile<bf16_t> &value);

} // namespace nntile::tile
