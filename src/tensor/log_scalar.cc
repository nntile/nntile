/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/log_scalar.cc
 * Log scalar value from Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/log_scalar.hh"
#include "nntile/starpu/log_scalar.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tensor
{

template<typename T>
void log_scalar_async(const std::string &name, const Tensor<T> &value)
{
    // Check if value is a scalar tensor
    if(value.nelems != 1)
    {
        throw std::runtime_error("value must be a scalar tensor");
    }
    // Treat special case of a source destination tile
    int mpi_rank = starpu_mpi_world_rank();
    auto value_handle = value.get_tile_handle(0);
    int value_rank = value_handle.mpi_get_rank();
    // Execute on destination node
    if(mpi_rank == value_rank)
    {
        starpu::log_scalar.submit<std::tuple<T>>(name, value_handle);
    }
}

template<typename T>
void log_scalar(const std::string &name, const Tensor<T> &value)
{
    log_scalar_async<T>(name, value);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void log_scalar_async<fp32_t>(const std::string &name,
        const Tensor<fp32_t> &value);

template
void log_scalar_async<fp64_t>(const std::string &name,
        const Tensor<fp64_t> &value);

template
void log_scalar_async<fp32_fast_tf32_t>(const std::string &name,
        const Tensor<fp32_fast_tf32_t> &value);

template
void log_scalar_async<fp32_fast_fp16_t>(const std::string &name,
        const Tensor<fp32_fast_fp16_t> &value);

template
void log_scalar_async<fp32_fast_bf16_t>(const std::string &name,
        const Tensor<fp32_fast_bf16_t> &value);

template
void log_scalar_async<bf16_t>(const std::string &name,
        const Tensor<bf16_t> &value);

template
void log_scalar<fp32_t>(const std::string &name, const Tensor<fp32_t> &value);

template
void log_scalar<fp64_t>(const std::string &name, const Tensor<fp64_t> &value);

template
void log_scalar<fp32_fast_tf32_t>(const std::string &name,
        const Tensor<fp32_fast_tf32_t> &value);

template
void log_scalar<fp32_fast_fp16_t>(const std::string &name,
        const Tensor<fp32_fast_fp16_t> &value);

template
void log_scalar<fp32_fast_bf16_t>(const std::string &name,
        const Tensor<fp32_fast_bf16_t> &value);

template
void log_scalar<bf16_t>(const std::string &name, const Tensor<bf16_t> &value);

} // namespace nntile::tensor
