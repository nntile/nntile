/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/clear.cc
 * Clear Tensor<T>
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/tensor/clear.hh"

// Other NNTile headers
#include "nntile/tile/clear.hh"
#include "nntile/starpu/config.hh"

// Namespace
namespace nntile::tensor
{

template<typename T>
void clear_async(const Tensor<T> &dst)
{
    for(Index i = 0; i < dst.grid.nelems; ++i)
    {
        auto dst_tile_handle = dst.get_tile_handle(i);
        auto dst_tile = dst.get_tile(i);
        tile::clear_async<T>(dst_tile);
        // Flush cache for the output tile on every node
        dst_tile_handle.mpi_flush();
    }
}

template<typename T>
void clear(const Tensor<T> &dst)
{
    clear_async<T>(dst);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void clear_async<int64_t>(const Tensor<int64_t> &dst);

template
void clear_async<bool_t>(const Tensor<bool_t> &dst);

template
void clear_async<fp32_t>(const Tensor<fp32_t> &dst);

template
void clear_async<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &dst);

template
void clear_async<fp32_fast_fp16_t>(const Tensor<fp32_fast_fp16_t> &dst);

template
void clear_async<fp32_fast_bf16_t>(const Tensor<fp32_fast_bf16_t> &dst);

template
void clear_async<fp64_t>(const Tensor<fp64_t> &dst);

template
void clear_async<bf16_t>(const Tensor<bf16_t> &dst);

template
void clear_async<fp16_t>(const Tensor<fp16_t> &dst);

// Explicit instantiation
template
void clear<int64_t>(const Tensor<int64_t> &dst);

template
void clear<bool_t>(const Tensor<bool_t> &dst);

template
void clear<fp32_t>(const Tensor<fp32_t> &dst);

template
void clear<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &dst);

template
void clear<fp32_fast_fp16_t>(const Tensor<fp32_fast_fp16_t> &dst);

template
void clear<fp32_fast_bf16_t>(const Tensor<fp32_fast_bf16_t> &dst);

template
void clear<fp64_t>(const Tensor<fp64_t> &dst);

template
void clear<bf16_t>(const Tensor<bf16_t> &dst);

template
void clear<fp16_t>(const Tensor<fp16_t> &dst);

} // namespace nntile::tensor
