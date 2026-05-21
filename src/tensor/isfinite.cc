/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/isfinite.cc
 * Check NaN or Inf values for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/isfinite.hh"
#include "nntile/tile/isfinite.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tensor
{

//! Asynchronous tensor-wise check NaN or Inf values
//
// @param[inout] A: Tensor for check NaN or Inf values
// @param[inout] flag: Tensor to store the result of checks
template<typename T>
void isfinite_async(const Tensor<T> &A, const Tensor<bool_t> &flag)
{
    // Check dimensions
    if(flag.ndim != 0)
    {
        throw std::runtime_error("flag.ndim != 0");
    }
    if(A.nelems == 0)
    {
        throw std::runtime_error("A.nelems == 0");
    }
    auto flag_tile = flag.get_tile(0);
    auto flag_tile_handle = flag.get_tile_handle(0);
    for(Index i = 0; i < A.grid.nelems; ++i)
    {
        auto tile_handle = A.get_tile_handle(i);
        auto tile = A.get_tile(i);
        tile::isfinite_async<T>(tile, flag_tile);
    }
    flag_tile_handle.mpi_flush();
}

//! Blocking version of tensor-wise check NaN or Inf values
//
// @param[inout] A: Tensor for the element-wise check NaN or Inf values
// @param[inout] flag: Tensor to store the result of checks
template<typename T>
void isfinite(const Tensor<T> &A, const Tensor<bool_t> &flag)
{
    isfinite_async<T>(A, flag);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void isfinite_async<fp32_t>(const Tensor<fp32_t> &A, const Tensor<bool_t> &flag);

template
void isfinite_async<fp64_t>(const Tensor<fp64_t> &A, const Tensor<bool_t> &flag);

template
void isfinite_async<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &A, const Tensor<bool_t> &flag);

template
void isfinite_async<fp32_fast_fp16_t>(const Tensor<fp32_fast_fp16_t> &A, const Tensor<bool_t> &flag);

template
void isfinite_async<fp32_fast_bf16_t>(const Tensor<fp32_fast_bf16_t> &A, const Tensor<bool_t> &flag);

template
void isfinite_async<bf16_t>(const Tensor<bf16_t> &A, const Tensor<bool_t> &flag);

template
void isfinite_async<fp16_t>(const Tensor<fp16_t> &A, const Tensor<bool_t> &flag);


// Explicit instantiation
template
void isfinite<fp32_t>(const Tensor<fp32_t> &A, const Tensor<bool_t> &flag);

template
void isfinite<fp64_t>(const Tensor<fp64_t> &A, const Tensor<bool_t> &flag);

template
void isfinite<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &A, const Tensor<bool_t> &flag);

template
void isfinite<fp32_fast_fp16_t>(const Tensor<fp32_fast_fp16_t> &A, const Tensor<bool_t> &flag);

template
void isfinite<fp32_fast_bf16_t>(const Tensor<fp32_fast_bf16_t> &A, const Tensor<bool_t> &flag);

template
void isfinite<bf16_t>(const Tensor<bf16_t> &A, const Tensor<bool_t> &flag);

template
void isfinite<fp16_t>(const Tensor<fp16_t> &A, const Tensor<bool_t> &flag);

} // namespace nntile::tensor
