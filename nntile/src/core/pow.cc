/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/core/pow.cc
 * Power operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/core/pow.hh"
#include "nntile/starpu/pow.hh"
#include "nntile/starpu/config.hh"

namespace nntile::core
{

//! Asynchronous tile-wise power operation
/*! @param[inout] A: Tile for the element-wise power operation
 * */
template<typename T>
void pow_async(int starpu_worker_hint, Scalar alpha, Scalar exp, const Tile<T> &A)
{
    int mpi_rank = starpu_mpi_world_rank();
    int a_rank = A.mpi_get_rank();
    if(mpi_rank == a_rank)
    {
        // Submit task without any arguments checked
        starpu::pow.submit<std::tuple<T>>(starpu_worker_hint, A.nelems, alpha, exp, A);
    }
}

//! Blocking version of tile-wise power operation
/*! @param[inout] A: Tile for the element-wise power operation
 * */
template<typename T>
void pow(int starpu_worker_hint, Scalar alpha, Scalar exp, const Tile<T> &A)
{
    pow_async<T>(starpu_worker_hint, alpha, exp, A);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void pow_async<fp32_t>(int starpu_worker_hint, Scalar alpha, Scalar exp, const Tile<fp32_t> &A);

template
void pow_async<fp64_t>(int starpu_worker_hint, Scalar alpha, Scalar exp, const Tile<fp64_t> &A);

template
void pow_async<bf16_t>(int starpu_worker_hint, Scalar alpha, Scalar exp, const Tile<bf16_t> &A);

template
void pow_async<fp32_fast_tf32_t>(int starpu_worker_hint, Scalar alpha, Scalar exp,
        const Tile<fp32_fast_tf32_t> &A);

template
void pow_async<fp32_fast_fp16_t>(int starpu_worker_hint, Scalar alpha, Scalar exp,
        const Tile<fp32_fast_fp16_t> &A);

template
void pow_async<fp32_fast_bf16_t>(int starpu_worker_hint, Scalar alpha, Scalar exp,
        const Tile<fp32_fast_bf16_t> &A);

// Explicit instantiation
template
void pow<fp32_t>(int starpu_worker_hint, Scalar alpha, Scalar exp, const Tile<fp32_t> &A);

template
void pow<fp64_t>(int starpu_worker_hint, Scalar alpha, Scalar exp, const Tile<fp64_t> &A);

template
void pow<bf16_t>(int starpu_worker_hint, Scalar alpha, Scalar exp, const Tile<bf16_t> &A);

template
void pow<fp32_fast_tf32_t>(int starpu_worker_hint, Scalar alpha, Scalar exp,
        const Tile<fp32_fast_tf32_t> &A);

template
void pow<fp32_fast_fp16_t>(int starpu_worker_hint, Scalar alpha, Scalar exp,
        const Tile<fp32_fast_fp16_t> &A);

template
void pow<fp32_fast_bf16_t>(int starpu_worker_hint, Scalar alpha, Scalar exp,
        const Tile<fp32_fast_bf16_t> &A);

} // namespace nntile::core
