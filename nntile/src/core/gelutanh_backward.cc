/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/core/gelutanh_backward.cc
 * Backward approximate GeLU operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/core/gelutanh_backward.hh"
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/ops/gelutanh_backward.hh"
#else
#include "nntile/starpu/gelutanh_backward.hh"
#endif
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/sync_defer.hh"
#else
#include "nntile/starpu/config.hh"
#endif

namespace nntile::core
{

//! Asynchronous tile-wise backward approximate GeLU operation
/*! @param[inout] A: Tile for the element-wise backward approximate GeLU operation
 * */
template<typename T>
void gelutanh_backward_async(int starpu_worker_hint, Scalar alpha, const Tile<T> &x, const Tile<T> &dy,
        Scalar beta, const Tile<T> &dx)
{
    // Check shapes
    if(x.shape != dy.shape)
    {
        throw std::runtime_error("x.shape != dy.shape");
    }
    if(x.shape != dx.shape)
    {
        throw std::runtime_error("x.shape != dx.shape");
    }
    #ifdef NNTILE_USE_NNHAUL
    int mpi_rank = 0;
    #else
    int mpi_rank = starpu_mpi_world_rank();
    #endif
    #ifdef NNTILE_USE_NNHAUL
    int dx_rank = 0;
    #else
    int dx_rank = dx.mpi_get_rank();
    #endif
    #ifndef NNTILE_USE_NNHAUL
    x.mpi_transfer(dx_rank, mpi_rank);
    #endif
    #ifndef NNTILE_USE_NNHAUL
    dy.mpi_transfer(dx_rank, mpi_rank);
    #endif
    if(mpi_rank == dx_rank)
    {
        // Submit task without any arguments checked
        #ifdef NNTILE_USE_NNHAUL
        haul::gelutanh_backward.submit<std::tuple<T>>(starpu_worker_hint, x.nelems, alpha, x, dy, beta, dx);
        #else
        starpu::gelutanh_backward.submit<std::tuple<T>>(starpu_worker_hint, x.nelems, alpha, x, dy, beta, dx);
        #endif

    }
}

//! Blocking version of tile-wise backward approximate GeLU operation
/*! @param[inout] A: Tile for the element-wise backward approximate GeLU operation
 * */
template<typename T>
void gelutanh_backward(int starpu_worker_hint, Scalar alpha, const Tile<T> &x, const Tile<T> &dy,
        Scalar beta, const Tile<T> &dx)
{
    gelutanh_backward_async<T>(starpu_worker_hint, alpha, x, dy, beta, dx);
    #ifdef NNTILE_USE_NNHAUL
    nntile::nnhaul_task_wait_for_all_unless_deferred();
#else
    nntile::starpu_task_wait_for_all_unless_deferred();
#endif
}

// Explicit instantiation
template
void gelutanh_backward_async<fp32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_t> &x, const Tile<fp32_t> &dy,
        Scalar beta, const Tile<fp32_t> &dx);

template
void gelutanh_backward_async<fp32_fast_tf32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_tf32_t> &x, const Tile<fp32_fast_tf32_t> &dy,
        Scalar beta, const Tile<fp32_fast_tf32_t> &dx);

template
void gelutanh_backward_async<fp32_fast_fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_fp16_t> &x, const Tile<fp32_fast_fp16_t> &dy,
        Scalar beta, const Tile<fp32_fast_fp16_t> &dx);

template
void gelutanh_backward_async<fp32_fast_bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_bf16_t> &x, const Tile<fp32_fast_bf16_t> &dy,
        Scalar beta, const Tile<fp32_fast_bf16_t> &dx);

template
void gelutanh_backward_async<fp64_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp64_t> &x, const Tile<fp64_t> &dy,
        Scalar beta, const Tile<fp64_t> &dx);

template
void gelutanh_backward_async<bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<bf16_t> &x, const Tile<bf16_t> &dy,
        Scalar beta, const Tile<bf16_t> &dx);

template
void gelutanh_backward_async<fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp16_t> &x, const Tile<fp16_t> &dy,
        Scalar beta, const Tile<fp16_t> &dx);

// Explicit instantiation
template
void gelutanh_backward<fp32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_t> &x, const Tile<fp32_t> &dy,
        Scalar beta, const Tile<fp32_t> &dx);

template
void gelutanh_backward<fp32_fast_tf32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_tf32_t> &x, const Tile<fp32_fast_tf32_t> &dy,
        Scalar beta, const Tile<fp32_fast_tf32_t> &dx);

template
void gelutanh_backward<fp32_fast_fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_fp16_t> &x, const Tile<fp32_fast_fp16_t> &dy,
        Scalar beta, const Tile<fp32_fast_fp16_t> &dx);

template
void gelutanh_backward<fp32_fast_bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_bf16_t> &x, const Tile<fp32_fast_bf16_t> &dy,
        Scalar beta, const Tile<fp32_fast_bf16_t> &dx);

template
void gelutanh_backward<fp64_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp64_t> &x, const Tile<fp64_t> &dy,
        Scalar beta, const Tile<fp64_t> &dx);

template
void gelutanh_backward<bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<bf16_t> &x, const Tile<bf16_t> &dy,
        Scalar beta, const Tile<bf16_t> &dx);

template
void gelutanh_backward<fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp16_t> &x, const Tile<fp16_t> &dy,
        Scalar beta, const Tile<fp16_t> &dx);

} // namespace nntile::core
