/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/hypot_scalar_inverse.cc
 * hypot_scalar_inverse operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/hypot_scalar_inverse.hh"
#include "nntile/starpu/hypot_scalar_inverse.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tile
{

template<typename T>
void hypot_scalar_inverse_async(Scalar eps, Scalar alpha, const Tile<T> &dst)
{
    int mpi_rank = starpu_mpi_world_rank();
    int dst_rank = dst.mpi_get_rank();
    if(mpi_rank == dst_rank)
    {
        starpu::hypot_scalar_inverse.submit<std::tuple<T>>(dst.nelems, eps,
                alpha, dst);
    }
}

template<typename T>
void hypot_scalar_inverse(Scalar eps, Scalar alpha, const Tile<T> &dst)
{
    hypot_scalar_inverse_async<T>(eps, alpha, dst);
    starpu_task_wait_for_all();
}

// Explicit instantiation of template
template
void hypot_scalar_inverse_async<fp32_t>(Scalar eps, Scalar alpha,
        const Tile<fp32_t> &dst);

template
void hypot_scalar_inverse_async<bf16_t>(Scalar eps, Scalar alpha,
        const Tile<bf16_t> &dst);

template
void hypot_scalar_inverse_async<fp32_fast_tf32_t>(Scalar eps, Scalar alpha,
        const Tile<fp32_fast_tf32_t> &dst);

template
void hypot_scalar_inverse_async<fp32_fast_fp16_t>(Scalar eps, Scalar alpha,
        const Tile<fp32_fast_fp16_t> &dst);

template
void hypot_scalar_inverse_async<fp32_fast_bf16_t>(Scalar eps, Scalar alpha,
        const Tile<fp32_fast_bf16_t> &dst);

template
void hypot_scalar_inverse_async<fp64_t>(Scalar eps, Scalar alpha,
        const Tile<fp64_t> &dst);

template
void hypot_scalar_inverse_async<fp16_t>(Scalar eps, Scalar alpha,
        const Tile<fp16_t> &dst);

// Explicit instantiation of template
template
void hypot_scalar_inverse<fp32_t>(Scalar eps, Scalar alpha,
        const Tile<fp32_t> &dst);

template
void hypot_scalar_inverse<fp32_fast_tf32_t>(Scalar eps, Scalar alpha,
        const Tile<fp32_fast_tf32_t> &dst);

template
void hypot_scalar_inverse<fp32_fast_fp16_t>(Scalar eps, Scalar alpha,
        const Tile<fp32_fast_fp16_t> &dst);

template
void hypot_scalar_inverse<fp32_fast_bf16_t>(Scalar eps, Scalar alpha,
        const Tile<fp32_fast_bf16_t> &dst);

template
void hypot_scalar_inverse<fp64_t>(Scalar eps, Scalar alpha,
        const Tile<fp64_t> &dst);

template
void hypot_scalar_inverse<bf16_t>(Scalar eps, Scalar alpha,
        const Tile<bf16_t> &dst);

template
void hypot_scalar_inverse<fp16_t>(Scalar eps, Scalar alpha,
        const Tile<fp16_t> &dst);

} // namespace nntile::tile
