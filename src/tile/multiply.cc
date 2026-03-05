/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/multiply.cc
 * Per-element product of two Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/multiply.hh"
#include "nntile/starpu/multiply.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tile
{

//! Asynchronous tile-wise multiply operation
template<typename T>
void multiply_async(Scalar alpha, const Tile<T> &src1, const Tile<T> &src2,
        const Tile<T> &dst)
{
    // Check shapes
    if(src1.shape != src2.shape)
    {
        throw std::runtime_error("src1.shape != src2.shape");
    }
    if(src1.shape != dst.shape)
    {
        throw std::runtime_error("src1.shape != dst.shape");
    }
    int mpi_rank = starpu_mpi_world_rank();
    int dst_rank = dst.mpi_get_rank();
    src1.mpi_transfer(dst_rank, mpi_rank);
    src2.mpi_transfer(dst_rank, mpi_rank);
    if(mpi_rank == dst_rank)
    {
        starpu::multiply.submit<std::tuple<T>>(src1.nelems, alpha, src1, src2,
                dst);
    }
}

//! Blocking version of tile-wise multiply operation
template<typename T>
void multiply(Scalar alpha, const Tile<T> &src1, const Tile<T> &src2,
        const Tile<T> &dst)
{
    multiply_async<T>(alpha, src1, src2, dst);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void multiply_async<fp32_t>(Scalar alpha, const Tile<fp32_t> &src1,
        const Tile<fp32_t> &src2, const Tile<fp32_t> &dst);

template
void multiply_async<fp32_fast_tf32_t>(Scalar alpha,
        const Tile<fp32_fast_tf32_t> &src1,
        const Tile<fp32_fast_tf32_t> &src2,
        const Tile<fp32_fast_tf32_t> &dst);

template
void multiply_async<fp32_fast_fp16_t>(Scalar alpha,
        const Tile<fp32_fast_fp16_t> &src1,
        const Tile<fp32_fast_fp16_t> &src2,
        const Tile<fp32_fast_fp16_t> &dst);

template
void multiply_async<fp32_fast_bf16_t>(Scalar alpha,
        const Tile<fp32_fast_bf16_t> &src1,
        const Tile<fp32_fast_bf16_t> &src2,
        const Tile<fp32_fast_bf16_t> &dst);

template
void multiply_async<fp64_t>(Scalar alpha, const Tile<fp64_t> &src1,
        const Tile<fp64_t> &src2, const Tile<fp64_t> &dst);

template
void multiply_async<bf16_t>(Scalar alpha, const Tile<bf16_t> &src1,
        const Tile<bf16_t> &src2, const Tile<bf16_t> &dst);

template
void multiply_async<fp16_t>(Scalar alpha, const Tile<fp16_t> &src1,
        const Tile<fp16_t> &src2, const Tile<fp16_t> &dst);

// Explicit instantiation
template
void multiply<fp32_t>(Scalar alpha, const Tile<fp32_t> &src1,
        const Tile<fp32_t> &src2, const Tile<fp32_t> &dst);

template
void multiply<fp32_fast_tf32_t>(Scalar alpha,
        const Tile<fp32_fast_tf32_t> &src1,
        const Tile<fp32_fast_tf32_t> &src2,
        const Tile<fp32_fast_tf32_t> &dst);

template
void multiply<fp32_fast_fp16_t>(Scalar alpha,
        const Tile<fp32_fast_fp16_t> &src1,
        const Tile<fp32_fast_fp16_t> &src2,
        const Tile<fp32_fast_fp16_t> &dst);

template
void multiply<fp32_fast_bf16_t>(Scalar alpha,
        const Tile<fp32_fast_bf16_t> &src1,
        const Tile<fp32_fast_bf16_t> &src2,
        const Tile<fp32_fast_bf16_t> &dst);

template
void multiply<fp64_t>(Scalar alpha, const Tile<fp64_t> &src1,
        const Tile<fp64_t> &src2, const Tile<fp64_t> &dst);

template
void multiply<bf16_t>(Scalar alpha, const Tile<bf16_t> &src1,
        const Tile<bf16_t> &src2, const Tile<bf16_t> &dst);

template
void multiply<fp16_t>(Scalar alpha, const Tile<fp16_t> &src1,
        const Tile<fp16_t> &src2, const Tile<fp16_t> &dst);

} // namespace nntile::tile
