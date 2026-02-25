/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/copy.cc
 * Copy one tile into another matching tile
 *
 * @version 1.1.0
 * */

#include "nntile/tile/copy.hh"
#include "nntile/starpu/copy.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tile
{

//! Asynchronous version of tile-wise copy operation
/*! A simple copy from one tile into another
 *
 * @param[in] src: Source tile
 * @param[inout] dst: Destination tile
 * */
template<typename T>
void copy_async(const Tile<T> &src, const Tile<T> &dst)
{
    // Check shapes
    if(src.shape != dst.shape)
    {
        throw std::runtime_error("src.shape != dst.shape");
    }
    int mpi_rank = starpu_mpi_world_rank();
    int dst_rank = dst.mpi_get_rank();
    src.mpi_transfer(dst_rank, mpi_rank);
    if(mpi_rank == dst_rank)
    {
        starpu::copy.submit(src, dst);
    }
}

//! Blocking version of tile-wise copy operation
/*! A simple copy from one tile into another
 *
 * @param[in] src: Source tile
 * @param[inout] dst: Destination tile
 * */
template<typename T>
void copy(const Tile<T> &src, const Tile<T> &dst)
{
    copy_async<T>(src, dst);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void copy_async<fp32_t>(const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void copy_async<fp64_t>(const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

template
void copy_async<int64_t>(const Tile<int64_t> &src, const Tile<int64_t> &dst);

template
void copy_async<bf16_t>(const Tile<bf16_t> &src, const Tile<bf16_t> &dst);

template
void copy_async<fp16_t>(const Tile<fp16_t> &src, const Tile<fp16_t> &dst);

template
void copy_async<fp32_fast_tf32_t>(const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst);

template
void copy_async<fp32_fast_fp16_t>(const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst);

template
void copy_async<fp32_fast_bf16_t>(const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst);

template
void copy_async<bool_t>(const Tile<bool_t> &src, const Tile<bool_t> &dst);

// Explicit instantiation
template
void copy<fp32_t>(const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void copy<fp64_t>(const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

template
void copy<int64_t>(const Tile<int64_t> &src, const Tile<int64_t> &dst);

template
void copy<bf16_t>(const Tile<bf16_t> &src, const Tile<bf16_t> &dst);

template
void copy<fp16_t>(const Tile<fp16_t> &src, const Tile<fp16_t> &dst);

template
void copy<fp32_fast_tf32_t>(const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst);

template
void copy<fp32_fast_fp16_t>(const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst);

template
void copy<fp32_fast_bf16_t>(const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst);

template
void copy<bool_t>(const Tile<bool_t> &src, const Tile<bool_t> &dst);

} // namespace nntile::tile
