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
#include <starpu.h>

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
    Index ndim = src.ndim;
    // Submit copy procedure
    int ret = starpu_data_cpy(static_cast<starpu_data_handle_t>(dst),
            static_cast<starpu_data_handle_t>(src), 1, nullptr, nullptr);
    if(ret != 0)
    {
        throw std::runtime_error("Error in starpu_data_cpy");
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

// Explicit instantiation
template
void copy<fp32_t>(const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void copy<fp64_t>(const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

template
void copy<int64_t>(const Tile<int64_t> &src, const Tile<int64_t> &dst);

template
void copy<bf16_t>(const Tile<bf16_t> &src, const Tile<bf16_t> &dst);

} // namespace nntile::tile
