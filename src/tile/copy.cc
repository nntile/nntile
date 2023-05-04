/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/copy.cc
 * Copy one tile into another matching tile
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @author Aleksandr Katrutsa
 * @date 2023-05-04
 * */

#include "nntile/tile/copy.hh"
#include "nntile/starpu/copy.hh"
#include <starpu.h>

namespace nntile
{
namespace tile
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
    // Submit copy procedure
    starpu::copy::submit(src, dst);
    //int ret = starpu_data_cpy(static_cast<starpu_data_handle_t>(dst),
    //        static_cast<starpu_data_handle_t>(src), 1, nullptr, nullptr);
    //if(ret != 0)
    //{
    //    throw std::runtime_error("Error in starpu_data_cpy");
    //}
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
void copy_async<Index>(const Tile<Index> &src, const Tile<Index> &dst);

// Explicit instantiation
template
void copy<fp32_t>(const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void copy<fp64_t>(const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

template
void copy<Index>(const Tile<Index> &src, const Tile<Index> &dst);

} // namespace tile
} // namespace nntile

