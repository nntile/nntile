/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/max.cc
 * Per-element maximum of two Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-02-10
 * */

#include "nntile/tile/max.hh"
#include "nntile/starpu/max.hh"

namespace nntile
{
namespace tile
{

//! Asynchronous version of tile-wise maximum operation
/*! @param[in] src: Input tile for element-wise maximum operation
 * @param[inout] dst: Input and output tile for the maximum operation
 * */
template<typename T>
void max_async(const Tile<T> &src, const Tile<T> &dst)
{
    // Check shapes
    if(src.shape != dst.shape)
    {
        throw std::runtime_error("src.shape != dst.shape");
    }
    // Submit task
    starpu::max::submit<T>(src.nelems, src, dst);
}

//! Blocking version of tile-wise maximum operation
/*! @param[in] src: Input tile for element-wise maximum operation
 * @param[inout] dst: Input and output tile for the maximum operation
 * */
template<typename T>
void max(const Tile<T> &src, const Tile<T> &dst)
{
    max_async<T>(src, dst);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void max_async<fp32_t>(const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void max_async<fp64_t>(const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

// Explicit instantiation
template
void max<fp32_t>(const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void max<fp64_t>(const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

} // namespace tile
} // namespace nntile
