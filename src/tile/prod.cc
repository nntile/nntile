/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/prod.cc
 * Per-element product of two Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-12-02
 * */

#include "nntile/tile/prod.hh"
#include "nntile/starpu/prod.hh"

namespace nntile
{
namespace tile
{

//! Asynchronous version of tile-wise prod operation
/*! @param[in] src: Input tile for element-wise prod operation
 * @param[inout] dst: Input and output tile for the prod operation
 * */
template<typename T>
void prod_async(const Tile<T> &src, const Tile<T> &dst)
{
    // Check shapes
    if(src.shape != dst.shape)
    {
        throw std::runtime_error("src.shape != dst.shape");
    }
    // Submit task
    starpu::prod::submit<T>(src.nelems, src, dst);
}

//! Blocking version of tile-wise prod operation
/*! @param[in] src: Input tile for element-wise prod operation
 * @param[inout] dst: Input and output tile for the prod operation
 * */
template<typename T>
void prod(const Tile<T> &src, const Tile<T> &dst)
{
    prod_async<T>(src, dst);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void prod_async<fp32_t>(const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void prod_async<fp64_t>(const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

// Explicit instantiation
template
void prod<fp32_t>(const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void prod<fp64_t>(const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

} // namespace tile
} // namespace nntile

