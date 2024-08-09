/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/maximum.cc
 * Per-element maximum of two Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/maximum.hh"
#include "nntile/starpu/maximum.hh"

namespace nntile::tile
{

//! Asynchronous version of tile-wise maximum operation
/*! @param[in] src: Input tile for element-wise maximum operation
 * @param[inout] dst: Input and output tile for the maximum operation
 * */
template<typename T>
void maximum_async(const Tile<T> &src, const Tile<T> &dst)
{
    // Check shapes
    if(src.shape != dst.shape)
    {
        throw std::runtime_error("src.shape != dst.shape");
    }
    // Submit task
    starpu::maximum::submit<T>(src.nelems, src, dst);
}

//! Blocking version of tile-wise maximum operation
/*! @param[in] src: Input tile for element-wise maximum operation
 * @param[inout] dst: Input and output tile for the maximum operation
 * */
template<typename T>
void maximum(const Tile<T> &src, const Tile<T> &dst)
{
    maximum_async<T>(src, dst);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void maximum_async<fp32_t>(const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void maximum_async<fp64_t>(const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

// Explicit instantiation
template
void maximum<fp32_t>(const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void maximum<fp64_t>(const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

} // namespace nntile::tile
