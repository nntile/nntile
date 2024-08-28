/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/gelutanh.cc
 * Approximate GeLU operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/gelutanh.hh"
#include "nntile/starpu/gelutanh.hh"

namespace nntile::tile
{

//! Asyncrhonous tile-wise approximate GeLU operation
/*! @param[inout] A: Tile for the element-wise GeLU operation
 * */
template<typename T>
void gelutanh_async(const Tile<T> &src, const Tile<T> &dst)
{
    // Submit task without any arguments checked
    starpu::gelutanh::submit<T>(src.nelems, src, dst);
}

//! Blocking version of tile-wise approximate GeLU operation
/*! @param[inout] A: Tile for the element-wise GeLU operation
 * */
template<typename T>
void gelutanh(const Tile<T> &src, const Tile<T> &dst)
{
    gelutanh_async<T>(src, dst);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void gelutanh_async<fp32_t>(const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void gelutanh_async<fp32_fast_tf32_t>(const Tile<fp32_fast_tf32_t> &src, const Tile<fp32_fast_tf32_t> &dst);

template
void gelutanh_async<fp64_t>(const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

template
void gelutanh_async<bf16_t>(const Tile<bf16_t> &src, const Tile<bf16_t> &dst);

// Explicit instantiation
template
void gelutanh<fp32_t>(const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void gelutanh<fp32_fast_tf32_t>(const Tile<fp32_fast_tf32_t> &src, const Tile<fp32_fast_tf32_t> &dst);

template
void gelutanh<fp64_t>(const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

template
void gelutanh<bf16_t>(const Tile<bf16_t> &src, const Tile<bf16_t> &dst);

} // namespace nntile::tile
