/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/scale_inplace.cc
 * Inplace scale of Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/scale_inplace.hh"
#include "nntile/starpu/scale_inplace.hh"

namespace nntile::tile
{

//! Tile-wise scale_inplace operation
template<typename T>
void scale_inplace_async(Scalar alpha, const Tile<T> &data)
{
    // Insert task
    starpu::scale_inplace.submit<std::tuple<T>>(data.nelems, alpha, data);
}

//! Tile-wise scale_inplace operation
template<typename T>
void scale_inplace(Scalar alpha, const Tile<T> &data)
{
    scale_inplace_async<T>(alpha, data);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void scale_inplace_async<fp32_t>(Scalar alpha, const Tile<fp32_t> &data);

template
void scale_inplace_async<fp32_fast_tf32_t>(Scalar alpha, const Tile<fp32_fast_tf32_t> &data);

template
void scale_inplace_async<fp32_fast_fp16_t>(Scalar alpha, const Tile<fp32_fast_fp16_t> &data);

template
void scale_inplace_async<fp64_t>(Scalar alpha, const Tile<fp64_t> &data);

template
void scale_inplace_async<fp16_t>(Scalar alpha, const Tile<fp16_t> &data);

template
void scale_inplace_async<bf16_t>(Scalar alpha, const Tile<bf16_t> &data);

// Explicit instantiation
template
void scale_inplace<fp32_t>(Scalar alpha, const Tile<fp32_t> &data);

template
void scale_inplace<fp32_fast_tf32_t>(Scalar alpha, const Tile<fp32_fast_tf32_t> &data);

template
void scale_inplace<fp32_fast_fp16_t>(Scalar alpha, const Tile<fp32_fast_fp16_t> &data);

template
void scale_inplace<fp64_t>(Scalar alpha, const Tile<fp64_t> &data);

template
void scale_inplace<fp16_t>(Scalar alpha, const Tile<fp16_t> &data);

template
void scale_inplace<bf16_t>(Scalar alpha, const Tile<bf16_t> &data);

} // namespace nntile::tile
