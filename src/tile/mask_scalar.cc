/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/mask_scalar.cc
 * Mask scalar operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/mask_scalar.hh"
#include "nntile/starpu/mask_scalar.hh"

namespace nntile::tile
{

//! Asynchronous tile-wise mask scalar operation
/*! @param[inout] A: Tile for the element-wise mask scalar operation
 * */
template<typename T>
void mask_scalar_async(const Tile<bool_t> &mask, Scalar val, const Tile<T> &A)
{
    // Submit task without any arguments checked
    starpu::mask_scalar::submit<T>(A.matrix_shape[A.ndim-1][0],
            A.shape[A.ndim-1], mask, val, A);
}

//! Blocking version of tile-wise mask scalar operation
/*! @param[inout] A: Tile for the element-wise mask scalar operation
 * */
template<typename T>
void mask_scalar(const Tile<bool_t> &mask, Scalar val, const Tile<T> &A)
{
    mask_scalar_async<T>(mask, val, A);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void mask_scalar_async<fp32_t>(const Tile<bool_t> &mask, Scalar val,
        const Tile<fp32_t> &A);

template
void mask_scalar_async<fp32_fast_tf32_t>(const Tile<bool_t> &mask, Scalar val,
        const Tile<fp32_fast_tf32_t> &A);

template
void mask_scalar_async<fp64_t>(const Tile<bool_t> &mask, Scalar val,
        const Tile<fp64_t> &A);

template
void mask_scalar_async<bf16_t>(const Tile<bool_t> &mask, Scalar val,
        const Tile<bf16_t> &A);

// Explicit instantiation
template
void mask_scalar<fp32_t>(const Tile<bool_t> &mask, Scalar val,
        const Tile<fp32_t> &A);

template
void mask_scalar<fp32_fast_tf32_t>(const Tile<bool_t> &mask, Scalar val,
        const Tile<fp32_fast_tf32_t> &A);

template
void mask_scalar<fp64_t>(const Tile<bool_t> &mask, Scalar val,
        const Tile<fp64_t> &A);

template
void mask_scalar<bf16_t>(const Tile<bool_t> &mask, Scalar val,
        const Tile<bf16_t> &A);

} // namespace nntile::tile
