/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/multiply_inplace.cc
 * Per-element multiplication of two Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/multiply_inplace.hh"
#include "nntile/starpu/multiply_inplace.hh"

namespace nntile::tile
{

//! Asynchronous version of tile-wise multiply operation
/*! @param[in] src: Input tile for element-wise multiply operation
 * @param[inout] dst: Input and output tile for the multiply operation
 * */
template<typename T>
void multiply_inplace_async(const Tile<T> &src, const Tile<T> &dst)
{
    // Check shapes
    if(src.shape != dst.shape)
    {
        throw std::runtime_error("src.shape != dst.shape");
    }
    // Submit task
    starpu::multiply_inplace.submit<std::tuple<T>>(src.nelems, src, dst);
}

//! Blocking version of tile-wise multiply operation
/*! @param[in] src: Input tile for element-wise multiply operation
 * @param[inout] dst: Input and output tile for the multiply operation
 * */
template<typename T>
void multiply_inplace(const Tile<T> &src, const Tile<T> &dst)
{
    multiply_inplace_async<T>(src, dst);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void multiply_inplace_async<fp32_t>(const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst);

template
void multiply_inplace_async<fp32_fast_tf32_t>(const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst);

template
void multiply_inplace_async<fp32_fast_fp16_t>(const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst);

template
void multiply_inplace_async<fp32_fast_bf16_t>(const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst);

template
void multiply_inplace_async<fp64_t>(const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst);

template
void multiply_inplace_async<bf16_t>(const Tile<bf16_t> &src,
        const Tile<bf16_t> &dst);

template
void multiply_inplace_async<fp16_t>(const Tile<fp16_t> &src, const Tile<fp16_t> &dst);

// Explicit instantiation
template
void multiply_inplace<fp32_t>(const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void multiply_inplace<fp32_fast_tf32_t>(const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst);

template
void multiply_inplace<fp32_fast_fp16_t>(const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst);

template
void multiply_inplace<fp32_fast_bf16_t>(const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst);

template
void multiply_inplace<fp64_t>(const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

template
void multiply_inplace<bf16_t>(const Tile<bf16_t> &src, const Tile<bf16_t> &dst);

template
void multiply_inplace<fp16_t>(const Tile<fp16_t> &src, const Tile<fp16_t> &dst);

} // namespace nntile::tile
