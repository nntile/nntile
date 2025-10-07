/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/hypot.cc
 * hypot operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/hypot.hh"
#include "nntile/starpu/hypot.hh"

namespace nntile::tile
{

//! Tile-wise hypot operation
template<typename T>
void hypot_async(Scalar alpha, const Tile<T> &src1, Scalar beta, const Tile<T> &src2, const Tile<T> &dst)
{
    // Check dimensions
    if(dst.ndim != src1.ndim || dst.ndim != src2.ndim)
    {
        throw std::runtime_error("dst.ndim != src1.ndim or dst.ndim != src2.ndim");
    }
    // Check shapes of tiles
    for(Index i = 0; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != src1.shape[i] || dst.shape[i] != src2.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != src1.shape[i] or dst.shape[i] != src2.shape[i]");
        }
    }
    // Insert corresponding task
    starpu::hypot.submit<std::tuple<T>>(src1.nelems, alpha, src1, beta, src2, dst);
}

//! Tile-wise hypot operation
template<typename T>
void hypot(Scalar alpha, const Tile<T> &src1, Scalar beta, const Tile<T> &src2, const Tile<T> &dst)
{
    hypot_async<T>(alpha, src1, beta, src2, dst);
    starpu_task_wait_for_all();
}

// Explicit instantiation of template
template
void hypot_async<fp32_t>(Scalar alpha, const Tile<fp32_t> &src1, Scalar beta,
        const Tile<fp32_t> &src2, const Tile<fp32_t> &dst);

template
void hypot_async<fp32_fast_tf32_t>(Scalar alpha, const Tile<fp32_fast_tf32_t> &src1, Scalar beta,
        const Tile<fp32_fast_tf32_t> &src2, const Tile<fp32_fast_tf32_t> &dst);

template
void hypot_async<fp32_fast_fp16_t>(Scalar alpha, const Tile<fp32_fast_fp16_t> &src1, Scalar beta,
        const Tile<fp32_fast_fp16_t> &src2, const Tile<fp32_fast_fp16_t> &dst);

template
void hypot_async<fp32_fast_bf16_t>(Scalar alpha, const Tile<fp32_fast_bf16_t> &src1, Scalar beta,
        const Tile<fp32_fast_bf16_t> &src2, const Tile<fp32_fast_bf16_t> &dst);

template
void hypot_async<fp64_t>(Scalar alpha, const Tile<fp64_t> &src1, Scalar beta,
        const Tile<fp64_t> &src2, const Tile<fp64_t> &dst);

template
void hypot_async<bf16_t>(Scalar alpha, const Tile<bf16_t> &src1, Scalar beta,
        const Tile<bf16_t> &src2, const Tile<bf16_t> &dst);

template
void hypot_async<fp16_t>(Scalar alpha, const Tile<fp16_t> &src1, Scalar beta,
        const Tile<fp16_t> &src2, const Tile<fp16_t> &dst);

// Explicit instantiation of template
template
void hypot<fp32_t>(Scalar alpha, const Tile<fp32_t> &src1, Scalar beta,
        const Tile<fp32_t> &src2, const Tile<fp32_t> &dst);

template
void hypot<fp32_fast_tf32_t>(Scalar alpha, const Tile<fp32_fast_tf32_t> &src1, Scalar beta,
        const Tile<fp32_fast_tf32_t> &src2, const Tile<fp32_fast_tf32_t> &dst);

template
void hypot<fp32_fast_fp16_t>(Scalar alpha, const Tile<fp32_fast_fp16_t> &src1, Scalar beta,
        const Tile<fp32_fast_fp16_t> &src2, const Tile<fp32_fast_fp16_t> &dst);

template
void hypot<fp32_fast_bf16_t>(Scalar alpha, const Tile<fp32_fast_bf16_t> &src1, Scalar beta,
        const Tile<fp32_fast_bf16_t> &src2, const Tile<fp32_fast_bf16_t> &dst);

template
void hypot<fp64_t>(Scalar alpha, const Tile<fp64_t> &src1, Scalar beta,
        const Tile<fp64_t> &src2, const Tile<fp64_t> &dst);

template
void hypot<bf16_t>(Scalar alpha, const Tile<bf16_t> &src1, Scalar beta,
        const Tile<bf16_t> &src2, const Tile<bf16_t> &dst);

template
void hypot<fp16_t>(Scalar alpha, const Tile<fp16_t> &src1, Scalar beta,
        const Tile<fp16_t> &src2, const Tile<fp16_t> &dst);

} // namespace nntile::tile
