/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/hypot.cc
 * hypot operation for Tensor<T>'s
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/hypot.hh"
#include "nntile/tile/hypot.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tensor
{

//! Tensor-wise hypot operation
template<typename T>
void hypot_async(Scalar alpha, const Tensor<T> &src1, Scalar beta, const Tensor<T> &src2, const Tensor<T> &dst)
{
    // Check dimensions
    if(dst.ndim != src1.ndim || dst.ndim != src2.ndim)
    {
        throw std::runtime_error("dst.ndim != src1.ndim or dst.ndim != src2.ndim");
    }
    // Check shapes of tensors
    for(Index i = 0; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != src1.shape[i] || dst.shape[i] != src2.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != src1.shape[i] or dst.shape[i] != src2.shape[i]");
        }
        if(dst.basetile_shape[i] != src1.basetile_shape[i] ||
           dst.basetile_shape[i] != src2.basetile_shape[i])
        {
            throw std::runtime_error("dst.basetile_shape[i] != "
                    "src1.basetile_shape[i] or dst.basetile_shape[i] != "
                    "src2.basetile_shape[i]");
        }
    }
    // Apply per-tile hypot asynchronously as needed
    for(Index i = 0; i < src1.grid.nelems; ++i)
    {
        auto dst_tile_handle = dst.get_tile_handle(i);
        auto src1_tile = src1.get_tile(i);
        auto src2_tile = src2.get_tile(i);
        auto dst_tile = dst.get_tile(i);
        tile::hypot_async<T>(alpha, src1_tile, beta, src2_tile, dst_tile);
        // Flush cache for the output tile on every node
        dst_tile_handle.mpi_flush();
    }
}

//! Tensor-wise hypot operation
template<typename T>
void hypot(Scalar alpha, const Tensor<T> &src1, Scalar beta, const Tensor<T> &src2, const Tensor<T> &dst)
{
    hypot_async<T>(alpha, src1, beta, src2, dst);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation of template
template
void hypot_async<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src1, Scalar beta,
        const Tensor<fp32_t> &src2, const Tensor<fp32_t> &dst);

template
void hypot_async<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src1, Scalar beta,
        const Tensor<fp32_fast_tf32_t> &src2, const Tensor<fp32_fast_tf32_t> &dst);

template
void hypot_async<fp32_fast_fp16_t>(Scalar alpha, const Tensor<fp32_fast_fp16_t> &src1, Scalar beta,
        const Tensor<fp32_fast_fp16_t> &src2, const Tensor<fp32_fast_fp16_t> &dst);

template
void hypot_async<fp32_fast_bf16_t>(Scalar alpha, const Tensor<fp32_fast_bf16_t> &src1, Scalar beta,
        const Tensor<fp32_fast_bf16_t> &src2, const Tensor<fp32_fast_bf16_t> &dst);

template
void hypot_async<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src1, Scalar beta,
        const Tensor<fp64_t> &src2, const Tensor<fp64_t> &dst);

template
void hypot_async<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src1, Scalar beta,
        const Tensor<bf16_t> &src2, const Tensor<bf16_t> &dst);

template
void hypot_async<fp16_t>(Scalar alpha, const Tensor<fp16_t> &src1, Scalar beta,
        const Tensor<fp16_t> &src2, const Tensor<fp16_t> &dst);

// Explicit instantiation of template
template
void hypot<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src1, Scalar beta,
        const Tensor<fp32_t> &src2, const Tensor<fp32_t> &dst);

template
void hypot<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src1, Scalar beta,
        const Tensor<fp32_fast_tf32_t> &src2, const Tensor<fp32_fast_tf32_t> &dst);

template
void hypot<fp32_fast_fp16_t>(Scalar alpha, const Tensor<fp32_fast_fp16_t> &src1, Scalar beta,
        const Tensor<fp32_fast_fp16_t> &src2, const Tensor<fp32_fast_fp16_t> &dst);

template
void hypot<fp32_fast_bf16_t>(Scalar alpha, const Tensor<fp32_fast_bf16_t> &src1, Scalar beta,
        const Tensor<fp32_fast_bf16_t> &src2, const Tensor<fp32_fast_bf16_t> &dst);

template
void hypot<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src1, Scalar beta,
        const Tensor<fp64_t> &src2, const Tensor<fp64_t> &dst);

template
void hypot<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src1, Scalar beta,
        const Tensor<bf16_t> &src2, const Tensor<bf16_t> &dst);

template
void hypot<fp16_t>(Scalar alpha, const Tensor<fp16_t> &src1, Scalar beta,
        const Tensor<fp16_t> &src2, const Tensor<fp16_t> &dst);

} // namespace nntile::tensor
