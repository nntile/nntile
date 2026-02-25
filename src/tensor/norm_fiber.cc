/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/norm_fiber.cc
 * Euclidean norms over slices into a fiber of a product of a Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/norm_fiber.hh"
#include "nntile/tile/norm_fiber.hh"
#include "nntile/tile/norm_fiber_inplace.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tensor
{

//! Tensor-wise norm_fiber
template<typename T>
void norm_fiber_async(Scalar alpha, const Tensor<T> &src1, Scalar beta,
        const Tensor<T> &src2,
        const Tensor<T> &dst, Index axis, Index batch_ndim, int redux)
{
    // Check dimensions
    if(dst.ndim != batch_ndim+1)
    {
        throw std::runtime_error("dst.ndim != batch_ndim+1");
    }
    if(src2.ndim != dst.ndim)
    {
        throw std::runtime_error("src2.ndim != dst.ndim");
    }
    // Treat special case of src1.ndim=0
    if(src1.ndim == 0)
    {
        throw std::runtime_error("Scalar input makes no sense");
    }
    // Check axis
    if(axis < 0)
    {
        throw std::runtime_error("axis < 0");
    }
    if(axis >= src1.ndim-batch_ndim)
    {
        throw std::runtime_error("axis >= src1.ndim-batch_ndim");
    }
    // Check shapes
    if(dst.shape != src2.shape)
    {
        throw std::runtime_error("dst.shape != src2.shape");
    }
    if(dst.basetile_shape != src2.basetile_shape)
    {
        throw std::runtime_error("dst.basetile_shape != src2.basetile_shape");
    }
    if(dst.shape[0] != src1.shape[axis])
    {
        throw std::runtime_error("dst.shape[0] != src1.shape[axis]");
    }
    if(dst.basetile_shape[0] != src1.basetile_shape[axis])
    {
        throw std::runtime_error("dst.basetile_shape[0] != "
                "src1.basetile_shape[axis]");
    }
    for(Index i = 0; i < batch_ndim; ++i)
    {
        if(dst.shape[i+1] != src1.shape[src1.ndim-batch_ndim+i])
        {
            throw std::runtime_error("dst.shape[i+1] != "
                    "src1.shape[src1.ndim-batch_ndim+i]");
        }
        if(dst.basetile_shape[i+1] != src1.basetile_shape[src1.ndim-batch_ndim+i])
        {
            throw std::runtime_error("dst.basetile_shape[i+1] != "
                    "src1.basetile_shape[src1.ndim-batch_ndim+i]");
        }
    }

    // Do actual calculations
    constexpr Scalar one = 1.0;
    // go over bigger tensor
    for(Index i = 0; i < src1.grid.nelems; ++i)
    {
        auto src1_tile = src1.get_tile(i);
        auto src1_tile_index = src1.grid.linear_to_index(i);
        // Get corresponding dst tile
        std::vector<Index> dst_tile_index(dst.ndim);
        dst_tile_index[0] = src1_tile_index[axis];

        for(Index j = 0; j < batch_ndim; ++j)
        {
            dst_tile_index[j+1] = src1_tile_index[src1.ndim-batch_ndim+j];
        }

        auto dst_tile_handle = dst.get_tile_handle(dst_tile_index);
        auto dst_tile = dst.get_tile(dst_tile_index);
        auto src2_tile = src2.get_tile(dst_tile_index);
        bool init_first = true;
        for(Index j = 0; j < src1.ndim-batch_ndim; ++j)
        {
            if(j != axis and src1_tile_index[j] != 0)
            {
                init_first = false;
                break;
            }
        }
        if(init_first)
        {
            // The first time we need to take into account src2
            tile::norm_fiber_async<T>(alpha, src1_tile, beta, src2_tile,
                    dst_tile, axis, batch_ndim, redux);
        }
        else
        {
            // The rest of the times we shall rely on an inplace update
            tile::norm_fiber_inplace_async<T>(alpha, src1_tile, one, dst_tile,
                    axis, batch_ndim, redux);
        }
        // Flush cache for the output tile on every node
        dst_tile_handle.mpi_flush();
    }
}

//! Tensor-wise norm_fiber
template<typename T>
void norm_fiber(Scalar alpha, const Tensor<T> &src1, Scalar beta, const Tensor<T> &src2, const Tensor<T> &dst,
        Index axis, Index batch_ndim, int redux)
{
    norm_fiber_async<T>(alpha, src1, beta, src2, dst, axis, batch_ndim, redux);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void norm_fiber_async<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src1,
        Scalar beta, const Tensor<fp32_t> &src2,
        const Tensor<fp32_t> &dst, Index axis, Index batch_ndim,
        int redux);

template
void norm_fiber_async<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src1,
        Scalar beta, const Tensor<fp32_fast_tf32_t> &src2,
        const Tensor<fp32_fast_tf32_t> &dst, Index axis, Index batch_ndim,
        int redux);

template
void norm_fiber_async<fp32_fast_fp16_t>(Scalar alpha, const Tensor<fp32_fast_fp16_t> &src1,
        Scalar beta, const Tensor<fp32_fast_fp16_t> &src2,
        const Tensor<fp32_fast_fp16_t> &dst, Index axis, Index batch_ndim,
        int redux);

template
void norm_fiber_async<fp32_fast_bf16_t>(Scalar alpha, const Tensor<fp32_fast_bf16_t> &src1,
        Scalar beta, const Tensor<fp32_fast_bf16_t> &src2,
        const Tensor<fp32_fast_bf16_t> &dst, Index axis, Index batch_ndim,
        int redux);

template
void norm_fiber_async<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src1,
        Scalar beta, const Tensor<fp64_t> &src2,
        const Tensor<fp64_t> &dst, Index axis, Index batch_ndim,
        int redux);

template
void norm_fiber_async<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src1, Scalar beta,
        const Tensor<bf16_t> &src2,
        const Tensor<bf16_t> &dst, Index axis, Index batch_ndim,
        int redux);

// Explicit instantiation
template
void norm_fiber<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src1, Scalar beta,
        const Tensor<fp32_t> &src2,
        const Tensor<fp32_t> &dst, Index axis, Index batch_ndim,
        int redux);

template
void norm_fiber<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src1, Scalar beta,
        const Tensor<fp32_fast_tf32_t> &src2,
        const Tensor<fp32_fast_tf32_t> &dst, Index axis, Index batch_ndim,
        int redux);

template
void norm_fiber<fp32_fast_fp16_t>(Scalar alpha, const Tensor<fp32_fast_fp16_t> &src1, Scalar beta,
        const Tensor<fp32_fast_fp16_t> &src2,
        const Tensor<fp32_fast_fp16_t> &dst, Index axis, Index batch_ndim,
        int redux);

template
void norm_fiber<fp32_fast_bf16_t>(Scalar alpha, const Tensor<fp32_fast_bf16_t> &src1, Scalar beta,
        const Tensor<fp32_fast_bf16_t> &src2,
        const Tensor<fp32_fast_bf16_t> &dst, Index axis, Index batch_ndim,
        int redux);

template
void norm_fiber<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src1, Scalar beta,
        const Tensor<fp64_t> &src2,
        const Tensor<fp64_t> &dst, Index axis, Index batch_ndim,
        int redux);

template
void norm_fiber<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src1, Scalar beta,
        const Tensor<bf16_t> &src2,
        const Tensor<bf16_t> &dst, Index axis, Index batch_ndim,
        int redux);

} // namespace nntile::tensor
