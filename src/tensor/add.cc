/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/add.cc
 * Add operation for Tensor<T>'s
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/add.hh"
#include "nntile/starpu/add.hh"

namespace nntile::tensor
{

//! Tensor-wise add operation
template<typename T>
void add_async(Scalar alpha, const Tensor<T> &src1, Scalar beta,
        const Tensor<T> &src2, const Tensor<T> &dst)
{
    // Check dimensions
    if(dst.ndim != src1.ndim)
    {
        throw std::runtime_error("dst.ndim != src1.ndim");
    }
    if(dst.ndim != src2.ndim)
    {
        throw std::runtime_error("dst.ndim != src2.ndim");
    }
    // Check shapes of tensors
    for(Index i = 0; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != src1.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != src1.shape[i]");
        }
        if(dst.shape[i] != src2.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != src2.shape[i]");
        }
        if(dst.basetile_shape[i] != src1.basetile_shape[i])
        {
            throw std::runtime_error("dst.basetile_shape[i] != "
                    "src1.basetile_shape[i]");
        }
        if(dst.basetile_shape[i] != src2.basetile_shape[i])
        {
            throw std::runtime_error("dst.basetile_shape[i] != "
                    "src2.basetile_shape[i]");
        }
    }
    // Do nothing if alpha is zero
    if(alpha == 0.0 && beta == 1.)
    {
        return;
    }
    // Apply per-tile add asynchronously as needed
    int mpi_rank = starpu_mpi_world_rank();

    for(Index i = 0; i < src1.grid.nelems; ++i)
    {
        // Get handle for corresponding tiles of src and dst
        auto src1_tile_handle = src1.get_tile_handle(i);
        auto src2_tile_handle = src2.get_tile_handle(i);
        auto dst_tile_handle = dst.get_tile_handle(i);
        // MPI rank of the destination tile
        int dst_tile_rank = dst_tile_handle.mpi_get_rank();
        // Transfer data
        src1_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
        src2_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
        // Execute only on destination node
        if(mpi_rank == dst_tile_rank)
        {
            auto traits = src1.get_tile_traits(i);
            starpu::add::submit<T>(traits.nelems, alpha, src1_tile_handle,
                    beta, src2_tile_handle, dst_tile_handle);
        }
        // Flush cache for the output tile on every node
        dst_tile_handle.mpi_flush();
    }

}

//! Tensor-wise add operation
template<typename T>
void add(Scalar alpha, const Tensor<T> &src1, Scalar beta,
        const Tensor<T> &src2, const Tensor<T> &dst)
{
    add_async<T>(alpha, src1, beta, src2, dst);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation of template
template
void add_async<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src1, Scalar beta,
        const Tensor<fp32_t> &src2, const Tensor<fp32_t> &dst);

template
void add_async<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src1, Scalar beta,
        const Tensor<bf16_t> &src2, const Tensor<bf16_t> &dst);

template
void add_async<fp32_fast_tf32_t>(Scalar alpha,
        const Tensor<fp32_fast_tf32_t> &src1, Scalar beta,
        const Tensor<fp32_fast_tf32_t> &src2,
        const Tensor<fp32_fast_tf32_t> &dst);

template
void add_async<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src1, Scalar beta,
        const Tensor<fp64_t> &src2, const Tensor<fp64_t> &dst);

// Explicit instantiation of template

template
void add<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src1, Scalar beta,
        const Tensor<fp32_t> &src2, const Tensor<fp32_t> &dst);

template
void add<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src1, Scalar beta,
        const Tensor<bf16_t> &src2, const Tensor<bf16_t> &dst);

template
void add<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src1,
        Scalar beta, const Tensor<fp32_fast_tf32_t> &src2,
        const Tensor<fp32_fast_tf32_t> &dst);

template
void add<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src1, Scalar beta,
        const Tensor<fp64_t> &src2, const Tensor<fp64_t> &dst);

} // namespace nntile::tensor
