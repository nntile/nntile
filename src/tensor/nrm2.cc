/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/nrm2.cc
 * Euclidean norm of Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/nrm2.hh"
#include "nntile/starpu/nrm2.hh"
#include "nntile/starpu/clear.hh"
#include "nntile/starpu/hypot.hh"

namespace nntile::tensor
{

//! Compute Euclidean norm
template<typename T>
void nrm2_async(Scalar alpha, const Tensor<T> &src, Scalar beta, const Tensor<T> &dst,
        const Tensor<T> &tmp)
{
    // Check dimensions
    if(dst.ndim != 0)
    {
        throw std::runtime_error("dst.ndim != 0");
    }
    if(src.ndim != tmp.ndim)
    {
        throw std::runtime_error("src.ndim != tmp.ndim");
    }
    // Check shapes of src and tmp
    if(tmp.shape != src.grid.shape)
    {
        throw std::runtime_error("tmp.shape != src.grid.shape");
    }
    for(Index i = 0; i < src.ndim; ++i)
    {
        if(tmp.basetile_shape[i] != 1)
        {
            throw std::runtime_error("tmp.basetile_shape[i] != 1");
        }
    }
    // Do actual calculations. At first calculate norms of tiles
    int mpi_rank = starpu_mpi_world_rank();
    int ret;
    Index ndim = src.ndim;
    auto dst_tile_handle = dst.get_tile_handle(0);
    auto dst_tile_rank = dst_tile_handle.mpi_get_rank();
    for(Index i = 0; i < src.grid.nelems; ++i)
    {
        auto src_tile_handle = src.get_tile_handle(i);
        auto src_tile_traits = src.get_tile_traits(i);
        auto tmp_tile_handle = tmp.get_tile_handle(i);
        int src_tile_rank = src_tile_handle.mpi_get_rank();
        int tmp_tile_rank = tmp_tile_handle.mpi_get_rank();
        auto tmp_tile_tag = tmp_tile_handle.mpi_get_tag();
        // Flush cache for the output tile on every node
        tmp_tile_handle.mpi_flush();
        // Execute on source tile
        if(mpi_rank == src_tile_rank)
        {
            starpu::nrm2::submit<T>(src_tile_traits.nelems, src_tile_handle,
                    tmp_tile_handle);
            // Transfer result if needed
            if(mpi_rank != tmp_tile_rank)
            {
                // No need to check for cached send, as output was just updated
                //ret = starpu_mpi_isend_detached(
                //        static_cast<starpu_data_handle_t>(tmp_tile_handle),
                //        tmp_tile_rank, tmp_tile_tag, MPI_COMM_WORLD, nullptr,
                //        nullptr);
                //if(ret != 0)
                //{
                //    throw std::runtime_error("Error in starpu_mpi_isend_"
                //            "detached");
                //}
            }
        }
        // Init receive of tmp tile
        else if(mpi_rank == tmp_tile_rank)
        {
            // No need to check for cached recv, as output was just updated
            //ret = starpu_mpi_irecv_detached(
            //        static_cast<starpu_data_handle_t>(tmp_tile_handle),
            //        src_tile_rank, tmp_tile_tag, MPI_COMM_WORLD, nullptr,
            //        nullptr);
            //if(ret != 0)
            //{
            //    throw std::runtime_error("Error in starpu_mpi_irecv_"
            //            "detached");
            //}
        }
        // Update total norm
        tmp_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
        if(mpi_rank == dst_tile_rank)
        {
            // Init dst_tile_handle
            if(i == 0)
            {
                starpu::hypot::submit<T>(1, alpha, tmp_tile_handle, beta,
                        dst_tile_handle);
            }
            // Accumulate dst_tile_handle
            else
            {
                starpu::hypot::submit<T>(1, alpha, tmp_tile_handle, 1.0,
                        dst_tile_handle);
            }
        }
    }
    //tmp.invalidate_submit();
    // Flush cache for the output tile on every node
    dst_tile_handle.mpi_flush();
}

template<typename T>
void nrm2(Scalar alpha, const Tensor<T> &src, Scalar beta, const Tensor<T> &dst,
        const Tensor<T> &tmp)
{
    nrm2_async<T>(alpha, src, beta, dst, tmp);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void nrm2_async<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src, Scalar beta,
        const Tensor<fp32_t> &dst, const Tensor<fp32_t> &tmp);

template
void nrm2_async<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src, Scalar beta,
        const Tensor<fp64_t> &dst, const Tensor<fp64_t> &tmp);

// Explicit instantiation
template
void nrm2<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src, Scalar beta,
        const Tensor<fp32_t> &dst, const Tensor<fp32_t> &tmp);

template
void nrm2<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src, Scalar beta,
        const Tensor<fp64_t> &dst, const Tensor<fp64_t> &tmp);

} // namespace nntile::tensor
