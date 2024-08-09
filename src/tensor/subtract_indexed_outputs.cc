/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/subtract_indexed_outputs.cc
 * Subtraction of value from certain elements in Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/subtract_indexed_outputs.hh"
#include "nntile/starpu/subtract_indexed_outputs.hh"

namespace nntile::tensor
{

template<typename T>
void subtract_indexed_outputs_async(Scalar val, const Tensor<int64_t> &labels,
        const Tensor<T> &dst)
{
    if(labels.ndim != dst.ndim-1)
    {
        throw std::runtime_error("labels.ndim != dst.ndim-1");
    }
    for(Index i = 0; i < labels.ndim; ++i)
    {
        if(labels.shape[i] != dst.shape[i+1])
        {
            throw std::runtime_error("labels.shape[i] != dst.shape[i+1]");
        }
        if(labels.basetile_shape[i] != dst.basetile_shape[i+1])
        {
            throw std::runtime_error("labels.basetile_shape[i] != "
                    "dst.basetile_shape[i+1]");
        }
    }
    if(dst.shape[0] != dst.basetile_shape[0])
    {
        throw std::runtime_error("dst.shape[0] != dst.basetile_shape[0]");
    }
    // Do actual calculations
    int mpi_rank = starpu_mpi_world_rank();
    for(Index i = 0; i < dst.grid.nelems; ++i)
    {
        auto labels_tile_handle = labels.get_tile_handle(i);
        auto labels_traits = labels.get_tile_traits(i);
        auto dst_tile_handle = dst.get_tile_handle(i);
        int dst_tile_rank = dst_tile_handle.mpi_get_rank();
        // Transfer data
        labels_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
        // Execute on destination node
        if(mpi_rank == dst_tile_rank)
        {
            // Insert task
            starpu::subtract_indexed_outputs::submit<T>(dst.shape[0],
                    labels_traits.nelems, val, labels_tile_handle,
                    dst_tile_handle);
        }
        dst_tile_handle.mpi_flush();
    }
}

template<typename T>
void subtract_indexed_outputs(Scalar val, const Tensor<int64_t> &labels,
        const Tensor<T> &dst)
{
    subtract_indexed_outputs_async<T>(val, labels, dst);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void subtract_indexed_outputs_async<fp32_t>(Scalar val,
        const Tensor<int64_t> &labels, const Tensor<fp32_t> &dst);

template
void subtract_indexed_outputs_async<fp32_fast_tf32_t>(Scalar val,
        const Tensor<int64_t> &labels, const Tensor<fp32_fast_tf32_t> &dst);

template
void subtract_indexed_outputs_async<fp64_t>(Scalar val,
        const Tensor<int64_t> &labels, const Tensor<fp64_t> &dst);

template
void subtract_indexed_outputs_async<bf16_t>(Scalar val, const Tensor<int64_t> &labels,
        const Tensor<bf16_t> &dst);

// Explicit instantiation
template
void subtract_indexed_outputs<fp32_t>(Scalar val, const Tensor<int64_t> &labels,
        const Tensor<fp32_t> &dst);

template
void subtract_indexed_outputs<fp32_fast_tf32_t>(Scalar val, const Tensor<int64_t> &labels,
        const Tensor<fp32_fast_tf32_t> &dst);

template
void subtract_indexed_outputs<fp64_t>(Scalar val, const Tensor<int64_t> &labels,
        const Tensor<fp64_t> &dst);

template
void subtract_indexed_outputs<bf16_t>(Scalar val, const Tensor<int64_t> &labels,
        const Tensor<bf16_t> &dst);

} // namespace nntile::tensor
