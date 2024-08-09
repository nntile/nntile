/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/add_fiber.cc
 * Tensor wrappers for addition of a tensor and a broadcasted fiber
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/add_fiber.hh"
#include "nntile/starpu/add_fiber.hh"

namespace nntile::tensor
{

template<typename T>
void add_fiber_async(Scalar alpha, const Tensor<T> &src, Scalar beta,
        const Tensor<T> &dst, Index axis, Index batch_ndim)
//! Tensor<T> addition of a tensor and a broadcasted fiber
/*! Reshapes input tensor and fiber into 3-dimensional and 1-dimensional arrays
 * and performs the following operations:
 *      dst[i,l,j,b] = beta*dst[i,l,j,b] + alpha*src[l,b]
 *
 * @param[in] alpha: Scalar factor for src
 * @param[in] src: Input fiber, that is reshaped into 1D array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Resulting tensor, that is reshaped into 3D array
 * */
{
    // Check dimensions
    if(src.ndim != batch_ndim+1)
    {
        throw std::runtime_error("src.ndim != batch_ndim+1");
    }
    // Check axis
    if(axis < 0)
    {
        throw std::runtime_error("axis < 0");
    }
    if(axis >= dst.ndim-batch_ndim)
    {
        throw std::runtime_error("axis >= dst.ndim-batch_ndim");
    }
    // Check shapes of tensors
    if(src.shape[0] != dst.shape[axis])
    {
        throw std::runtime_error("src.shape[0] != dst.shape[axis]");
    }
    if(src.basetile_shape[0] != dst.basetile_shape[axis])
    {
        throw std::runtime_error("src.basetile_shape[0] != "
                "dst.basetile_shape[axis]");
    }
    for(Index i = 0; i < batch_ndim; ++i)
    {
        if(src.shape[i+1] != dst.shape[dst.ndim-batch_ndim+i])
        {
            throw std::runtime_error("src.shape[i+1] != "
                    "dst.shape[dst.ndim-batch_ndim+i]");
        }
        if(src.basetile_shape[i+1] != dst.basetile_shape[dst.ndim-batch_ndim+i])
        {
            throw std::runtime_error("src.basetile_shape[i+1] != "
                    "dst.basetile_shape[dst.ndim-batch_ndim+i]");
        }
    }
    // Do nothing if alpha is zero
    if(alpha == 0.0)
    {
        return;
    }
    // Apply per-tile add_fiber asynchronously as needed
    int mpi_rank = starpu_mpi_world_rank();
    int ret;
    for(Index i = 0; i < dst.grid.nelems; ++i)
    {
        auto dst_tile_index = dst.grid.linear_to_index(i);
        auto dst_tile_traits = dst.get_tile_traits(i);
        auto dst_tile_handle = dst.get_tile_handle(i);
        int dst_tile_rank = dst_tile_handle.mpi_get_rank();
        // Get corresponding src tile
        std::vector<Index> src_tile_index(src.ndim);
        src_tile_index[0] = dst_tile_index[axis];
        for(Index j = 0; j < batch_ndim; ++j)
        {
            src_tile_index[j+1] = dst_tile_index[dst.ndim-batch_ndim+j];
        }
        auto src_tile_handle = src.get_tile_handle(src_tile_index);
        auto src_tile_traits = src.get_tile_traits(src_tile_index);
        int src_tile_rank = src_tile_handle.mpi_get_rank();
        // Transfer data
        src_tile_handle.mpi_transfer(dst_tile_rank, mpi_rank);
        // Execute on destination node
        if(mpi_rank == dst_tile_rank)
        {
            // Reshape inputs: src_tile -> (k,batch), dst_tile -> (m,k,n,batch)
            Index m, n, k, batch;
            batch = src_tile_traits.matrix_shape[1][1];
            m = dst_tile_traits.stride[axis];
            n = dst_tile_traits.matrix_shape[axis+1][1] / batch;
            k = dst_tile_traits.shape[axis];
            // Insert corresponding task
            starpu::add_fiber::submit<T>(m, n, k, batch, alpha,
                    src_tile_handle, beta, dst_tile_handle);
        }
        // Flush cache for the output tile on every node
        dst_tile_handle.mpi_flush();
    }
}

template<typename T>
void add_fiber(Scalar alpha, const Tensor<T> &src, Scalar beta, const Tensor<T> &dst,
        Index axis, Index batch_ndim)
//! Tensor<T> addition of a tensor and a broadcasted fiber
/*! Blocking version of add_fiber_async<T>.
 * Reshapes input tensor and fiber into 3-dimensional and 1-dimensional arrays
 * and performs the following operations:
 *      dst[i,l,j] = beta*dst[i,l,j] + alpha*src[l]
 *
 * @param[in] alpha: Scalar factor for src
 * @param[in] src: Input fiber, that is reshaped into 1D array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Resulting tensor, that is reshaped into 3D array
 * */
{
    add_fiber_async<T>(alpha, src, beta, dst, axis, batch_ndim);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation of template
template
void add_fiber_async<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src,
        Scalar beta, const Tensor<fp32_t> &dst, Index axis, Index batch_ndim);

template
void add_fiber_async<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src,
        Scalar beta, const Tensor<fp32_fast_tf32_t> &dst, Index axis, Index batch_ndim);

template
void add_fiber_async<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src,
        Scalar beta, const Tensor<fp64_t> &dst, Index axis, Index batch_ndim);

template
void add_fiber_async<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src, Scalar beta,
        const Tensor<bf16_t> &dst, Index axis, Index batch_ndim);

// Explicit instantiation of template
template
void add_fiber<fp32_t>(Scalar alpha, const Tensor<fp32_t> &src, Scalar beta,
        const Tensor<fp32_t> &dst, Index axis, Index batch_ndim);

template
void add_fiber<fp32_fast_tf32_t>(Scalar alpha, const Tensor<fp32_fast_tf32_t> &src,
        Scalar beta, const Tensor<fp32_fast_tf32_t> &dst, Index axis, Index batch_ndim);

template
void add_fiber<fp64_t>(Scalar alpha, const Tensor<fp64_t> &src, Scalar beta,
        const Tensor<fp64_t> &dst, Index axis, Index batch_ndim);

template
void add_fiber<bf16_t>(Scalar alpha, const Tensor<bf16_t> &src, Scalar beta,
        const Tensor<bf16_t> &dst, Index axis, Index batch_ndim);

} // namespace nntile::tensor
