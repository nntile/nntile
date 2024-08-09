/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/randn.cc
 * Randn operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/randn.hh"
#include "nntile/starpu/randn.hh"

namespace nntile::tensor
{

//! Asynchronous tensor-wise random generation operation
/*! Randomly fill the output tensor as if it is a part of the provided
 * underlying tensor. The destination tensor shall be fully inside the
 * underlying tensor.
 *
 * @param[out] dst: Destination tensor
 * @param[in] start: Starting index of a subarray to generate. Contains ndim
 *      values.
 * @param[in] underlying_shape: Shape of the underlying array. Contains ndim
 *      values.
 * @param[in] seed: Random seed for the entire underlying array
 * @param[in] mean: Average value of the normal distribution
 * @param[in] stddev: Standard deviation of the normal distribution
 * */
template<typename T>
void randn_async(const Tensor<T> &dst, const std::vector<Index> &start,
        const std::vector<Index> &underlying_shape, unsigned long long seed,
        Scalar mean, Scalar stddev)
{
    // Check dimensions
    if(dst.ndim != start.size())
    {
        throw std::runtime_error("dst.ndim != start.size()");
    }
    if(dst.ndim != underlying_shape.size())
    {
        throw std::runtime_error("dst.ndim != underlying_shape.size()");
    }
    Index ndim = dst.ndim;
    int mpi_rank = starpu_mpi_world_rank();
    // Check start and underlying_shape
    for(Index i = 0; i < ndim; ++i)
    {
        if(start[i] < 0)
        {
            throw std::runtime_error("start[i] < 0");
        }
        if(start[i]+dst.shape[i] > underlying_shape[i])
        {
            throw std::runtime_error("start[i]+dst.shape[i] > "
                    "underlying_shape[i]");
        }
    }
    // Tackle ndim=0 case
    if(ndim == 0)
    {
        auto tile_handle = dst.get_tile_handle(0);
        int tile_rank = tile_handle.mpi_get_rank();
        if(mpi_rank == tile_rank)
        {
            starpu::Handle null_handle;
            starpu::randn::submit<T>(0, 1, seed, mean, stddev, start,
                    dst.shape, dst.stride, underlying_shape, tile_handle,
                    null_handle);
        }
        // Flush cache for the output tile on every node
        tile_handle.mpi_flush();
        return;
    }
    // Temporary index
    starpu::VariableHandle tmp_index(sizeof(int64_t)*2*ndim, STARPU_SCRATCH);
    // Now do the job
    std::vector<Index> tile_start(start), tile_index(dst.ndim);
    for(Index i = 0; i < dst.grid.nelems; ++i)
    {
        // Get all the info about tile
        auto tile_handle = dst.get_tile_handle(i);
        int tile_rank = tile_handle.mpi_get_rank();
        // Insert task
        if(mpi_rank == tile_rank)
        {
            auto tile_traits = dst.get_tile_traits(i);
            starpu::randn::submit<T>(ndim, tile_traits.nelems, seed, mean,
                    stddev, tile_start, tile_traits.shape, tile_traits.stride,
                    underlying_shape, tile_handle, tmp_index);
        }
        // Flush cache for the output tile on every node
        tile_handle.mpi_flush();
        // Generate index and starting point for the next tile
        if(i == dst.grid.nelems-1)
        {
            break;
        }
        ++tile_index[0];
        tile_start[0] += dst.basetile_shape[0];
        Index j = 0;
        while(tile_index[j] == dst.grid.shape[j])
        {
            tile_index[j] = 0;
            tile_start[j] = start[j];
            ++j;
            ++tile_index[j];
            tile_start[j] += dst.basetile_shape[j];
        }
    }
}

//! Blocking version of tensor-wise random generation operation
/*! Randomly fill the output tensor as if it is a part of the provided
 * underlying tensor. The destination tensor shall be fully inside the
 * underlying tensor.
 *
 * @param[out] dst: Destination tensor
 * @param[in] start: Starting index of a subarray to generate. Contains ndim
 *      values.
 * @param[in] underlying_shape: Shape of the underlying array. Contains ndim
 *      values.
 * @param[in] seed: Random seed for the entire underlying array
 * @param[in] mean: Average value of the normal distribution
 * @param[in] stddev: Standard deviation of the normal distribution
 * */
template<typename T>
void randn(const Tensor<T> &dst, const std::vector<Index> &start,
        const std::vector<Index> &underlying_shape, unsigned long long seed,
        Scalar mean, Scalar stddev)
{
    randn_async<T>(dst, start, underlying_shape, seed, mean, stddev);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void randn_async<fp32_t>(const Tensor<fp32_t> &dst,
        const std::vector<Index> &start,
        const std::vector<Index> &underlying_shape, unsigned long long seed,
        Scalar mean, Scalar stddev);

template
void randn_async<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &dst,
        const std::vector<Index> &start,
        const std::vector<Index> &underlying_shape, unsigned long long seed,
        Scalar mean, Scalar stddev);

template
void randn_async<fp64_t>(const Tensor<fp64_t> &dst,
        const std::vector<Index> &start,
        const std::vector<Index> &underlying_shape, unsigned long long seed,
        Scalar mean, Scalar stddev);

template
void randn_async<bf16_t>(const Tensor<bf16_t> &dst,
        const std::vector<Index> &start,
        const std::vector<Index> &underlying_shape, unsigned long long seed,
        Scalar mean, Scalar stddev);

// Explicit instantiation
template
void randn<fp32_t>(const Tensor<fp32_t> &dst, const std::vector<Index> &start,
        const std::vector<Index> &underlying_shape, unsigned long long seed,
        Scalar mean, Scalar stddev);

template
void randn<fp32_fast_tf32_t>(const Tensor<fp32_fast_tf32_t> &dst, const std::vector<Index> &start,
        const std::vector<Index> &underlying_shape, unsigned long long seed,
        Scalar mean, Scalar stddev);

template
void randn<fp64_t>(const Tensor<fp64_t> &dst, const std::vector<Index> &start,
        const std::vector<Index> &underlying_shape, unsigned long long seed,
        Scalar mean, Scalar stddev);

template
void randn<bf16_t>(const Tensor<bf16_t> &dst, const std::vector<Index> &start,
        const std::vector<Index> &underlying_shape, unsigned long long seed,
        Scalar mean, Scalar stddev);

} // namespace nntile::tensor
