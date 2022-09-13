/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/randn.cc
 * Randn operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-12
 * */

#include "nntile/tensor/randn.hh"
#include "nntile/starpu/randn.hh"

namespace nntile
{
namespace tensor
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
        T mean, T stddev)
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
    // Temporary index
    StarpuVariableHandle tmp_index(sizeof(Index)*2*ndim, STARPU_SCRATCH);
    // Now do the job
    std::vector<Index> tile_start(start), tile_index(dst.ndim);
    for(Index i = 0; i < dst.grid.nelems; ++i)
    {
        // Get all the info about tile
        auto tile_handle = dst.get_tile_handle(i);
        int tile_rank = starpu_mpi_data_get_rank(tile_handle);
        // Insert task
        if(mpi_rank == tile_rank)
        {
            auto tile_traits = dst.get_tile_traits(i);
            starpu::randn::submit<T>(ndim, tile_traits.nelems, seed, mean,
                    stddev, tile_start, tile_traits.shape, tile_traits.stride,
                    underlying_shape, tile_handle, tmp_index);
        }
        // Generate index and starting point for the next tile
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
        T mean, T stddev)
{
    randn_async<T>(dst, start, underlying_shape, seed, mean, stddev);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void randn<fp32_t>(const Tensor<fp32_t> &dst, const std::vector<Index> &start,
        const std::vector<Index> &underlying_shape, unsigned long long seed,
        fp32_t mean, fp32_t stddev);

template
void randn<fp64_t>(const Tensor<fp64_t> &dst, const std::vector<Index> &start,
        const std::vector<Index> &underlying_shape, unsigned long long seed,
        fp64_t mean, fp64_t stddev);

} // namespace tensor
} // namespace nntile

