/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/randn.cc
 * Randn operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-31
 * */

#include "nntile/tile/randn.hh"
#include "nntile/starpu/randn.hh"

namespace nntile
{
namespace tile
{

//! Asynchronous tile-wise random generation operation
/*! Randomly fill the output tile as if it is a part of the provided
 * underlying tile. The destination tile shall be fully inside the
 * underlying tile.
 *
 * @param[out] dst: Destination tile
 * @param[in] start: Starting index of a subarray to generate. Contains ndim
 *      values.
 * @param[in] underlying_shape: Shape of the underlying array. Contains ndim
 *      values.
 * @param[in] seed: Random seed for the entire underlying array
 * @param[in] mean: Average value of the normal distribution
 * @param[in] stddev: Standard deviation of the normal distribution
 * */
template<typename T>
void randn_async(const Tile<T> &dst, const std::vector<Index> &start,
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
    if(ndim != 0)
    {
        // Temporary index
        StarpuVariableHandle tmp_index(sizeof(Index)*2*ndim, STARPU_R);
        // Insert task
        starpu::randn::submit<T>(ndim, dst.nelems, seed, mean, stddev, start,
                dst.shape, dst.stride, underlying_shape, dst, tmp_index);
    }
    else
    {
        starpu::randn::submit<T>(ndim, dst.nelems, seed, mean, stddev, start,
                dst.shape, dst.stride, underlying_shape, dst, nullptr);
    }
}

//! Blocking version of tile-wise random generation operation
/*! Randomly fill the output tile as if it is a part of the provided
 * underlying tile. The destination tile shall be fully inside the
 * underlying tile.
 *
 * @param[out] dst: Destination tile
 * @param[in] start: Starting index of a subarray to generate. Contains ndim
 *      values.
 * @param[in] underlying_shape: Shape of the underlying array. Contains ndim
 *      values.
 * @param[in] seed: Random seed for the entire underlying array
 * @param[in] mean: Average value of the normal distribution
 * @param[in] stddev: Standard deviation of the normal distribution
 * */
template<typename T>
void randn(const Tile<T> &dst, const std::vector<Index> &start,
        const std::vector<Index> &underlying_shape, unsigned long long seed,
        T mean, T stddev)
{
    randn_async<T>(dst, start, underlying_shape, seed, mean, stddev);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void randn<fp32_t>(const Tile<fp32_t> &dst, const std::vector<Index> &start,
        const std::vector<Index> &underlying_shape, unsigned long long seed,
        fp32_t mean, fp32_t stddev);

template
void randn<fp64_t>(const Tile<fp64_t> &dst, const std::vector<Index> &start,
        const std::vector<Index> &underlying_shape, unsigned long long seed,
        fp64_t mean, fp64_t stddev);

} // namespace tile
} // namespace nntile

