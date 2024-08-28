/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/randn.cc
 * Randn operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/randn.hh"
#include "nntile/starpu/randn.hh"

namespace nntile::tile
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
        starpu::VariableHandle tmp_index(sizeof(int64_t)*2*ndim, STARPU_R);
        // Insert task
        starpu::randn::submit<T>(ndim, dst.nelems, seed, mean, stddev, start,
                dst.shape, dst.stride, underlying_shape, dst, tmp_index);
    }
    else
    {
        starpu::Handle null_handle;
        starpu::randn::submit<T>(0, 1, seed, mean, stddev, start,
                dst.shape, dst.stride, underlying_shape, dst, null_handle);
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
        Scalar mean, Scalar stddev)
{
    randn_async<T>(dst, start, underlying_shape, seed, mean, stddev);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void randn_async<fp32_t>(const Tile<fp32_t> &dst,
        const std::vector<Index> &start,
        const std::vector<Index> &underlying_shape, unsigned long long seed,
        Scalar mean, Scalar stddev);

template
void randn_async<fp32_fast_tf32_t>(const Tile<fp32_fast_tf32_t> &dst,
        const std::vector<Index> &start,
        const std::vector<Index> &underlying_shape, unsigned long long seed,
        Scalar mean, Scalar stddev);

template
void randn_async<fp64_t>(const Tile<fp64_t> &dst,
        const std::vector<Index> &start,
        const std::vector<Index> &underlying_shape, unsigned long long seed,
        Scalar mean, Scalar stddev);

template
void randn_async<bf16_t>(const Tile<bf16_t> &dst,
        const std::vector<Index> &start,
        const std::vector<Index> &underlying_shape, unsigned long long seed,
        Scalar mean, Scalar stddev);

// Explicit instantiation
template
void randn<fp32_t>(const Tile<fp32_t> &dst, const std::vector<Index> &start,
        const std::vector<Index> &underlying_shape, unsigned long long seed,
        Scalar mean, Scalar stddev);

template
void randn<fp32_fast_tf32_t>(const Tile<fp32_fast_tf32_t> &dst, const std::vector<Index> &start,
        const std::vector<Index> &underlying_shape, unsigned long long seed,
        Scalar mean, Scalar stddev);

template
void randn<fp64_t>(const Tile<fp64_t> &dst, const std::vector<Index> &start,
        const std::vector<Index> &underlying_shape, unsigned long long seed,
        Scalar mean, Scalar stddev);

template
void randn<bf16_t>(const Tile<bf16_t> &dst, const std::vector<Index> &start,
        const std::vector<Index> &underlying_shape, unsigned long long seed,
        Scalar mean, Scalar stddev);

} // namespace nntile::tile
