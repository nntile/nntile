/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/randn.hh
 * Randn operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{

//! Asynchronous tile-wise random generation operation
//
// @param[out] dst: Destination tile
// @param[in] offset: Offset of the destination tile in the underlying tile
// @param[in] shape: Shape of the underlying tile
// @param[in] stride: Stride of the underlying tile
// @param[in] seed: Seed for the normal random distribution
// @param[in] mean: Average of the normal random distribution
// @param[in] stddev: Deviation of the normal random distribution
//
// Randomly fill the output tile as if it is a part of the provided
// underlying tile. The destination tile shall be fully inside the
// underlying tile.
template<typename T>
void randn_async(const Tile<T> &dst, const std::vector<Index> &offset,
        const std::vector<Index> &shape, const std::vector<Index> &stride,
        unsigned long long seed, T mean=0, T stddev=1);

extern template
void randn_async(const Tile<fp32_t> &dst, const std::vector<Index> &offset,
        const std::vector<Index> &shape, const std::vector<Index> &stride,
        unsigned long long seed, fp32_t mean=0, fp32_t stddev=1);

extern template
void randn_async(const Tile<fp64_t> &dst, const std::vector<Index> &offset,
        const std::vector<Index> &shape, const std::vector<Index> &stride,
        unsigned long long seed, fp64_t mean=0, fp64_t stddev=1);

//! Asynchronous tile-wise random generation operation
//
// @param[out] dst: Destination tile
// @param[in] seed: Seed for the normal random distribution
// @param[in] mean: Average of the normal random distribution
// @param[in] stddev: Deviation of the normal random distribution
template<typename T>
void randn_async(const Tile<T> &dst, unsigned long long seed, T mean=0,
        T stddev=1)
{
    randn_async<T>(dst, std::vector<Index>(dst.ndim), dst.shape, dst.stride,
            seed, mean, stddev);
}

//! Blocking version of tile-wise random generation operation
//
// @param[out] dst: Destination tile
// @param[in] offset: Offset of the destination tile in the underlying tile
// @param[in] shape: Shape of the underlying tile
// @param[in] stride: Stride of the underlying tile
// @param[in] seed: Seed for the normal random distribution
// @param[in] mean: Average of the normal random distribution
// @param[in] stddev: Deviation of the normal random distribution
//
// Randomly fill the output tile as if it is a part of the provided
// underlying tile. The destination tile shall be fully inside the
// underlying tile.
template<typename T>
void randn(const Tile<T> &dst, const std::vector<Index> &offset,
        const std::vector<Index> &shape, const std::vector<Index> &stride,
        unsigned long long seed, T mean=0, T stddev=1)
{
    randn_async<T>(dst, offset, shape, stride, seed, mean, stddev);
    starpu_task_wait_for_all();
}

//! Blocking version of tile-wise random generation operation
//
// @param[out] dst: Destination tile
// @param[in] seed: Seed for the normal random distribution
// @param[in] mean: Average of the normal random distribution
// @param[in] stddev: Deviation of the normal random distribution
template<typename T>
void randn(const Tile<T> &dst, unsigned long long seed, T mean=0, T stddev=1)
{
    randn_async<T>(dst, std::vector<Index>(dst.ndim), dst.shape, dst.stride,
            seed, mean, stddev);
    starpu_task_wait_for_all();
}

} // namespace nntile

