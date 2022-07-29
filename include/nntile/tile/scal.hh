/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/scal.hh
 * Scaling operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#pragma once
#include <nntile/tile/tile.hh>

namespace nntile
{

extern starpu_perfmodel scal_perfmodel_fp32, scal_perfmodel_fp64;

extern StarpuCodelet scal_codelet_fp32, scal_codelet_fp64;

template<typename T>
constexpr StarpuCodelet *scal_codelet()
{
    throw std::runtime_error("Non-supported type");
    return nullptr;
}

template<>
constexpr StarpuCodelet *scal_codelet<fp32_t>()
{
    return &scal_codelet_fp32;
}

template<>
constexpr StarpuCodelet *scal_codelet<fp64_t>()
{
    return &scal_codelet_fp64;
}

template<typename T>
void scal_work(const Tile<T> &src, T alpha);

} // namespace nntile

