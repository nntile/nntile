/*! @randnright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/randn.hh
 * Randn operation on StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-31
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/starpu.hh>

namespace nntile
{
namespace starpu
{
namespace randn
{

// Randn operation on StarPU buffers
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept;

extern StarpuCodelet codelet_fp32, codelet_fp64;

template<typename T>
constexpr StarpuCodelet *codelet()
{
    throw std::runtime_error("Non-supported type");
    return nullptr;
}

template<>
constexpr StarpuCodelet *codelet<fp32_t>()
{
    return &codelet_fp32;
}

template<>
constexpr StarpuCodelet *codelet<fp64_t>()
{
    return &codelet_fp64;
}

void init();

void restrict_where(uint32_t where);

void restore_where();

template<typename T>
void submit(Index ndim, Index nelems, unsigned long long seed,
        T mean, T stddev, const std::vector<Index> &start,
        const std::vector<Index> &shape, const std::vector<Index> &stride,
        const std::vector<Index> &underlying_shape, starpu_data_handle_t data,
        starpu_data_handle_t tmp_index);

} // namespace randn
} // namespace starpu
} // namespace nntile

