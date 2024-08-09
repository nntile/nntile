/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/subcopy.hh
 * Copy subarray based on contiguous indices
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/starpu/config.hh>

namespace nntile::starpu::subcopy
{

// Complex copying through StarPU buffers is available only on CPU
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept;

//extern Codelet codelet_fp16;
extern Codelet codelet_fp32, codelet_fp64, codelet_int64,
       codelet_bool, codelet_fp32_fast_tf32, codelet_bf16;

template<typename T>
constexpr Codelet *codelet()
{
    throw std::runtime_error("Non-supported type");
    return nullptr;
}

//template<>
//constexpr Codelet *codelet<fp16_t>()
//{
//    return &codelet_fp16;
//}

template<>
constexpr Codelet *codelet<fp32_t>()
{
    return &codelet_fp32;
}

template<>
constexpr Codelet *codelet<fp64_t>()
{
    return &codelet_fp64;
}

template<>
constexpr Codelet *codelet<int64_t>()
{
    return &codelet_int64;
}

template<>
constexpr Codelet *codelet<bool_t>()
{
    return &codelet_bool;
}

template<>
constexpr Codelet *codelet<bf16_t>()
{
    return &codelet_bf16;
}

template<>
constexpr Codelet *codelet<fp32_fast_tf32_t>()
{
    return &codelet_fp32_fast_tf32;
}

void init();

void restrict_where(uint32_t where);

void restore_where();

template<typename T>
void submit(Index ndim, const std::vector<Index> &src_start,
        const std::vector<Index> &src_stride,
        const std::vector<Index> &dst_start,
        const std::vector<Index> &dst_stride,
        const std::vector<Index> &copy_shape, Handle src, Handle dst,
        Handle tmp_index, starpu_data_access_mode mode);

} // namespace nntile::starpu::subcopy
