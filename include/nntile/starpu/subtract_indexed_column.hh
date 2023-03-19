/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/subtract_indexed_column.hh
 * Subtraction of a given value from indexed column for StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-03-18
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/starpu/config.hh>

namespace nntile
{
namespace starpu
{
namespace subtract_indexed_column
{
    
template<typename T>
struct args_t
{
    Index n_row;
    T value;
};

// Subtraction of a given value from the indexed column of StarPU buffer on CPU
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept;

#ifdef NNTILE_USE_CUDA
// Subtraction of a given value from the indexed column of StarPU buffer on CUDA
// template<typename T>
// void cuda(void *buffers[], void *cl_args)
//     noexcept;
#endif // NNTILE_USE_CUDA

extern Codelet codelet_fp32, codelet_fp64;

template<typename T>
constexpr Codelet *codelet()
{
    throw std::runtime_error("Non-supported type");
    return nullptr;
}

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

void init();

void restrict_where(uint32_t where);

void restore_where();

template<typename T>
void submit(Index n_row, T val, Handle class_labels, Handle dst);

} // namespace subtract_indexed_column
} // namespace starpu
} // namespace nntile