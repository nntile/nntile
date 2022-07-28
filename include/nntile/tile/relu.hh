/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/relu.hh
 * ReLU operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{

template<typename T>
void relu_kernel_cpu(Index nelems, T *data)
    noexcept;

template<typename T>
void relu_starpu_cpu(void *buffers[], void *cl_args)
    noexcept;

#ifdef NNTILE_USE_CUDA
template<typename T>
void relu_kernel_cuda(Index nelems, T *data, const dim3 &grid,
        const dim3 &block, const cudaStream_t stream)
    noexcept;

template<typename T>
void relu_starpu_cuda(void *buffers[], void *cl_args)
    noexcept;
#endif // NNTILE_USE_CUDA

extern starpu_perfmodel relu_perfmodel_fp32, relu_perfmodel_fp64;

extern StarpuCodelet relu_codelet_fp32, relu_codelet_fp64;

void relu_restrict_where(uint32_t where);

void relu_restore_where();

template<typename T>
constexpr StarpuCodelet *relu_codelet()
{
    throw std::runtime_error("Non-supported type");
    return nullptr;
}

template<>
constexpr StarpuCodelet *relu_codelet<fp32_t>()
{
    return &relu_codelet_fp32;
}

template<>
constexpr StarpuCodelet *relu_codelet<fp64_t>()
{
    return &relu_codelet_fp64;
}

//! Asynchronous tile-wise ReLU operation
//
// @param[inout] A: Tile for the element-wise ReLU operation
template<typename T>
void relu_async(const Tile<T> &A);

extern template
void relu_async(const Tile<fp32_t> &A);

extern template
void relu_async(const Tile<fp64_t> &A);

//! Blocking version of tile-wise ReLU operation
//
// @param[inout] A: Tile for the element-wise ReLU operation
template<typename T>
void relu(const Tile<T> &A)
{
    relu_async<T>(A);
    starpu_task_wait_for_all();
}

} // namespace nntile

