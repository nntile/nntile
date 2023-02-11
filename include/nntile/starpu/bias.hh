/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/bias.hh
 * Bias operation on a StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-27
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/starpu/config.hh>
#include <nntile/defs.h>

namespace nntile
{
namespace starpu
{
namespace bias
{

struct argc_t {
    argc_t(Index nargc) : num_arguments(nargc) {}
    Index num_arguments;
};

//! Structure for arguments
struct args_t : argc_t
{
    args_t(Index nargc, Index m_, Index n_, Index k_) : argc_t(nargc), m(m_), n(n_), k(k_) {}
    Index m;
    Index n;
    Index k;
};

//! Structure for arguments
template<typename T>
struct val_size_t : argc_t
{
    val_size_t(Index nargc, T val_, Index nelems_) : argc_t(nargc), val(val_), nelems(nelems_) {}
    T val;
    Index nelems;
};

// Apply bias along middle axis of StarPU buffer on CPU
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept;

#ifdef NNTILE_USE_CUDA
// Apply bias along middle axis of StarPU buffer on CUDA
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept;
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
void submit(Index m, Index n, Index k, Handle src, Handle dst);

template<typename T>
void submit(T val, Index num_elements, Handle src);

} // namespace bias
} // namespace starpu
} // namespace nntile

