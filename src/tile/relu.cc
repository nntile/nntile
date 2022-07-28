/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/relu.cc
 * ReLU operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/tile/relu.hh"
#include <cmath>

namespace nntile
{

template<typename T>
void relu_kernel_cpu(Index nelems, T *data)
    noexcept
{
    constexpr T zero = 0;
    for(Index i = 0; i < nelems; ++i)
    {
        data[i] = std::max(zero, data[i]);
    }
}

template<typename T>
void relu_starpu_cpu(void *buffers[], void *cl_args)
    noexcept
{
    Index nelems;
    starpu_codelet_unpack_args(cl_args, &nelems);
    T *data = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
    relu_kernel_cpu<T>(nelems, data);
}

starpu_perfmodel relu_perfmodel_fp32 =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "nntile_relu_fp32",
};

starpu_perfmodel relu_perfmodel_fp64 =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "nntile_relu_fp64",
};

StarpuCodelet relu_codelet_fp32("nntile_relu_fp32",
        &relu_perfmodel_fp32,
        {relu_starpu_cpu<fp32_t>},
#       ifdef NNTILE_USE_CUDA
            {relu_starpu_cuda<fp32_t>}
#       else // NNTILE_USE_CUDA
            {}
#       endif // NNTILE_USE_CUDA
        );

StarpuCodelet relu_codelet_fp64("nntile_relu_fp64",
        &relu_perfmodel_fp64,
        {relu_starpu_cpu<fp64_t>},
#       ifdef NNTILE_USE_CUDA
            {relu_starpu_cuda<fp64_t>}
#       else // NNTILE_USE_CUDA
            {}
#       endif // NNTILE_USE_CUDA
        );

void relu_restrict_where(uint32_t where)
{
    relu_codelet_fp32.restrict_where(where);
    relu_codelet_fp64.restrict_where(where);
}

void relu_restore_where()
{
    relu_codelet_fp32.restore_where();
    relu_codelet_fp64.restore_where();
}

template<typename T>
void relu_async(const Tile<T> &A)
{
    starpu_task_insert(relu_codelet<T>(),
            STARPU_VALUE, &A.nelems, sizeof(A.nelems),
            STARPU_RW, static_cast<starpu_data_handle_t>(A),
            // std::erf is assumed as a single flop
            STARPU_FLOPS, static_cast<double>(A.nelems),
            0);
}

template
void relu_async(const Tile<fp32_t> &A);

template
void relu_async(const Tile<fp64_t> &A);

} // namespace nntile

