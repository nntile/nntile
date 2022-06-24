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
static void cpu_relu(void *buffers[], void *cl_args)
{
    Index nelems;
    starpu_codelet_unpack_args(cl_args, &nelems);
    T *data = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
    for(Index i = 0; i < nelems; ++i)
    {
        data[i] = std::max(T{0}, data[i]);
    }
}

template<typename T>
void relu_async(const Tile<T> &A)
{
    static struct starpu_codelet codelet_relu =
    {
//#       ifdef NNTILE_USE_CUDA
//        .cuda_funcs = {relu_codelet_gpu<T>},
//        .cuda_flags = {STARPU_CUDA_ASYNC},
//#       else
        .cpu_funcs = {cpu_relu<T>},
//#       endif
        .nbuffers = 1,
        .modes = {STARPU_RW}
    };
    starpu_task_insert(&codelet_relu,
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

