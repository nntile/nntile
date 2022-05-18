/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/gelu.cc
 * GeLU operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/tile/gelu.hh"
#include <cmath>

namespace nntile
{

template<typename T>
static
void cpu_gelu(void *buffers[], void *cl_args)
    noexcept
{
    Index nelems;
    starpu_codelet_unpack_args(cl_args, &nelems);
    T *data = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
    constexpr T one = 1, pt5 = 0.5, sqrt2 = std::sqrt(T{2.0});
    for(Index i = 0; i < nelems; ++i)
    {
        T tmp = pt5*(std::erf(data[i]/sqrt2)) + pt5;
        data[i] *= tmp;
    }
}

template<typename T>
void gelu_work(const Tile<T> &A)
{
    static struct starpu_codelet codelet_gelu =
    {
        .cpu_funcs = {cpu_gelu<T>},
        .nbuffers = 1,
        .modes = {STARPU_RW}
    };
    int ret = starpu_task_insert(&codelet_gelu,
            STARPU_VALUE, &A.nelems, sizeof(A.nelems),
            STARPU_RW, static_cast<starpu_data_handle_t>(A),
            // std::erf is assumed as a single flop
            STARPU_FLOPS, static_cast<double>(5*A.nelems),
            0);
    if(ret != 0)
    {
        throw std::runtime_error("ret != 0");
    }
}

template
void gelu_work(const Tile<fp32_t> &A);

template
void gelu_work(const Tile<fp64_t> &A);

} // namespace nntile

