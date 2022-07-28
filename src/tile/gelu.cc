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
void gelu_kernel_cpu(Index nelems, T *data)
    noexcept
{
    constexpr T one = 1, pt5 = 0.5;
    const T sqrt2 = std::sqrt(T{2.0});
    for(Index i = 0; i < nelems; ++i)
    {
        T tmp = pt5*(std::erf(data[i]/sqrt2)) + pt5;
        data[i] *= tmp;
    }
}

template<typename T>
void gelu_starpu_cpu(void *buffers[], void *cl_args)
    noexcept
{
    Index nelems;
    starpu_codelet_unpack_args(cl_args, &nelems);
    T *data = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
    gelu_kernel_cpu<T>(nelems, data);
}

starpu_perfmodel gelu_perfmodel_fp32 =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "nntile_gelu_fp32",
};

starpu_perfmodel gelu_perfmodel_fp64 =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "nntile_gelu_fp64",
};

StarpuCodelet gelu_codelet_fp32("nntile_gelu_fp32",
        &gelu_perfmodel_fp32,
        {gelu_starpu_cpu<fp32_t>},
        {}
        );

StarpuCodelet gelu_codelet_fp64("nntile_gelu_fp64",
        &gelu_perfmodel_fp64,
        {gelu_starpu_cpu<fp64_t>},
        {}
        );

template<typename T>
void gelu_work(const Tile<T> &A)
{
    int ret = starpu_task_insert(gelu_codelet<T>(),
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

