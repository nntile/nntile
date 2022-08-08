/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/bias.cc
 * Bias operation on a StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-08
 * */

#include "nntile/starpu/bias.hh"
#include "nntile/kernel/cpu/bias.hh"

namespace nntile
{
namespace starpu
{

//! Apply bias along middle axis of StarPU buffer
template<typename T>
void bias_cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<bias_args *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<StarpuVariableInterface **>(buffers);
    // Launch kernel
    const T *src = interfaces[0]->get_ptr<T>();
    T *dst = interfaces[1]->get_ptr<T>();
    nntile::kernel::cpu::bias<T>(args->m, args->n, args->k, src, dst);
}

starpu_perfmodel bias_perfmodel_fp32 =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "nntile_bias_fp32",
};

starpu_perfmodel bias_perfmodel_fp64 =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "nntile_bias_fp64",
};

StarpuCodelet bias_codelet_fp32("nntile_bias_fp32",
        &bias_perfmodel_fp32,
        {bias_cpu<fp32_t>},
        {}
        );

StarpuCodelet bias_codelet_fp64("nntile_bias_fp64",
        &bias_perfmodel_fp64,
        {bias_cpu<fp64_t>},
        {}
        );

void bias_restrict_where(uint32_t where)
{
    bias_codelet_fp32.restrict_where(where);
    bias_codelet_fp64.restrict_where(where);
}

void bias_restore_where()
{
    bias_codelet_fp32.restore_where();
    bias_codelet_fp64.restore_where();
}

template<typename T>
constexpr StarpuCodelet *bias_codelet()
{
    throw std::runtime_error("Non-supported type");
    return nullptr;
}

template<>
constexpr StarpuCodelet *bias_codelet<fp32_t>()
{
    return &bias_codelet_fp32;
}

template<>
constexpr StarpuCodelet *bias_codelet<fp64_t>()
{
    return &bias_codelet_fp64;
}

//! Insert task bias
template<typename T>
void bias(Index m, Index n, Index k, starpu_data_handle_t src,
        starpu_data_handle_t dst)
{
    // Codelet arguments
    auto args = new bias_args
    {
        .m = m,
        .n = n,
        .k = k
    };
    fp64_t nflops = m * n * k;
    // Submit task
    int ret = starpu_task_insert(bias_codelet<T>(),
            STARPU_R, src,
            STARPU_CL_ARGS, args, sizeof(*args),
            Starpu::STARPU_RW_COMMUTE, dst,
            STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in bias task submission");
    }
}

// Explicit instantiation
template
void bias<fp32_t>(Index m, Index n, Index k, starpu_data_handle_t src,
        starpu_data_handle_t dst);

template
void bias<fp64_t>(Index m, Index n, Index k, starpu_data_handle_t src,
        starpu_data_handle_t dst);

} // namespace starpu
} // namespace nntile

