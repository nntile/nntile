/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/sumnorm.cc
 * Sum and Euclidian norm for StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-11
 * */

#include "nntile/starpu/sumnorm.hh"
#include "nntile/kernel/cpu/sumnorm.hh"

namespace nntile
{
namespace starpu
{

//! Sum and Euclidian norm along middle axis of StarPU buffer
template<typename T>
void sumnorm_cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<sumnorm_args *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<StarpuVariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    T *dst = interfaces[1]->get_ptr<T>();
    // Launch kernel
    kernel::cpu::sumnorm<T>(args->m, args->n, args->k, src, dst);
}

//! Footprint for sumnorm tasks that depends only on m, n and k
static
uint32_t sumnorm_footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<sumnorm_args *>(task->cl_arg);
    // Apply hash over parameters m, n and k. This way if we swap values of m,
    // n and k, then the total size of buffers will remain the same, but the
    // footprint will be different
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->m, sizeof(args->m), hash);
    hash = starpu_hash_crc32c_be_n(&args->n, sizeof(args->n), hash);
    hash = starpu_hash_crc32c_be_n(&args->k, sizeof(args->k), hash);
    return hash;
}

StarpuCodelet sumnorm_codelet_fp32("nntile_sumnorm_fp32",
        sumnorm_footprint,
        {sumnorm_cpu<fp32_t>},
        {}
        );

StarpuCodelet sumnorm_codelet_fp64("nntile_sumnorm_fp64",
        sumnorm_footprint,
        {sumnorm_cpu<fp64_t>},
        {}
        );

void sumnorm_restrict_where(uint32_t where)
{
    sumnorm_codelet_fp32.restrict_where(where);
    sumnorm_codelet_fp64.restrict_where(where);
}

void sumnorm_restore_where()
{
    sumnorm_codelet_fp32.restore_where();
    sumnorm_codelet_fp64.restore_where();
}

template<typename T>
constexpr StarpuCodelet *sumnorm_codelet()
{
    throw std::runtime_error("Non-supported type");
    return nullptr;
}

template<>
constexpr StarpuCodelet *sumnorm_codelet<fp32_t>()
{
    return &sumnorm_codelet_fp32;
}

template<>
constexpr StarpuCodelet *sumnorm_codelet<fp64_t>()
{
    return &sumnorm_codelet_fp64;
}

template<typename T>
void sumnorm(Index m, Index n, Index k, starpu_data_handle_t src,
        starpu_data_handle_t dst)
//! Insert sumnorm task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    auto args = new sumnorm_args
    {
        .m = m,
        .n = n,
        .k = k
    };
    //fp64_t nflops = m * n * k;
    // Submit task
    int ret = starpu_task_insert(sumnorm_codelet<T>(),
            STARPU_R, src,
            STARPU_CL_ARGS, args, sizeof(*args),
            Starpu::STARPU_RW_COMMUTE, dst,
            //STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in sumnorm task submission");
    }
}

// Explicit instantiation
template
void sumnorm<fp32_t>(Index m, Index n, Index k, starpu_data_handle_t src,
        starpu_data_handle_t dst);

template
void sumnorm<fp64_t>(Index m, Index n, Index k, starpu_data_handle_t src,
        starpu_data_handle_t dst);

} // namespace starpu
} // namespace nntile

