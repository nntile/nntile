/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file src/starpu/swap_two_axes.cc
 * swap_two_axes operation for StarPU buffers.
 *
 * @version 1.1.0
 * */

#include "nntile/starpu/swap_two_axes.hh"

#include <cstdlib>
#include <stdexcept>

#include "nntile/kernel/swap_two_axes.hh"

namespace nntile::starpu
{

template<typename T>
SwapTwoAxes<std::tuple<T>>::SwapTwoAxes():
    codelet("nntile_swap_two_axes", footprint, cpu_funcs, cuda_funcs)
{
}

template<typename T>
void SwapTwoAxes<std::tuple<T>>::cpu(
    void *buffers[],
    void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    auto args = reinterpret_cast<args_t *>(cl_args);
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    T *dst = interfaces[1]->get_ptr<T>();
    kernel::swap_two_axes::cpu<T>(
        args->d0,
        args->d1,
        args->d2,
        args->d3,
        args->d4,
        src,
        dst);
#endif
}

template<>
void SwapTwoAxes<std::tuple<fp32_fast_tf32_t>>::cpu(
    void *buffers[],
    void *cl_args) noexcept
{
    SwapTwoAxes<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void SwapTwoAxes<std::tuple<fp32_fast_fp16_t>>::cpu(
    void *buffers[],
    void *cl_args) noexcept
{
    SwapTwoAxes<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void SwapTwoAxes<std::tuple<fp32_fast_bf16_t>>::cpu(
    void *buffers[],
    void *cl_args) noexcept
{
    SwapTwoAxes<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
template<typename T>
void SwapTwoAxes<std::tuple<T>>::cuda(
    void *buffers[],
    void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    auto args = reinterpret_cast<args_t *>(cl_args);
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    T *dst = interfaces[1]->get_ptr<T>();
    cudaStream_t stream = starpu_cuda_get_local_stream();
    kernel::swap_two_axes::cuda<T>(
        stream,
        args->d0,
        args->d1,
        args->d2,
        args->d3,
        args->d4,
        src,
        dst);
#endif
}

template<>
void SwapTwoAxes<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args) noexcept
{
    SwapTwoAxes<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void SwapTwoAxes<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args) noexcept
{
    SwapTwoAxes<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void SwapTwoAxes<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args) noexcept
{
    SwapTwoAxes<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

template<typename T>
uint32_t SwapTwoAxes<std::tuple<T>>::footprint(struct starpu_task *task)
{
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->d0, sizeof(args->d0), hash);
    hash = starpu_hash_crc32c_be_n(&args->d1, sizeof(args->d1), hash);
    hash = starpu_hash_crc32c_be_n(&args->d2, sizeof(args->d2), hash);
    hash = starpu_hash_crc32c_be_n(&args->d3, sizeof(args->d3), hash);
    hash = starpu_hash_crc32c_be_n(&args->d4, sizeof(args->d4), hash);
    return hash;
}

template<typename T>
void SwapTwoAxes<std::tuple<T>>::submit(
    int starpu_worker_hint,
    Index d0,
    Index d1,
    Index d2,
    Index d3,
    Index d4,
    Handle src,
    Handle dst)
{
    args_t *args = static_cast<args_t *>(std::malloc(sizeof(*args)));
    args->d0 = d0;
    args->d1 = d1;
    args->d2 = d2;
    args->d3 = d3;
    args->d4 = d4;
    const double nflops = static_cast<double>(sizeof(T)) * 2.0 *
        static_cast<double>(d0) * static_cast<double>(d1) *
        static_cast<double>(d2) * static_cast<double>(d3) *
        static_cast<double>(d4);
    const int ret = nntile_starpu_task_insert(
        &codelet,
        starpu_worker_hint,
        STARPU_R,
        src.get(),
        STARPU_W,
        dst.get(),
        STARPU_CL_ARGS,
        args,
        sizeof(*args),
        STARPU_FLOPS,
        nflops,
        0);
    if (ret != 0)
    {
        throw std::runtime_error("Error in swap_two_axes task submission");
    }
}

template class SwapTwoAxes<std::tuple<nntile::fp64_t>>;
template class SwapTwoAxes<std::tuple<nntile::fp32_t>>;
template class SwapTwoAxes<std::tuple<nntile::fp32_fast_tf32_t>>;
template class SwapTwoAxes<std::tuple<nntile::fp32_fast_fp16_t>>;
template class SwapTwoAxes<std::tuple<nntile::fp32_fast_bf16_t>>;
template class SwapTwoAxes<std::tuple<nntile::bf16_t>>;
template class SwapTwoAxes<std::tuple<nntile::fp16_t>>;

swap_two_axes_pack_t swap_two_axes;

} // namespace nntile::starpu
