/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file src/starpu/torch_add.cc
 * Torch-native add StarPU codelet (CPU/CUDA aten::add.out).
 *
 * @version 1.1.0
 */

#include "nntile/starpu/torch_add.hh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/core/grad_mode.h>
#include <ATen/ops/add.h>

#include "nntile/starpu/torch_blob.hh"
#ifdef NNTILE_USE_CUDA
#include "nntile/starpu/torch_cuda_env.hh"
#endif

namespace nntile::starpu
{

namespace
{

using torch_blob::blob_fp32;
using torch_blob::to_i64;

void run_add(
    const TorchAdd<std::tuple<fp32_t>>::args_t *args,
    float *self_ptr,
    float *other_ptr,
    float *out_ptr,
    c10::optional<at::Device> device = c10::nullopt)
{
    const Index ndim = args->ndim;
    auto sizes = to_i64(args->sizes, ndim);
    auto self_strides = to_i64(args->self_strides, ndim);
    auto other_strides = to_i64(args->other_strides, ndim);
    auto out_strides = to_i64(args->out_strides, ndim);

    at::Tensor self = blob_fp32(
        self_ptr,
        sizes,
        self_strides,
        device);
    at::Tensor other = blob_fp32(
        other_ptr,
        sizes,
        other_strides,
        device);
    at::Tensor out = blob_fp32(
        out_ptr,
        sizes,
        out_strides,
        device);

    at::AutoDispatchBelowADInplaceOrView guard;
    at::NoGradGuard no_grad;
    at::add_out(
        out,
        self,
        other,
        static_cast<double>(args->alpha));
}

} // namespace

template<typename T>
TorchAdd<std::tuple<T>>::TorchAdd():
    codelet("nntile_torch_add", footprint, cpu_funcs, cuda_funcs)
{
}

template<>
void TorchAdd<std::tuple<fp32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        auto *args = reinterpret_cast<args_t *>(cl_args);
        auto **ifaces =
            reinterpret_cast<VariableInterface **>(buffers);
        run_add(
            args,
            ifaces[0]->get_ptr<float>(),
            ifaces[1]->get_ptr<float>(),
            ifaces[2]->get_ptr<float>(),
            at::kCPU);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_add CPU codelet failed: %s\n",
            ex.what());
        std::abort();
    }
#endif
}

#ifdef NNTILE_USE_CUDA
template<>
void TorchAdd<std::tuple<fp32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        TorchCudaEnv cuda_env;
        auto *args = reinterpret_cast<args_t *>(cl_args);
        auto **ifaces =
            reinterpret_cast<VariableInterface **>(buffers);
        run_add(
            args,
            ifaces[0]->get_ptr<float>(),
            ifaces[1]->get_ptr<float>(),
            ifaces[2]->get_ptr<float>(),
            cuda_env.device());
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_add CUDA codelet failed: %s\n",
            ex.what());
        std::abort();
    }
#endif
}
#endif // NNTILE_USE_CUDA

template<typename T>
uint32_t TorchAdd<std::tuple<T>>::footprint(struct starpu_task *task)
{
    auto *args = reinterpret_cast<args_t *>(task->cl_arg);
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(
        &args->ndim,
        sizeof(args->ndim),
        hash);
    hash = starpu_hash_crc32c_be_n(
        args->sizes,
        sizeof(Index) * static_cast<size_t>(args->ndim),
        hash);
    return hash;
}

template<typename T>
void TorchAdd<std::tuple<T>>::submit(
    int starpu_worker_hint,
    const args_t &meta,
    Handle self,
    Handle other,
    Handle out
)
{
    if (meta.ndim < 0 || meta.ndim > torch_add_max_ndim)
    {
        throw std::runtime_error(
            "torch_add.submit: ndim out of range");
    }
    args_t *args =
        reinterpret_cast<args_t *>(std::malloc(sizeof(*args)));
    if (args == nullptr)
    {
        throw std::runtime_error("torch_add.submit: malloc failed");
    }
    *args = meta;

    Index nelems = 1;
    for (Index i = 0; i < meta.ndim; ++i)
    {
        nelems *= meta.sizes[i];
    }
    double nflops =
        static_cast<double>(sizeof(T)) * 3.0 *
        static_cast<double>(nelems);

    int ret = nntile_starpu_task_insert(
        &codelet,
        starpu_worker_hint,
        STARPU_R,
        self.get(),
        STARPU_R,
        other.get(),
        STARPU_CL_ARGS,
        args,
        sizeof(*args),
        STARPU_W,
        out.get(),
        STARPU_FLOPS,
        nflops,
        0);
    if (ret != 0)
    {
        throw std::runtime_error(
            "nntile::starpu::torch_add.submit failed");
    }
}

template class TorchAdd<std::tuple<nntile::fp32_t>>;

torch_add_pack_t torch_add;

} // namespace nntile::starpu
