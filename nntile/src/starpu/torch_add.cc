/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file src/starpu/torch_add.cc
 * Torch-native add StarPU codelet (CPU aten::add.out).
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
#include <ATen/ops/from_blob.h>

namespace nntile::starpu
{

namespace
{

// Use std::int64_t (not nntile::int64_t) for at::IntArrayRef.
std::vector<std::int64_t> to_i64(const Index *data, Index n)
{
    std::vector<std::int64_t> out(static_cast<size_t>(n));
    for (Index i = 0; i < n; ++i)
    {
        out[static_cast<size_t>(i)] =
            static_cast<std::int64_t>(data[i]);
    }
    return out;
}

at::Tensor blob_tensor(
    float *ptr,
    const std::vector<std::int64_t> &sizes,
    const std::vector<std::int64_t> &strides,
    const at::TensorOptions &opts)
{
    return at::from_blob(
        ptr,
        at::IntArrayRef(sizes),
        at::IntArrayRef(strides),
        /*deleter=*/[](void *) {},
        opts);
}

} // namespace

template<typename T>
TorchAdd<std::tuple<T>>::TorchAdd():
    codelet("nntile_torch_add", footprint, cpu_funcs, cuda_funcs)
{
    codelet.restrict_where(STARPU_CPU);
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
        float *self_ptr = ifaces[0]->get_ptr<float>();
        float *other_ptr = ifaces[1]->get_ptr<float>();
        float *out_ptr = ifaces[2]->get_ptr<float>();

        const Index ndim = args->ndim;
        auto sizes = to_i64(args->sizes, ndim);
        auto self_strides = to_i64(args->self_strides, ndim);
        auto other_strides = to_i64(args->other_strides, ndim);
        auto out_strides = to_i64(args->out_strides, ndim);

        auto opts = at::TensorOptions()
            .dtype(at::kFloat)
            .device(at::kCPU);

        // Empty deleter: StarPU owns the buffers.
        at::Tensor self = blob_tensor(
            self_ptr,
            sizes,
            self_strides,
            opts);
        at::Tensor other = blob_tensor(
            other_ptr,
            sizes,
            other_strides,
            opts);
        at::Tensor out = blob_tensor(
            out_ptr,
            sizes,
            out_strides,
            opts);

        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        at::add_out(
            out,
            self,
            other,
            static_cast<double>(args->alpha));
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
