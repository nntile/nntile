/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file src/starpu/torch_dispatch.cc
 * Torch-native family StarPU codelets (CPU aten *_out).
 *
 * @version 1.1.0
 */

#include "nntile/starpu/torch_dispatch.hh"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/core/grad_mode.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/bmm.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/embedding.h>
#include <ATen/ops/gelu.h>
#include <ATen/ops/gelu_backward.h>
#include <ATen/ops/hypot.h>
#include <ATen/ops/linalg_vector_norm.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/narrow_copy.h>
#include <ATen/ops/native_layer_norm.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/repeat.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <ATen/ops/silu.h>
#include <ATen/ops/silu_backward.h>
#include <ATen/ops/softmax.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/threshold_backward.h>
#include <ATen/ops/transpose_copy.h>

#include "nntile/starpu/torch_blob.hh"

namespace nntile::starpu
{

namespace
{

using torch_blob::blob_fp32;
using torch_blob::blob_i64;
using torch_blob::to_i64;

std::vector<std::int64_t> sizes_of(
    const TorchDispatchArgs &args,
    Index slot,
    bool is_out)
{
    const Index ndim = is_out ? args.out_ndim[slot] : args.in_ndim[slot];
    const Index *raw = is_out ? args.out_sizes[slot] : args.in_sizes[slot];
    return to_i64(raw, ndim);
}

std::vector<std::int64_t> strides_of(
    const TorchDispatchArgs &args,
    Index slot,
    bool is_out)
{
    const Index ndim = is_out ? args.out_ndim[slot] : args.in_ndim[slot];
    const Index *raw =
        is_out ? args.out_strides[slot] : args.in_strides[slot];
    return to_i64(raw, ndim);
}

at::Tensor in_fp32(
    float *ptr,
    const TorchDispatchArgs &args,
    Index slot)
{
    return blob_fp32(
        ptr,
        sizes_of(args, slot, false),
        strides_of(args, slot, false));
}

at::Tensor out_fp32(
    float *ptr,
    const TorchDispatchArgs &args,
    Index slot)
{
    return blob_fp32(
        ptr,
        sizes_of(args, slot, true),
        strides_of(args, slot, true));
}

void run_unary(TorchDispatchArgs *args, float *in, float *out)
{
    at::Tensor self = in_fp32(in, *args, 0);
    at::Tensor result = out_fp32(out, *args, 0);
    switch (args->kind)
    {
    case TorchKind::Relu:
        at::relu_out(result, self);
        break;
    case TorchKind::Silu:
        at::silu_out(result, self);
        break;
    case TorchKind::Gelu:
        at::gelu_out(
            result,
            self,
            args->iargs[0] ? "tanh" : "none");
        break;
    case TorchKind::Softmax:
        at::_softmax_out(
            result,
            self,
            static_cast<std::int64_t>(args->iargs[0]),
            /*half_to_float=*/false);
        break;
    case TorchKind::Sum:
    {
        std::vector<std::int64_t> dims;
        const Index nd = args->iargs[0];
        for (Index i = 0; i < nd; ++i)
        {
            dims.push_back(static_cast<std::int64_t>(args->iargs[2 + i]));
        }
        const bool keepdim = args->iargs[1] != 0;
        if (dims.empty())
        {
            at::sum_out(result, self);
        }
        else
        {
            at::sum_out(result, self, dims, keepdim);
        }
        break;
    }
    case TorchKind::VectorNorm:
    {
        const std::int64_t dim = static_cast<std::int64_t>(args->iargs[2]);
        const bool keepdim = args->iargs[1] != 0;
        at::linalg_vector_norm_out(
            result,
            self,
            /*ord=*/2.0,
            at::OptionalIntArrayRef({dim}),
            keepdim,
            /*dtype=*/c10::nullopt);
        break;
    }
    case TorchKind::NarrowCopy:
    {
        const std::int64_t dim = static_cast<std::int64_t>(args->iargs[0]);
        const std::int64_t start = static_cast<std::int64_t>(args->iargs[1]);
        const std::int64_t length = static_cast<std::int64_t>(args->iargs[2]);
        at::narrow_copy_out(result, self, dim, start, length);
        break;
    }
    case TorchKind::Repeat:
    {
        std::vector<std::int64_t> repeats;
        const Index ndim = args->in_ndim[0];
        for (Index i = 0; i < ndim; ++i)
        {
            repeats.push_back(static_cast<std::int64_t>(args->iargs[i]));
        }
        at::repeat_out(result, self, repeats);
        break;
    }
    case TorchKind::MulScalar:
        at::mul_out(
            result,
            self,
            static_cast<double>(args->scalars[0]));
        break;
    case TorchKind::TransposeCopy:
    {
        const std::int64_t d0 =
            static_cast<std::int64_t>(args->iargs[0]);
        const std::int64_t d1 =
            static_cast<std::int64_t>(args->iargs[1]);
        at::transpose_copy_out(result, self, d0, d1);
        break;
    }
    default:
        throw std::runtime_error("torch_unary: unsupported kind");
    }
}

void run_binary(
    TorchDispatchArgs *args,
    float *a,
    float *b,
    float *out)
{
    at::Tensor ta = in_fp32(a, *args, 0);
    at::Tensor tb = in_fp32(b, *args, 1);
    at::Tensor result = out_fp32(out, *args, 0);
    switch (args->kind)
    {
    case TorchKind::Mul:
        at::mul_out(result, ta, tb);
        break;
    case TorchKind::Hypot:
        at::hypot_out(result, ta, tb);
        break;
    case TorchKind::ThresholdBackward:
        at::threshold_backward_out(
            result,
            ta,
            tb,
            static_cast<double>(args->scalars[0]));
        break;
    case TorchKind::SiluBackward:
        at::silu_backward_out(result, ta, tb);
        break;
    case TorchKind::GeluBackward:
        at::gelu_backward_out(
            result,
            ta,
            tb,
            args->iargs[0] ? "tanh" : "none");
        break;
    case TorchKind::SoftmaxBackward:
        at::_softmax_backward_data_out(
            result,
            ta,
            tb,
            static_cast<std::int64_t>(args->iargs[0]),
            ta.scalar_type());
        break;
    case TorchKind::Mm:
        at::mm_out(result, ta, tb);
        break;
    case TorchKind::Bmm:
        at::bmm_out(result, ta, tb);
        break;
    case TorchKind::Matmul:
        at::matmul_out(result, ta, tb);
        break;
    case TorchKind::Linear:
        // weight is tb (out_features, in_features); bias optional via
        // ternary.
        at::linear_out(result, ta, tb, c10::nullopt);
        break;
    default:
        throw std::runtime_error("torch_binary: unsupported kind");
    }
}

void run_ternary(
    TorchDispatchArgs *args,
    float *a,
    float *b,
    float *c,
    float *out)
{
    at::Tensor ta = in_fp32(a, *args, 0);
    at::Tensor tb = in_fp32(b, *args, 1);
    at::Tensor tc = in_fp32(c, *args, 2);
    at::Tensor result = out_fp32(out, *args, 0);
    switch (args->kind)
    {
    case TorchKind::Addmm:
        at::addmm_out(
            result,
            ta,
            tb,
            tc,
            static_cast<double>(args->scalars[0]),
            static_cast<double>(args->scalars[1]));
        break;
    case TorchKind::Linear:
        at::linear_out(result, ta, tb, tc);
        break;
    case TorchKind::Sdpa:
    {
        // No public *_out for SDPA; materialize into preallocated out.
        const bool is_causal = args->iargs[1] != 0;
        auto attn = at::scaled_dot_product_attention(
            ta,
            tb,
            tc,
            /*attn_mask=*/c10::nullopt,
            /*dropout_p=*/0.0,
            is_causal,
            /*scale=*/c10::nullopt);
        result.copy_(attn);
        break;
    }
    default:
        throw std::runtime_error("torch_ternary: unsupported kind");
    }
}

uint32_t args_footprint(const TorchDispatchArgs *args)
{
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->kind, sizeof(args->kind), hash);
    hash = starpu_hash_crc32c_be_n(&args->n_in, sizeof(args->n_in), hash);
    hash = starpu_hash_crc32c_be_n(
        args->in_sizes[0],
        sizeof(Index) * static_cast<size_t>(args->in_ndim[0]),
        hash);
    return hash;
}

TorchDispatchArgs *clone_args(const TorchDispatchArgs &meta)
{
    auto *args = reinterpret_cast<TorchDispatchArgs *>(
        std::malloc(sizeof(TorchDispatchArgs)));
    if (args == nullptr)
    {
        throw std::runtime_error("torch_dispatch: malloc failed");
    }
    *args = meta;
    return args;
}

} // namespace

template<typename T>
TorchUnary<std::tuple<T>>::TorchUnary():
    codelet("nntile_torch_unary", footprint, cpu_funcs, cuda_funcs)
{
    codelet.restrict_where(STARPU_CPU);
}

template<>
void TorchUnary<std::tuple<fp32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        auto *args = reinterpret_cast<args_t *>(cl_args);
        auto **ifaces =
            reinterpret_cast<VariableInterface **>(buffers);
        float *in = ifaces[0]->get_ptr<float>();
        float *out = ifaces[1]->get_ptr<float>();
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        run_unary(args, in, out);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_unary failed: %s\n",
            ex.what());
        std::abort();
    }
#endif
}

template<typename T>
uint32_t TorchUnary<std::tuple<T>>::footprint(struct starpu_task *task)
{
    return args_footprint(
        reinterpret_cast<args_t *>(task->cl_arg));
}

template<typename T>
void TorchUnary<std::tuple<T>>::submit(
    int starpu_worker_hint,
    const args_t &meta,
    Handle in,
    Handle out
)
{
    args_t *args = clone_args(meta);
    int ret = nntile_starpu_task_insert(
        &codelet,
        starpu_worker_hint,
        STARPU_R,
        in.get(),
        STARPU_CL_ARGS,
        args,
        sizeof(*args),
        STARPU_W,
        out.get(),
        0);
    if (ret != 0)
    {
        throw std::runtime_error("torch_unary.submit failed");
    }
}

template<typename T>
TorchBinary<std::tuple<T>>::TorchBinary():
    codelet("nntile_torch_binary", footprint, cpu_funcs, cuda_funcs)
{
    codelet.restrict_where(STARPU_CPU);
}

template<>
void TorchBinary<std::tuple<fp32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        auto *args = reinterpret_cast<args_t *>(cl_args);
        auto **ifaces =
            reinterpret_cast<VariableInterface **>(buffers);
        float *a = ifaces[0]->get_ptr<float>();
        float *b = ifaces[1]->get_ptr<float>();
        float *out = ifaces[2]->get_ptr<float>();
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        run_binary(args, a, b, out);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_binary failed: %s\n",
            ex.what());
        std::abort();
    }
#endif
}

template<typename T>
uint32_t TorchBinary<std::tuple<T>>::footprint(struct starpu_task *task)
{
    return args_footprint(
        reinterpret_cast<args_t *>(task->cl_arg));
}

template<typename T>
void TorchBinary<std::tuple<T>>::submit(
    int starpu_worker_hint,
    const args_t &meta,
    Handle a,
    Handle b,
    Handle out
)
{
    args_t *args = clone_args(meta);
    int ret = nntile_starpu_task_insert(
        &codelet,
        starpu_worker_hint,
        STARPU_R,
        a.get(),
        STARPU_R,
        b.get(),
        STARPU_CL_ARGS,
        args,
        sizeof(*args),
        STARPU_W,
        out.get(),
        0);
    if (ret != 0)
    {
        throw std::runtime_error("torch_binary.submit failed");
    }
}

template<typename T>
TorchTernary<std::tuple<T>>::TorchTernary():
    codelet("nntile_torch_ternary", footprint, cpu_funcs, cuda_funcs)
{
    codelet.restrict_where(STARPU_CPU);
}

template<>
void TorchTernary<std::tuple<fp32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        auto *args = reinterpret_cast<args_t *>(cl_args);
        auto **ifaces =
            reinterpret_cast<VariableInterface **>(buffers);
        float *a = ifaces[0]->get_ptr<float>();
        float *b = ifaces[1]->get_ptr<float>();
        float *c = ifaces[2]->get_ptr<float>();
        // iargs[7]: out aliases first input (STARPU_RW on a).
        float *out = (args->iargs[7] != 0)
            ? a
            : ifaces[3]->get_ptr<float>();
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        run_ternary(args, a, b, c, out);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_ternary failed: %s\n",
            ex.what());
        std::abort();
    }
#endif
}

template<typename T>
uint32_t TorchTernary<std::tuple<T>>::footprint(struct starpu_task *task)
{
    return args_footprint(
        reinterpret_cast<args_t *>(task->cl_arg));
}

template<typename T>
void TorchTernary<std::tuple<T>>::submit(
    int starpu_worker_hint,
    const args_t &meta,
    Handle a,
    Handle b,
    Handle c,
    Handle out
)
{
    args_t *args = clone_args(meta);
    // Same StarPU handle as both read input and write output must use
    // STARPU_RW once (e.g. addmm accumulate into C). Do not submit the
    // handle twice as STARPU_R + STARPU_W.
    if (out.get() == b.get() || out.get() == c.get())
    {
        std::free(args);
        throw std::runtime_error(
            "torch_ternary.submit: out may only alias the first "
            "input (STARPU_RW); aliasing mat1/mat2 is unsupported");
    }
    const bool out_aliases_a = (out.get() == a.get());
    args->iargs[7] = out_aliases_a ? 1 : 0;
    int ret = 0;
    if (out_aliases_a)
    {
        ret = nntile_starpu_task_insert(
            &codelet,
            starpu_worker_hint,
            STARPU_RW,
            a.get(),
            STARPU_R,
            b.get(),
            STARPU_R,
            c.get(),
            STARPU_CL_ARGS,
            args,
            sizeof(*args),
            0);
    }
    else
    {
        ret = nntile_starpu_task_insert(
            &codelet,
            starpu_worker_hint,
            STARPU_R,
            a.get(),
            STARPU_R,
            b.get(),
            STARPU_R,
            c.get(),
            STARPU_CL_ARGS,
            args,
            sizeof(*args),
            STARPU_W,
            out.get(),
            0);
    }
    if (ret != 0)
    {
        throw std::runtime_error("torch_ternary.submit failed");
    }
}

TorchEmbedding::TorchEmbedding():
    codelet("nntile_torch_embedding", footprint, cpu_funcs, cuda_funcs)
{
    codelet.restrict_where(STARPU_CPU);
}

void TorchEmbedding::cpu(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        auto *args = reinterpret_cast<args_t *>(cl_args);
        auto **ifaces =
            reinterpret_cast<VariableInterface **>(buffers);
        float *weight = ifaces[0]->get_ptr<float>();
        auto *indices = ifaces[1]->get_ptr<std::int64_t>();
        float *out = ifaces[2]->get_ptr<float>();
        at::Tensor w = in_fp32(weight, *args, 0);
        at::Tensor idx = blob_i64(
            indices,
            sizes_of(*args, 1, false),
            strides_of(*args, 1, false));
        at::Tensor result = out_fp32(out, *args, 0);
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        at::embedding_out(
            result,
            w,
            idx,
            /*padding_idx=*/-1,
            /*scale_grad_by_freq=*/false,
            /*sparse=*/false);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_embedding failed: %s\n",
            ex.what());
        std::abort();
    }
#endif
}

uint32_t TorchEmbedding::footprint(struct starpu_task *task)
{
    return args_footprint(
        reinterpret_cast<args_t *>(task->cl_arg));
}

void TorchEmbedding::submit(
    int starpu_worker_hint,
    const args_t &meta,
    Handle weight,
    Handle indices,
    Handle out
)
{
    args_t *args = clone_args(meta);
    int ret = nntile_starpu_task_insert(
        &codelet,
        starpu_worker_hint,
        STARPU_R,
        weight.get(),
        STARPU_R,
        indices.get(),
        STARPU_CL_ARGS,
        args,
        sizeof(*args),
        STARPU_W,
        out.get(),
        0);
    if (ret != 0)
    {
        throw std::runtime_error("torch_embedding.submit failed");
    }
}

TorchCat::TorchCat():
    codelet("nntile_torch_cat", footprint, cpu_funcs, cuda_funcs)
{
    codelet.restrict_where(STARPU_CPU);
}

void TorchCat::cpu(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        auto *args = reinterpret_cast<args_t *>(cl_args);
        auto **ifaces =
            reinterpret_cast<VariableInterface **>(buffers);
        const Index n = args->iargs[1];
        std::vector<at::Tensor> inputs;
        inputs.reserve(static_cast<size_t>(n));
        for (Index i = 0; i < n; ++i)
        {
            float *ptr = ifaces[i]->get_ptr<float>();
            inputs.push_back(in_fp32(ptr, *args, i));
        }
        float *out_ptr = ifaces[n]->get_ptr<float>();
        at::Tensor result = out_fp32(out_ptr, *args, 0);
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        at::cat_out(
            result,
            inputs,
            static_cast<std::int64_t>(args->iargs[0]));
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_cat failed: %s\n",
            ex.what());
        std::abort();
    }
#endif
}

uint32_t TorchCat::footprint(struct starpu_task *task)
{
    return args_footprint(
        reinterpret_cast<args_t *>(task->cl_arg));
}

void TorchCat::submit(
    int starpu_worker_hint,
    const args_t &meta,
    const std::vector<Handle> &inputs,
    Handle out
)
{
    if (inputs.empty() ||
        inputs.size() > static_cast<size_t>(torch_dispatch_max_tensors))
    {
        throw std::runtime_error("torch_cat.submit: bad input count");
    }
    args_t *args = clone_args(meta);
    // Build varargs task_insert: R for each input, then CL_ARGS, W out.
    starpu_data_handle_t handles[torch_dispatch_max_tensors];
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        handles[i] = inputs[i].get();
    }
    // Use packed insert via repeated API — starpu_task_insert varargs.
    // Fall back to sequential narrow copies if n is awkward: for n<=8
    // build a one-shot insert with a fixed switch.
    const Index n = static_cast<Index>(inputs.size());
    int ret = 0;
    switch (n)
    {
    case 1:
        ret = nntile_starpu_task_insert(
            &codelet,
            starpu_worker_hint,
            STARPU_R,
            handles[0],
            STARPU_CL_ARGS,
            args,
            sizeof(*args),
            STARPU_W,
            out.get(),
            0);
        break;
    case 2:
        ret = nntile_starpu_task_insert(
            &codelet,
            starpu_worker_hint,
            STARPU_R,
            handles[0],
            STARPU_R,
            handles[1],
            STARPU_CL_ARGS,
            args,
            sizeof(*args),
            STARPU_W,
            out.get(),
            0);
        break;
    case 3:
        ret = nntile_starpu_task_insert(
            &codelet,
            starpu_worker_hint,
            STARPU_R,
            handles[0],
            STARPU_R,
            handles[1],
            STARPU_R,
            handles[2],
            STARPU_CL_ARGS,
            args,
            sizeof(*args),
            STARPU_W,
            out.get(),
            0);
        break;
    case 4:
        ret = nntile_starpu_task_insert(
            &codelet,
            starpu_worker_hint,
            STARPU_R,
            handles[0],
            STARPU_R,
            handles[1],
            STARPU_R,
            handles[2],
            STARPU_R,
            handles[3],
            STARPU_CL_ARGS,
            args,
            sizeof(*args),
            STARPU_W,
            out.get(),
            0);
        break;
    default:
        std::free(args);
        throw std::runtime_error(
            "torch_cat.submit: only up to 4 inputs in first cut");
    }
    if (ret != 0)
    {
        throw std::runtime_error("torch_cat.submit failed");
    }
}

template class TorchUnary<std::tuple<nntile::fp32_t>>;
template class TorchBinary<std::tuple<nntile::fp32_t>>;
template class TorchTernary<std::tuple<nntile::fp32_t>>;

TorchLayerNorm::TorchLayerNorm():
    codelet("nntile_torch_layer_norm", footprint, cpu_funcs, cuda_funcs)
{
    codelet.restrict_where(STARPU_CPU);
}

void TorchLayerNorm::cpu(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        auto *args = reinterpret_cast<args_t *>(cl_args);
        auto **ifaces =
            reinterpret_cast<VariableInterface **>(buffers);
        float *in_ptr = ifaces[0]->get_ptr<float>();
        float *out_ptr = ifaces[1]->get_ptr<float>();
        float *mean_ptr = ifaces[2]->get_ptr<float>();
        float *rstd_ptr = ifaces[3]->get_ptr<float>();
        const bool has_w = args->iargs[1] != 0;
        const bool has_b = args->iargs[2] != 0;
        Index buf = 4;
        at::Tensor weight;
        at::Tensor bias;
        if (has_w)
        {
            weight = in_fp32(ifaces[buf++]->get_ptr<float>(), *args, 1);
        }
        if (has_b)
        {
            bias = in_fp32(ifaces[buf++]->get_ptr<float>(), *args, 2);
        }
        at::Tensor input = in_fp32(in_ptr, *args, 0);
        at::Tensor out = out_fp32(out_ptr, *args, 0);
        at::Tensor mean = out_fp32(mean_ptr, *args, 1);
        at::Tensor rstd = out_fp32(rstd_ptr, *args, 2);
        const std::int64_t n = static_cast<std::int64_t>(args->iargs[0]);
        std::vector<std::int64_t> normalized_shape;
        for (std::int64_t i = 0; i < n; ++i)
        {
            normalized_shape.push_back(
                input.size(input.dim() - n + i));
        }
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        auto ln = at::native_layer_norm(
            input,
            normalized_shape,
            has_w ? c10::optional<at::Tensor>(weight) : c10::nullopt,
            has_b ? c10::optional<at::Tensor>(bias) : c10::nullopt,
            static_cast<double>(args->scalars[0]));
        out.copy_(std::get<0>(ln));
        mean.copy_(std::get<1>(ln));
        rstd.copy_(std::get<2>(ln));
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_layer_norm failed: %s\n",
            ex.what());
        std::abort();
    }
#endif
}

uint32_t TorchLayerNorm::footprint(struct starpu_task *task)
{
    return args_footprint(
        reinterpret_cast<args_t *>(task->cl_arg));
}

void TorchLayerNorm::submit(
    int starpu_worker_hint,
    const args_t &meta,
    Handle input,
    Handle weight,
    Handle bias,
    Handle out,
    Handle mean,
    Handle rstd,
    bool has_weight,
    bool has_bias
)
{
    args_t *args = clone_args(meta);
    args->iargs[1] = has_weight ? 1 : 0;
    args->iargs[2] = has_bias ? 1 : 0;
    int ret = 0;
    if (has_weight && has_bias)
    {
        ret = nntile_starpu_task_insert(
            &codelet,
            starpu_worker_hint,
            STARPU_R,
            input.get(),
            STARPU_CL_ARGS,
            args,
            sizeof(*args),
            STARPU_W,
            out.get(),
            STARPU_W,
            mean.get(),
            STARPU_W,
            rstd.get(),
            STARPU_R,
            weight.get(),
            STARPU_R,
            bias.get(),
            0);
    }
    else if (has_weight)
    {
        ret = nntile_starpu_task_insert(
            &codelet,
            starpu_worker_hint,
            STARPU_R,
            input.get(),
            STARPU_CL_ARGS,
            args,
            sizeof(*args),
            STARPU_W,
            out.get(),
            STARPU_W,
            mean.get(),
            STARPU_W,
            rstd.get(),
            STARPU_R,
            weight.get(),
            0);
    }
    else
    {
        ret = nntile_starpu_task_insert(
            &codelet,
            starpu_worker_hint,
            STARPU_R,
            input.get(),
            STARPU_CL_ARGS,
            args,
            sizeof(*args),
            STARPU_W,
            out.get(),
            STARPU_W,
            mean.get(),
            STARPU_W,
            rstd.get(),
            0);
    }
    if (ret != 0)
    {
        throw std::runtime_error("torch_layer_norm.submit failed");
    }
}

torch_unary_pack_t torch_unary;
torch_binary_pack_t torch_binary;
torch_ternary_pack_t torch_ternary;
TorchEmbedding torch_embedding;
TorchLayerNorm torch_layer_norm;
TorchCat torch_cat;

} // namespace nntile::starpu
