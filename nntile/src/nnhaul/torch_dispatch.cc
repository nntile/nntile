/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file src/nnhaul/torch_dispatch.cc
 * Family NNHaul codelets for torch-native aten kernels (CPU/CUDA).
 *
 * The kernel bodies are the same at::*_out calls as the StarPU families
 * (src/starpu/torch_dispatch.cc); only the substrate differs: buffers
 * come through nntile::haul::buf_as, CUDA binds to the NNHaul worker
 * stream via HaulTorchCudaEnv, and submit copies the packed args into
 * the runtime bag synchronously (nnhaul::insert takes a copy, so no
 * STARPU_CL_ARGS clone is needed).
 *
 * @version 1.1.0
 */

#include "nntile/nnhaul/torch_dispatch.hh"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/core/grad_mode.h>
#include <ATen/ops/_adaptive_avg_pool2d.h>
#include <ATen/ops/avg_pool2d.h>
#include <ATen/ops/convolution.h>
#include <ATen/ops/convolution_backward.h>
#include <ATen/ops/max_pool2d_with_indices.h>
#include <ATen/ops/max_pool2d_with_indices_backward.h>
#include <ATen/ops/upsample_bilinear2d.h>
#include <ATen/ops/upsample_bilinear2d_backward.h>
#include <ATen/ops/upsample_nearest2d.h>
#include <ATen/ops/upsample_nearest2d_backward.h>
#include <c10/util/Optional.h>

#ifdef NNTILE_USE_CUDA
#include "nntile/nnhaul/torch_cuda_env.hh"
#endif
#include "nntile/starpu/torch_blob.hh"

namespace nntile::haul
{

namespace
{

using starpu::torch_blob::blob_bool;
using starpu::torch_blob::blob_fp32;
using starpu::torch_blob::blob_i64;
using starpu::torch_blob::to_i64;
using starpu::TorchDispatchArgs;
using starpu::TorchKind;

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
    Index slot,
    c10::optional<at::Device> device = c10::nullopt)
{
    return blob_fp32(
        ptr,
        sizes_of(args, slot, false),
        strides_of(args, slot, false),
        static_cast<std::int64_t>(args.in_offset[slot]),
        device);
}

at::Tensor in_i64(
    std::int64_t *ptr,
    const TorchDispatchArgs &args,
    Index slot,
    c10::optional<at::Device> device = c10::nullopt)
{
    return blob_i64(
        ptr,
        sizes_of(args, slot, false),
        strides_of(args, slot, false),
        static_cast<std::int64_t>(args.in_offset[slot]),
        device);
}

at::Tensor in_bool(
    bool *ptr,
    const TorchDispatchArgs &args,
    Index slot,
    c10::optional<at::Device> device = c10::nullopt)
{
    return blob_bool(
        ptr,
        sizes_of(args, slot, false),
        strides_of(args, slot, false),
        static_cast<std::int64_t>(args.in_offset[slot]),
        device);
}

at::Tensor out_fp32(
    float *ptr,
    const TorchDispatchArgs &args,
    Index slot,
    c10::optional<at::Device> device = c10::nullopt)
{
    return blob_fp32(
        ptr,
        sizes_of(args, slot, true),
        strides_of(args, slot, true),
        static_cast<std::int64_t>(args.out_offset[slot]),
        device);
}

at::Tensor out_bool(
    bool *ptr,
    const TorchDispatchArgs &args,
    Index slot,
    c10::optional<at::Device> device = c10::nullopt)
{
    return blob_bool(
        ptr,
        sizes_of(args, slot, true),
        strides_of(args, slot, true),
        static_cast<std::int64_t>(args.out_offset[slot]),
        device);
}

at::Tensor out_i64(
    std::int64_t *ptr,
    const TorchDispatchArgs &args,
    Index slot,
    c10::optional<at::Device> device = c10::nullopt)
{
    return blob_i64(
        ptr,
        sizes_of(args, slot, true),
        strides_of(args, slot, true),
        static_cast<std::int64_t>(args.out_offset[slot]),
        device);
}

std::vector<std::int64_t> iarg_vec(
    const TorchDispatchArgs &args,
    Index start,
    Index count)
{
    std::vector<std::int64_t> values;
    values.reserve(static_cast<size_t>(count));
    for (Index i = 0; i < count; ++i)
    {
        values.push_back(static_cast<std::int64_t>(args.iargs[start + i]));
    }
    return values;
}

c10::optional<std::int64_t> optional_iarg(
    const TorchDispatchArgs &args,
    Index flag_slot,
    Index value_slot)
{
    if (args.iargs[flag_slot] == 0)
    {
        return c10::nullopt;
    }
    return static_cast<std::int64_t>(args.iargs[value_slot]);
}

c10::optional<double> optional_scale(
    const TorchDispatchArgs &args,
    Index has_slot,
    Index scalar_slot)
{
    if (args.iargs[has_slot] == 0)
    {
        return c10::nullopt;
    }
    return static_cast<double>(args.scalars[scalar_slot]);
}

at::Tensor in_tagged(
    void **buffers,
    const TorchDispatchArgs &args,
    Index slot,
    Index tag)
{
    switch (tag)
    {
    case 0:
        return in_fp32(buf_as<float>(buffers, slot), args, slot);
    case 1:
        return in_i64(buf_as<std::int64_t>(buffers, slot), args, slot);
    case 2:
        return in_bool(buf_as<bool>(buffers, slot), args, slot);
    default:
        throw std::runtime_error("torch_cast: bad src dtype tag");
    }
}

at::Tensor out_tagged(
    void **buffers,
    const TorchDispatchArgs &args,
    Index slot,
    Index tag)
{
    switch (tag)
    {
    case 0:
        return out_fp32(buf_as<float>(buffers, slot), args, slot);
    case 1:
        return out_i64(buf_as<std::int64_t>(buffers, slot), args, slot);
    case 2:
        return out_bool(buf_as<bool>(buffers, slot), args, slot);
    default:
        throw std::runtime_error("torch_cast: bad dst dtype tag");
    }
}

//! CopyIntoView may submit a single RW buffer when src/dst alias.
bool copy_into_view_aliases_in(const TorchDispatchArgs *args)
{
    return args->kind == TorchKind::CopyIntoView &&
        args->iargs[7] != 0;
}

void run_unary(
    TorchDispatchArgs *args,
    float *in,
    float *out,
    at::Device device)
{
    at::Tensor self = in_fp32(in, *args, 0, device);
    at::Tensor result = out_fp32(out, *args, 0, device);
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
    case TorchKind::Cos:
        at::cos_out(result, self);
        break;
    case TorchKind::Sin:
        at::sin_out(result, self);
        break;
    case TorchKind::Neg:
        at::neg_out(result, self);
        break;
    case TorchKind::Rsqrt:
        at::rsqrt_out(result, self);
        break;
    case TorchKind::Exp:
        at::exp_out(result, self);
        break;
    case TorchKind::Log:
        at::log_out(result, self);
        break;
    case TorchKind::Triu:
        at::triu_out(
            result,
            self,
            static_cast<std::int64_t>(args->iargs[0]));
        break;
    case TorchKind::AvgPool2d:
        at::avg_pool2d_out(
            result,
            self,
            iarg_vec(*args, 0, 2),
            iarg_vec(*args, 2, 2),
            iarg_vec(*args, 4, 2),
            args->iargs[6] != 0,
            args->iargs[7] != 0,
            optional_iarg(*args, 8, 9));
        break;
    case TorchKind::AdaptiveAvgPool2d:
        at::_adaptive_avg_pool2d_out(
            result,
            self,
            iarg_vec(*args, 0, 2));
        break;
    case TorchKind::UpsampleNearest2d:
        at::upsample_nearest2d_out(
            result,
            self,
            iarg_vec(*args, 0, 2),
            optional_scale(*args, 2, 0),
            optional_scale(*args, 3, 1));
        break;
    case TorchKind::UpsampleNearest2dBackward:
        at::upsample_nearest2d_backward_out(
            result,
            self,
            iarg_vec(*args, 0, 2),
            iarg_vec(*args, 2, 4),
            optional_scale(*args, 6, 0),
            optional_scale(*args, 7, 1));
        break;
    case TorchKind::UpsampleBilinear2d:
        at::upsample_bilinear2d_out(
            result,
            self,
            iarg_vec(*args, 0, 2),
            args->iargs[2] != 0,
            optional_scale(*args, 3, 0),
            optional_scale(*args, 4, 1));
        break;
    case TorchKind::UpsampleBilinear2dBackward:
        at::upsample_bilinear2d_backward_out(
            result,
            self,
            iarg_vec(*args, 0, 2),
            iarg_vec(*args, 2, 4),
            args->iargs[6] != 0,
            optional_scale(*args, 7, 0),
            optional_scale(*args, 8, 1));
        break;
    case TorchKind::Softmax:
        at::_softmax_out(
            result,
            self,
            static_cast<std::int64_t>(args->iargs[0]),
            /*half_to_float=*/false);
        break;
    case TorchKind::LogSoftmax:
        at::_log_softmax_out(
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
    case TorchKind::Mean:
    {
        std::vector<std::int64_t> dims;
        const Index nd = args->iargs[0];
        for (Index i = 0; i < nd; ++i)
        {
            dims.push_back(
                static_cast<std::int64_t>(args->iargs[2 + i]));
        }
        const bool keepdim = args->iargs[1] != 0;
        if (dims.empty())
        {
            at::mean_out(result, self);
        }
        else
        {
            at::mean_out(result, self, dims, keepdim);
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
        const std::int64_t dim =
            static_cast<std::int64_t>(args->iargs[0]);
        const std::int64_t start =
            static_cast<std::int64_t>(args->iargs[1]);
        const std::int64_t length =
            static_cast<std::int64_t>(args->iargs[2]);
        // narrow_copy.out is CPU-only in stock ATen; view + copy_
        // works on CPU and CUDA (worker stream).
        at::Tensor src = self.narrow(dim, start, length);
        if (src.sizes() != result.sizes())
        {
            throw std::runtime_error(
                "torch NarrowCopy: size mismatch after narrow");
        }
        result.copy_(src);
        break;
    }
    case TorchKind::Copy:
    case TorchKind::CopyIntoView:
    {
        // Copy: densify a view into contiguous out.
        // CopyIntoView: packed out layout is a view of the parent
        // buffer (RW); copy_ writes only that region.
        if (self.sizes() != result.sizes())
        {
            throw std::runtime_error(
                "torch Copy: in/out size mismatch (packed layout "
                "meta must match logical tensor sizes)");
        }
        result.copy_(self);
        break;
    }
    case TorchKind::Repeat:
    {
        // Factors are stored for the *output* rank. Do not use
        // in_ndim: the tile may still be the parent 1D bias storage
        // while factors pad a leading dim (addmm / linear bias
        // broadcast).
        std::vector<std::int64_t> repeats;
        const Index nrep = args->out_ndim[0];
        for (Index i = 0; i < nrep; ++i)
        {
            repeats.push_back(
                static_cast<std::int64_t>(args->iargs[i]));
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
    case TorchKind::PowScalar:
        at::pow_out(
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
        // transpose_copy.out is not registered for CUDA; transpose
        // view + copy_ works on CPU and CUDA.
        at::Tensor src = self.transpose(d0, d1);
        if (src.sizes() != result.sizes())
        {
            throw std::runtime_error(
                "torch TransposeCopy: size mismatch after transpose");
        }
        result.copy_(src);
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
    float *out,
    at::Device device)
{
    at::Tensor ta = in_fp32(a, *args, 0, device);
    at::Tensor tb = in_fp32(b, *args, 1, device);
    at::Tensor result = out_fp32(out, *args, 0, device);
    switch (args->kind)
    {
    case TorchKind::Mul:
        at::mul_out(result, ta, tb);
        break;
    case TorchKind::Add:
        at::add_out(
            result,
            ta,
            tb,
            static_cast<double>(args->scalars[0]));
        break;
    case TorchKind::Sub:
        at::sub_out(
            result,
            ta,
            tb,
            static_cast<double>(args->scalars[0]));
        break;
    case TorchKind::Div:
        at::div_out(result, ta, tb);
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
    case TorchKind::LogSoftmaxBackward:
        at::_log_softmax_backward_data_out(
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
    float *out,
    at::Device device)
{
    at::Tensor ta = in_fp32(a, *args, 0, device);
    at::Tensor tb = in_fp32(b, *args, 1, device);
    at::Tensor tc = in_fp32(c, *args, 2, device);
    at::Tensor result = out_fp32(out, *args, 0, device);
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
        // Fused SDPA stays StarPU-only; F.sdpa on nntile uses the
        // MATH composite instead. Kept for a later fused
        // implementation; nothing records this kind today.
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

} // anonymous namespace

std::uint64_t torch_args_footprint(
    void const *cl_args, std::size_t cl_arg_size) noexcept
{
    auto const *args =
        static_cast<TorchDispatchArgs const *>(cl_args);
    if (cl_arg_size < sizeof(TorchDispatchArgs))
    {
        return 0;
    }
    // FNV-1a over the fields starpu_hash_crc32c covered: kind, n_in,
    // and the first input's sizes.
    std::uint64_t hash = 1469598103934665603ull;
    hash = fnv1a(&args->kind, sizeof(args->kind), hash);
    hash = fnv1a(&args->n_in, sizeof(args->n_in), hash);
    hash = fnv1a(
        args->in_sizes[0],
        sizeof(Index) * static_cast<size_t>(args->in_ndim[0]),
        hash);
    return hash;
}

void torch_insert(
    Codelet &codelet,
    int worker,
    TorchDispatchArgs const &meta,
    std::vector<BufSpec> const &bufs)
{
    // nnhaul::insert copies cl_args into the submit bag under the
    // submit lock, so the caller's meta may die before the task runs.
    insert_task(codelet.raw, worker, bufs, &meta, sizeof(meta));
}

template<typename T>
TorchUnary<std::tuple<T>>::TorchUnary():
    codelet(
        "nntile_torch_unary",
        &TorchUnary<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &TorchUnary<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &TorchUnary<std::tuple<T>>::footprint)
{
}

template<typename T>
void TorchUnary<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    try
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        if (static_cast<TorchDispatchArgs *>(cl_args)->kind ==
            TorchKind::Tril)
        {
            auto *args = reinterpret_cast<TorchDispatchArgs *>(cl_args);
            bool *in = buf_as<bool>(buffers, 0);
            bool *out = buf_as<bool>(buffers, 1);
            at::Tensor self = in_bool(in, *args, 0, at::kCPU);
            at::Tensor result = out_bool(out, *args, 0, at::kCPU);
            at::tril_out(
                result,
                self,
                static_cast<std::int64_t>(args->iargs[0]));
            return;
        }
        auto *args = reinterpret_cast<TorchDispatchArgs *>(cl_args);
        float *in = buf_as<float>(buffers, 0);
        float *out = copy_into_view_aliases_in(args)
            ? in
            : buf_as<float>(buffers, 1);
        run_unary(args, in, out, at::kCPU);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(stderr, "nntile_torch_unary failed: %s\n", ex.what());
        std::abort();
    }
}

#ifdef NNTILE_USE_CUDA
template<typename T>
void TorchUnary<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    try
    {
        HaulTorchCudaEnv cuda_env;
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        auto *args = reinterpret_cast<TorchDispatchArgs *>(cl_args);
        if (args->kind == TorchKind::Tril)
        {
            bool *in = buf_as<bool>(buffers, 0);
            bool *out = buf_as<bool>(buffers, 1);
            at::Tensor self = in_bool(in, *args, 0, cuda_env.device());
            at::Tensor result =
                out_bool(out, *args, 0, cuda_env.device());
            at::tril_out(
                result,
                self,
                static_cast<std::int64_t>(args->iargs[0]));
            return;
        }
        float *in = buf_as<float>(buffers, 0);
        float *out = copy_into_view_aliases_in(args)
            ? in
            : buf_as<float>(buffers, 1);
        run_unary(args, in, out, cuda_env.device());
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_unary CUDA failed: %s\n",
            ex.what());
        std::abort();
    }
}
#endif // NNTILE_USE_CUDA

template<typename T>
void TorchUnary<std::tuple<T>>::submit(
    int worker_hint,
    args_t const &meta,
    TorchHandle const &in,
    TorchHandle const &out)
{
    args_t args = meta;
    if (args.kind == TorchKind::CopyIntoView)
    {
        // Preserve parent values outside the packed view (RW, not W).
        const bool out_aliases_in =
            (&out.get() == &in.get());
        args.iargs[7] = out_aliases_in ? 1 : 0;
        torch_insert(
            codelet,
            worker_hint,
            args,
            out_aliases_in
                ? std::vector<BufSpec>{
                    BufSpec{STARPU_RW, &in.get()}}
                : std::vector<BufSpec>{
                    BufSpec{STARPU_R, &in.get()},
                    BufSpec{STARPU_RW, &out.get()}});
        return;
    }
    torch_insert(
        codelet,
        worker_hint,
        args,
        {BufSpec{STARPU_R, &in.get()}, BufSpec{STARPU_W, &out.get()}});
}

template<typename T>
TorchBinary<std::tuple<T>>::TorchBinary():
    codelet(
        "nntile_torch_binary",
        &TorchBinary<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &TorchBinary<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &TorchBinary<std::tuple<T>>::footprint)
{
}

template<typename T>
void TorchBinary<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    try
    {
        auto *args = reinterpret_cast<TorchDispatchArgs *>(cl_args);
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        run_binary(
            args,
            buf_as<float>(buffers, 0),
            buf_as<float>(buffers, 1),
            buf_as<float>(buffers, 2),
            at::kCPU);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(stderr, "nntile_torch_binary failed: %s\n", ex.what());
        std::abort();
    }
}

#ifdef NNTILE_USE_CUDA
template<typename T>
void TorchBinary<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    try
    {
        HaulTorchCudaEnv cuda_env;
        auto *args = reinterpret_cast<TorchDispatchArgs *>(cl_args);
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        run_binary(
            args,
            buf_as<float>(buffers, 0),
            buf_as<float>(buffers, 1),
            buf_as<float>(buffers, 2),
            cuda_env.device());
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_binary CUDA failed: %s\n",
            ex.what());
        std::abort();
    }
}
#endif // NNTILE_USE_CUDA

template<typename T>
void TorchBinary<std::tuple<T>>::submit(
    int worker_hint,
    args_t const &meta,
    TorchHandle const &a,
    TorchHandle const &b,
    TorchHandle const &out)
{
    torch_insert(
        codelet,
        worker_hint,
        meta,
        {BufSpec{STARPU_R, &a.get()},
            BufSpec{STARPU_R, &b.get()},
            BufSpec{STARPU_W, &out.get()}});
}

template<typename T>
TorchTernary<std::tuple<T>>::TorchTernary():
    codelet(
        "nntile_torch_ternary",
        &TorchTernary<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &TorchTernary<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &TorchTernary<std::tuple<T>>::footprint)
{
}

template<typename T>
void TorchTernary<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    try
    {
        auto *args = reinterpret_cast<TorchDispatchArgs *>(cl_args);
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        run_ternary(
            args,
            buf_as<float>(buffers, 0),
            buf_as<float>(buffers, 1),
            buf_as<float>(buffers, 2),
            buf_as<float>(buffers, 3),
            at::kCPU);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(stderr, "nntile_torch_ternary failed: %s\n", ex.what());
        std::abort();
    }
}

#ifdef NNTILE_USE_CUDA
template<typename T>
void TorchTernary<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    try
    {
        HaulTorchCudaEnv cuda_env;
        auto *args = reinterpret_cast<TorchDispatchArgs *>(cl_args);
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        run_ternary(
            args,
            buf_as<float>(buffers, 0),
            buf_as<float>(buffers, 1),
            buf_as<float>(buffers, 2),
            buf_as<float>(buffers, 3),
            cuda_env.device());
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_ternary CUDA failed: %s\n",
            ex.what());
        std::abort();
    }
}
#endif // NNTILE_USE_CUDA

template<typename T>
void TorchTernary<std::tuple<T>>::submit(
    int worker_hint,
    args_t const &meta,
    TorchHandle const &a,
    TorchHandle const &b,
    TorchHandle const &c,
    TorchHandle const &out)
{
    torch_insert(
        codelet,
        worker_hint,
        meta,
        {BufSpec{STARPU_R, &a.get()},
            BufSpec{STARPU_R, &b.get()},
            BufSpec{STARPU_R, &c.get()},
            BufSpec{STARPU_W, &out.get()}});
}

// Explicit instantiations for the fp32 family packs.

//! Embedding: weight R, indices (i64) R, out W.
TorchEmbedding::TorchEmbedding():
    codelet(
        "nntile_torch_embedding",
        &TorchEmbedding::cpu,
#ifdef NNTILE_USE_CUDA
        &TorchEmbedding::cuda,
#else
        nullptr,
#endif
        &TorchEmbedding::footprint)
{
}

void TorchEmbedding::cpu(void *buffers[], void *cl_args) noexcept
{
    try
    {
        auto *args = reinterpret_cast<TorchDispatchArgs *>(cl_args);
        at::Tensor w = in_fp32(buf_as<float>(buffers, 0), *args, 0);
        at::Tensor idx =
            in_i64(buf_as<std::int64_t>(buffers, 1), *args, 1);
        at::Tensor result = out_fp32(buf_as<float>(buffers, 2), *args, 0);
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
        std::fprintf(stderr, "nntile_torch_embedding failed: %s\n", ex.what());
        std::abort();
    }
}

#ifdef NNTILE_USE_CUDA
void TorchEmbedding::cuda(void *buffers[], void *cl_args) noexcept
{
    try
    {
        HaulTorchCudaEnv cuda_env;
        (void)cuda_env;
        cpu(buffers, cl_args);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_embedding CUDA failed: %s\n",
            ex.what());
        std::abort();
    }
}
#endif // NNTILE_USE_CUDA

void TorchEmbedding::submit(
    int worker_hint,
    args_t const &meta,
    TorchHandle const &weight,
    TorchHandle const &indices,
    TorchHandle const &out)
{
    torch_insert(
        codelet,
        worker_hint,
        meta,
        {BufSpec{STARPU_R, &weight.get()},
            BufSpec{STARPU_R, &indices.get()},
            BufSpec{STARPU_W, &out.get()}});
}

//! Embedding dense backward: grad R, indices R, grad_weight W.
TorchEmbeddingDenseBackward::TorchEmbeddingDenseBackward():
    codelet(
        "nntile_torch_embedding_dense_backward",
        &TorchEmbeddingDenseBackward::cpu,
#ifdef NNTILE_USE_CUDA
        &TorchEmbeddingDenseBackward::cuda,
#else
        nullptr,
#endif
        &TorchEmbeddingDenseBackward::footprint)
{
}

void TorchEmbeddingDenseBackward::cpu(
    void *buffers[],
    void *cl_args) noexcept
{
    try
    {
        auto *args = reinterpret_cast<TorchDispatchArgs *>(cl_args);
        at::Tensor grad = in_fp32(buf_as<float>(buffers, 0), *args, 0);
        at::Tensor indices =
            in_i64(buf_as<std::int64_t>(buffers, 1), *args, 1);
        at::Tensor grad_weight =
            out_fp32(buf_as<float>(buffers, 2), *args, 0);
        const std::int64_t num_weights =
            static_cast<std::int64_t>(args->iargs[0]);
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        at::embedding_dense_backward_out(
            grad_weight,
            grad,
            indices,
            num_weights,
            /*padding_idx=*/static_cast<std::int64_t>(args->iargs[1]),
            /*scale_grad_by_freq=*/false);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_embedding_dense_backward failed: %s\n",
            ex.what());
        std::abort();
    }
}

#ifdef NNTILE_USE_CUDA
void TorchEmbeddingDenseBackward::cuda(
    void *buffers[],
    void *cl_args) noexcept
{
    try
    {
        HaulTorchCudaEnv cuda_env;
        (void)cuda_env;
        cpu(buffers, cl_args);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_embedding_dense_backward CUDA failed: %s\n",
            ex.what());
        std::abort();
    }
}
#endif // NNTILE_USE_CUDA

void TorchEmbeddingDenseBackward::submit(
    int worker_hint,
    args_t const &meta,
    TorchHandle const &grad,
    TorchHandle const &indices,
    TorchHandle const &grad_weight)
{
    torch_insert(
        codelet,
        worker_hint,
        meta,
        {BufSpec{STARPU_R, &grad.get()},
            BufSpec{STARPU_R, &indices.get()},
            BufSpec{STARPU_W, &grad_weight.get()}});
}

//! NLL loss forward: log_probs R, target(i64) R, loss W, tw W.
TorchNllLossForward::TorchNllLossForward():
    codelet(
        "nntile_torch_nll_loss_forward",
        &TorchNllLossForward::cpu,
#ifdef NNTILE_USE_CUDA
        &TorchNllLossForward::cuda,
#else
        nullptr,
#endif
        &TorchNllLossForward::footprint)
{
}

void TorchNllLossForward::cpu(void *buffers[], void *cl_args) noexcept
{
    try
    {
        auto *args = reinterpret_cast<TorchDispatchArgs *>(cl_args);
        at::Tensor log_probs =
            in_fp32(buf_as<float>(buffers, 0), *args, 0);
        at::Tensor target =
            in_i64(buf_as<std::int64_t>(buffers, 1), *args, 1);
        at::Tensor loss = out_fp32(buf_as<float>(buffers, 2), *args, 0);
        at::Tensor total_weight =
            out_fp32(buf_as<float>(buffers, 3), *args, 1);
        const std::int64_t reduction =
            static_cast<std::int64_t>(args->iargs[0]);
        const std::int64_t ignore_index =
            static_cast<std::int64_t>(args->iargs[1]);
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        at::nll_loss_forward_out(
            loss,
            total_weight,
            log_probs,
            target,
            /*weight=*/c10::nullopt,
            reduction,
            ignore_index);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_nll_loss_forward failed: %s\n",
            ex.what());
        std::abort();
    }
}

#ifdef NNTILE_USE_CUDA
void TorchNllLossForward::cuda(void *buffers[], void *cl_args) noexcept
{
    try
    {
        HaulTorchCudaEnv cuda_env;
        (void)cuda_env;
        cpu(buffers, cl_args);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_nll_loss_forward CUDA failed: %s\n",
            ex.what());
        std::abort();
    }
}
#endif // NNTILE_USE_CUDA

void TorchNllLossForward::submit(
    int worker_hint,
    args_t const &meta,
    TorchHandle const &log_probs,
    TorchHandle const &target,
    TorchHandle const &loss,
    TorchHandle const &total_weight)
{
    torch_insert(
        codelet,
        worker_hint,
        meta,
        {BufSpec{STARPU_R, &log_probs.get()},
            BufSpec{STARPU_R, &target.get()},
            BufSpec{STARPU_W, &loss.get()},
            BufSpec{STARPU_W, &total_weight.get()}});
}

//! NLL loss backward: grad_loss R, log_probs R, target R, tw R, gi W.
TorchNllLossBackward::TorchNllLossBackward():
    codelet(
        "nntile_torch_nll_loss_backward",
        &TorchNllLossBackward::cpu,
#ifdef NNTILE_USE_CUDA
        &TorchNllLossBackward::cuda,
#else
        nullptr,
#endif
        &TorchNllLossBackward::footprint)
{
}

void TorchNllLossBackward::cpu(void *buffers[], void *cl_args) noexcept
{
    try
    {
        auto *args = reinterpret_cast<TorchDispatchArgs *>(cl_args);
        at::Tensor grad_output =
            in_fp32(buf_as<float>(buffers, 0), *args, 0);
        at::Tensor log_probs =
            in_fp32(buf_as<float>(buffers, 1), *args, 1);
        at::Tensor target =
            in_i64(buf_as<std::int64_t>(buffers, 2), *args, 2);
        at::Tensor total_weight =
            in_fp32(buf_as<float>(buffers, 3), *args, 3);
        at::Tensor grad_input =
            out_fp32(buf_as<float>(buffers, 4), *args, 0);
        const std::int64_t reduction =
            static_cast<std::int64_t>(args->iargs[0]);
        const std::int64_t ignore_index =
            static_cast<std::int64_t>(args->iargs[1]);
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        at::nll_loss_backward_out(
            grad_input,
            grad_output,
            log_probs,
            target,
            /*weight=*/c10::nullopt,
            reduction,
            ignore_index,
            total_weight);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_nll_loss_backward failed: %s\n",
            ex.what());
        std::abort();
    }
}

#ifdef NNTILE_USE_CUDA
void TorchNllLossBackward::cuda(void *buffers[], void *cl_args) noexcept
{
    try
    {
        HaulTorchCudaEnv cuda_env;
        (void)cuda_env;
        cpu(buffers, cl_args);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_nll_loss_backward CUDA failed: %s\n",
            ex.what());
        std::abort();
    }
}
#endif // NNTILE_USE_CUDA

void TorchNllLossBackward::submit(
    int worker_hint,
    args_t const &meta,
    TorchHandle const &grad_output,
    TorchHandle const &log_probs,
    TorchHandle const &target,
    TorchHandle const &total_weight,
    TorchHandle const &grad_input)
{
    torch_insert(
        codelet,
        worker_hint,
        meta,
        {BufSpec{STARPU_R, &grad_output.get()},
            BufSpec{STARPU_R, &log_probs.get()},
            BufSpec{STARPU_R, &target.get()},
            BufSpec{STARPU_R, &total_weight.get()},
            BufSpec{STARPU_W, &grad_input.get()}});
}

//! Variable-arity cat: up to max_tensors fp32 inputs, one W out.
TorchCat::TorchCat():
    codelet(
        "nntile_torch_cat",
        &TorchCat::cpu,
#ifdef NNTILE_USE_CUDA
        &TorchCat::cuda,
#else
        nullptr,
#endif
        &TorchCat::footprint)
{
}

void TorchCat::cpu(void *buffers[], void *cl_args) noexcept
{
    try
    {
        auto *args = reinterpret_cast<TorchDispatchArgs *>(cl_args);
        const Index n = args->iargs[1];
        std::vector<at::Tensor> inputs;
        inputs.reserve(static_cast<size_t>(n));
        for (Index i = 0; i < n; ++i)
        {
            inputs.push_back(
                in_fp32(buf_as<float>(buffers, i), *args, i));
        }
        at::Tensor result =
            out_fp32(buf_as<float>(buffers, n), *args, 0);
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        at::cat_out(
            result,
            inputs,
            static_cast<std::int64_t>(args->iargs[0]));
    }
    catch (const std::exception &ex)
    {
        std::fprintf(stderr, "nntile_torch_cat failed: %s\n", ex.what());
        std::abort();
    }
}

#ifdef NNTILE_USE_CUDA
void TorchCat::cuda(void *buffers[], void *cl_args) noexcept
{
    try
    {
        HaulTorchCudaEnv cuda_env;
        (void)cuda_env;
        cpu(buffers, cl_args);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(stderr, "nntile_torch_cat CUDA failed: %s\n", ex.what());
        std::abort();
    }
}
#endif // NNTILE_USE_CUDA

void TorchCat::submit(
    int worker_hint,
    args_t const &meta,
    std::vector<TorchHandle> const &inputs,
    TorchHandle const &out)
{
    std::vector<BufSpec> bufs;
    bufs.reserve(inputs.size() + 1);
    for (TorchHandle const &h : inputs)
    {
        bufs.push_back(BufSpec{STARPU_R, &h.get()});
    }
    bufs.push_back(BufSpec{STARPU_W, &out.get()});
    torch_insert(codelet, worker_hint, meta, bufs);
}

//! Where: cond(bool) R, x R, y R, out W; i64 or fp32 values.
TorchWhere::TorchWhere():
    codelet(
        "nntile_torch_where",
        &TorchWhere::cpu,
#ifdef NNTILE_USE_CUDA
        &TorchWhere::cuda,
#else
        nullptr,
#endif
        &TorchWhere::footprint)
{
}

void TorchWhere::cpu(void *buffers[], void *cl_args) noexcept
{
    try
    {
        auto *args = reinterpret_cast<TorchDispatchArgs *>(cl_args);
        at::Tensor cond = in_bool(buf_as<bool>(buffers, 0), *args, 0);
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        if (args->iargs[15] != 0)
        {
            at::Tensor self =
                in_i64(buf_as<std::int64_t>(buffers, 1), *args, 1);
            at::Tensor other =
                in_i64(buf_as<std::int64_t>(buffers, 2), *args, 2);
            at::Tensor result =
                out_i64(buf_as<std::int64_t>(buffers, 3), *args, 0);
            at::where_out(result, cond, self, other);
        }
        else
        {
            at::Tensor self =
                in_fp32(buf_as<float>(buffers, 1), *args, 1);
            at::Tensor other =
                in_fp32(buf_as<float>(buffers, 2), *args, 2);
            at::Tensor result =
                out_fp32(buf_as<float>(buffers, 3), *args, 0);
            at::where_out(result, cond, self, other);
        }
    }
    catch (const std::exception &ex)
    {
        std::fprintf(stderr, "nntile_torch_where failed: %s\n", ex.what());
        std::abort();
    }
}

#ifdef NNTILE_USE_CUDA
void TorchWhere::cuda(void *buffers[], void *cl_args) noexcept
{
    try
    {
        HaulTorchCudaEnv cuda_env;
        (void)cuda_env;
        cpu(buffers, cl_args);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_where CUDA failed: %s\n",
            ex.what());
        std::abort();
    }
}
#endif // NNTILE_USE_CUDA

void TorchWhere::submit(
    int worker_hint,
    args_t const &meta,
    TorchHandle const &condition,
    TorchHandle const &self,
    TorchHandle const &other,
    TorchHandle const &out)
{
    torch_insert(
        codelet,
        worker_hint,
        meta,
        {BufSpec{STARPU_R, &condition.get()},
            BufSpec{STARPU_R, &self.get()},
            BufSpec{STARPU_R, &other.get()},
            BufSpec{STARPU_W, &out.get()}});
}

//! Write-only fills: arange (i64), arange (fp32), fill (i64/bool).
TorchArange::TorchArange():
    codelet(
        "nntile_torch_arange",
        &TorchArange::cpu,
#ifdef NNTILE_USE_CUDA
        &TorchArange::cuda,
#else
        nullptr,
#endif
        &TorchArange::footprint)
{
}

void TorchArange::cpu(void *buffers[], void *cl_args) noexcept
{
    try
    {
        auto *args = reinterpret_cast<TorchDispatchArgs *>(cl_args);
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        if (args->kind == TorchKind::ArangeFp32)
        {
            at::Tensor result =
                out_fp32(buf_as<float>(buffers, 0), *args, 0);
            at::arange_out(
                result,
                at::Scalar(
                    static_cast<double>(args->scalars[0])),
                at::Scalar(
                    static_cast<double>(args->scalars[1])),
                at::Scalar(
                    static_cast<double>(args->scalars[2])));
        }
        else if (args->kind == TorchKind::FillI64)
        {
            at::Tensor result =
                out_i64(buf_as<std::int64_t>(buffers, 0), *args, 0);
            result.fill_(
                static_cast<std::int64_t>(args->iargs[0]));
        }
        else if (args->kind == TorchKind::FillBool)
        {
            at::Tensor result =
                out_bool(buf_as<bool>(buffers, 0), *args, 0);
            result.fill_(args->iargs[0] != 0);
        }
        else
        {
            at::Tensor result =
                out_i64(buf_as<std::int64_t>(buffers, 0), *args, 0);
            at::arange_out(
                result,
                at::Scalar(
                    static_cast<std::int64_t>(args->iargs[0])),
                at::Scalar(
                    static_cast<std::int64_t>(args->iargs[1])),
                at::Scalar(
                    static_cast<std::int64_t>(args->iargs[2])));
        }
    }
    catch (const std::exception &ex)
    {
        std::fprintf(stderr, "nntile_torch_arange failed: %s\n", ex.what());
        std::abort();
    }
}

#ifdef NNTILE_USE_CUDA
void TorchArange::cuda(void *buffers[], void *cl_args) noexcept
{
    try
    {
        HaulTorchCudaEnv cuda_env;
        (void)cuda_env;
        cpu(buffers, cl_args);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_arange CUDA failed: %s\n",
            ex.what());
        std::abort();
    }
}
#endif // NNTILE_USE_CUDA

void TorchArange::submit(
    int worker_hint,
    args_t const &meta,
    TorchHandle const &out)
{
    torch_insert(
        codelet,
        worker_hint,
        meta,
        {BufSpec{STARPU_W, &out.get()}});
}

//! int64 elementwise: gt/lt → bool; add/sub/mul/minimum → i64;
//! fp32*bool mul special cases (iargs[15] = 2 bool, 3 fp32*bool).
TorchGt::TorchGt():
    codelet(
        "nntile_torch_gt",
        &TorchGt::cpu,
#ifdef NNTILE_USE_CUDA
        &TorchGt::cuda,
#else
        nullptr,
#endif
        &TorchGt::footprint)
{
}

void TorchGt::cpu(void *buffers[], void *cl_args) noexcept
{
    try
    {
        auto *args = reinterpret_cast<TorchDispatchArgs *>(cl_args);
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        if (args->kind == TorchKind::Mul && args->iargs[15] == 3)
        {
            at::Tensor ta = in_fp32(buf_as<float>(buffers, 0), *args, 0);
            at::Tensor tb =
                in_bool(buf_as<bool>(buffers, 1), *args, 1);
            at::Tensor result =
                out_fp32(buf_as<float>(buffers, 2), *args, 0);
            at::mul_out(result, ta, tb);
            return;
        }
        if (args->kind == TorchKind::Mul && args->iargs[15] == 2)
        {
            at::Tensor ta = in_bool(buf_as<bool>(buffers, 0), *args, 0);
            at::Tensor tb = in_bool(buf_as<bool>(buffers, 1), *args, 1);
            at::Tensor result =
                out_bool(buf_as<bool>(buffers, 2), *args, 0);
            at::mul_out(result, ta, tb);
            return;
        }
        if (args->kind == TorchKind::Eq)
        {
            at::Tensor ta = in_fp32(buf_as<float>(buffers, 0), *args, 0);
            at::Tensor tb = in_fp32(buf_as<float>(buffers, 1), *args, 1);
            at::Tensor result =
                out_bool(buf_as<bool>(buffers, 2), *args, 0);
            at::eq_out(result, ta, tb);
        }
        else
        {
            at::Tensor ta =
                in_i64(buf_as<std::int64_t>(buffers, 0), *args, 0);
            at::Tensor tb =
                in_i64(buf_as<std::int64_t>(buffers, 1), *args, 1);
            switch (args->kind)
            {
            case TorchKind::Lt:
            {
                at::Tensor result =
                    out_bool(buf_as<bool>(buffers, 2), *args, 0);
                at::lt_out(result, ta, tb);
                break;
            }
            case TorchKind::Sub:
            {
                at::Tensor result =
                    out_i64(buf_as<std::int64_t>(buffers, 2), *args, 0);
                at::sub_out(result, ta, tb, /*alpha=*/1);
                break;
            }
            case TorchKind::Add:
            {
                at::Tensor result =
                    out_i64(buf_as<std::int64_t>(buffers, 2), *args, 0);
                at::add_out(result, ta, tb, /*alpha=*/1);
                break;
            }
            case TorchKind::Mul:
            {
                at::Tensor result =
                    out_i64(buf_as<std::int64_t>(buffers, 2), *args, 0);
                at::mul_out(result, ta, tb);
                break;
            }
            case TorchKind::Minimum:
            {
                at::Tensor result =
                    out_i64(buf_as<std::int64_t>(buffers, 2), *args, 0);
                at::minimum_out(result, ta, tb);
                break;
            }
            default:
            {
                at::Tensor result =
                    out_bool(buf_as<bool>(buffers, 2), *args, 0);
                at::gt_out(result, ta, tb);
                break;
            }
            }
        }
    }
    catch (const std::exception &ex)
    {
        std::fprintf(stderr, "nntile_torch_gt failed: %s\n", ex.what());
        std::abort();
    }
}

#ifdef NNTILE_USE_CUDA
void TorchGt::cuda(void *buffers[], void *cl_args) noexcept
{
    try
    {
        HaulTorchCudaEnv cuda_env;
        (void)cuda_env;
        cpu(buffers, cl_args);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(stderr, "nntile_torch_gt CUDA failed: %s\n", ex.what());
        std::abort();
    }
}
#endif // NNTILE_USE_CUDA

void TorchGt::submit(
    int worker_hint,
    args_t const &meta,
    TorchHandle const &a,
    TorchHandle const &b,
    TorchHandle const &out)
{
    torch_insert(
        codelet,
        worker_hint,
        meta,
        {BufSpec{STARPU_R, &a.get()},
            BufSpec{STARPU_R, &b.get()},
            BufSpec{STARPU_W, &out.get()}});
}

//! int64 unary: abs / neg / copy (i64 registers and index math).
TorchI64Unary::TorchI64Unary():
    codelet(
        "nntile_torch_i64_unary",
        &TorchI64Unary::cpu,
#ifdef NNTILE_USE_CUDA
        &TorchI64Unary::cuda,
#else
        nullptr,
#endif
        &TorchI64Unary::footprint)
{
}

void TorchI64Unary::cpu(void *buffers[], void *cl_args) noexcept
{
    try
    {
        auto *args = reinterpret_cast<TorchDispatchArgs *>(cl_args);
        std::int64_t *in_ptr = buf_as<std::int64_t>(buffers, 0);
        std::int64_t *out_ptr = copy_into_view_aliases_in(args)
            ? in_ptr
            : buf_as<std::int64_t>(buffers, 1);
        at::Tensor self = in_i64(in_ptr, *args, 0);
        at::Tensor result = out_i64(out_ptr, *args, 0);
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        switch (args->kind)
        {
        case TorchKind::Abs:
            at::abs_out(result, self);
            break;
        case TorchKind::Neg:
            at::neg_out(result, self);
            break;
        case TorchKind::Copy:
        case TorchKind::CopyIntoView:
            result.copy_(self);
            break;
        default:
            throw std::runtime_error(
                "torch_i64_unary: unsupported kind");
        }
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_i64_unary failed: %s\n",
            ex.what());
        std::abort();
    }
}

#ifdef NNTILE_USE_CUDA
void TorchI64Unary::cuda(void *buffers[], void *cl_args) noexcept
{
    try
    {
        HaulTorchCudaEnv cuda_env;
        (void)cuda_env;
        cpu(buffers, cl_args);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_i64_unary CUDA failed: %s\n",
            ex.what());
        std::abort();
    }
}
#endif // NNTILE_USE_CUDA

void TorchI64Unary::submit(
    int worker_hint,
    args_t const &meta,
    TorchHandle const &in,
    TorchHandle const &out)
{
    torch_insert(
        codelet,
        worker_hint,
        meta,
        {BufSpec{STARPU_R, &in.get()}, BufSpec{STARPU_W, &out.get()}});
}

//! dtype cast between fp32 / i64 / bool (tags in iargs[0]/iargs[1]).
TorchCast::TorchCast():
    codelet(
        "nntile_torch_cast",
        &TorchCast::cpu,
#ifdef NNTILE_USE_CUDA
        &TorchCast::cuda,
#else
        nullptr,
#endif
        &TorchCast::footprint)
{
}

void TorchCast::cpu(void *buffers[], void *cl_args) noexcept
{
    try
    {
        auto *args = reinterpret_cast<TorchDispatchArgs *>(cl_args);
        at::Tensor self = in_tagged(buffers, *args, 0, args->iargs[0]);
        at::Tensor result =
            out_tagged(buffers, *args, 0, args->iargs[1]);
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        result.copy_(self);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(stderr, "nntile_torch_cast failed: %s\n", ex.what());
        std::abort();
    }
}

#ifdef NNTILE_USE_CUDA
void TorchCast::cuda(void *buffers[], void *cl_args) noexcept
{
    try
    {
        HaulTorchCudaEnv cuda_env;
        (void)cuda_env;
        cpu(buffers, cl_args);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(stderr, "nntile_torch_cast CUDA failed: %s\n", ex.what());
        std::abort();
    }
}
#endif // NNTILE_USE_CUDA

void TorchCast::submit(
    int worker_hint,
    args_t const &meta,
    TorchHandle const &in,
    TorchHandle const &out)
{
    torch_insert(
        codelet,
        worker_hint,
        meta,
        {BufSpec{STARPU_R, &in.get()}, BufSpec{STARPU_W, &out.get()}});
}

template class TorchUnary<std::tuple<fp32_t>>;
template class TorchBinary<std::tuple<fp32_t>>;
template class TorchTernary<std::tuple<fp32_t>>;

torch_unary_pack_t torch_unary;
torch_binary_pack_t torch_binary;
torch_ternary_pack_t torch_ternary;

//! aten::convolution via the public dispatcher entry: the simple port
//! calls at::convolution_out and lets aten select the backend (cuDNN on
//! CUDA), accepting extra copies a hand-tuned backend switch (the
//! StarPU path) would avoid. 1-D configs go through aten's own folding.
TorchConvolution::TorchConvolution():
    codelet(
        "nntile_torch_convolution",
        &TorchConvolution::cpu,
#ifdef NNTILE_USE_CUDA
        &TorchConvolution::cuda,
#else
        nullptr,
#endif
        &TorchConvolution::footprint)
{
}

void TorchConvolution::cpu(void *buffers[], void *cl_args) noexcept
{
    try
    {
        auto *args = reinterpret_cast<TorchDispatchArgs *>(cl_args);
        const bool has_bias = args->iargs[11] != 0;
        Index buf = 0;
        at::Tensor input = in_fp32(buf_as<float>(buffers, buf++), *args, 0);
        at::Tensor weight = in_fp32(buf_as<float>(buffers, buf++), *args, 1);
        at::Tensor bias;
        if (has_bias)
        {
            bias = in_fp32(buf_as<float>(buffers, buf++), *args, 2);
        }
        at::Tensor out = out_fp32(buf_as<float>(buffers, buf++), *args, 0);
        const Index ndim = args->iargs[0];
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        at::convolution_out(
            out,
            input,
            weight,
            has_bias ? std::optional<at::Tensor>(bias)
                     : std::nullopt,
            iarg_vec(*args, 3, ndim),
            iarg_vec(*args, 5, ndim),
            iarg_vec(*args, 7, ndim),
            args->iargs[2] != 0,
            iarg_vec(*args, 9, ndim),
            static_cast<std::int64_t>(args->iargs[1]));
    }
    catch (const std::exception &ex)
    {
        std::fprintf(stderr, "nntile_torch_convolution failed: %s\n", ex.what());
        std::abort();
    }
}

#ifdef NNTILE_USE_CUDA
void TorchConvolution::cuda(void *buffers[], void *cl_args) noexcept
{
    try
    {
        HaulTorchCudaEnv cuda_env;
        (void)cuda_env;
        cpu(buffers, cl_args);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_convolution CUDA failed: %s\n",
            ex.what());
        std::abort();
    }
}
#endif // NNTILE_USE_CUDA

void TorchConvolution::submit(
    int worker_hint,
    args_t const &meta,
    TorchHandle const &input,
    TorchHandle const &weight,
    TorchHandle const &bias,
    TorchHandle const &out,
    bool has_bias)
{
    args_t args = meta;
    args.kind = TorchKind::Convolution;
    args.iargs[11] = has_bias ? 1 : 0;
    std::vector<BufSpec> bufs;
    bufs.push_back(BufSpec{STARPU_R, &input.get()});
    bufs.push_back(BufSpec{STARPU_R, &weight.get()});
    if (has_bias)
    {
        bufs.push_back(BufSpec{STARPU_R, &bias.get()});
    }
    bufs.push_back(BufSpec{STARPU_W, &out.get()});
    torch_insert(codelet, worker_hint, args, bufs);
}

//! aten::convolution_backward via the public dispatcher entry
//! (out variant: writes the caller-provided grad buffers).
TorchConvolutionBackward::TorchConvolutionBackward():
    codelet(
        "nntile_torch_convolution_backward",
        &TorchConvolutionBackward::cpu,
#ifdef NNTILE_USE_CUDA
        &TorchConvolutionBackward::cuda,
#else
        nullptr,
#endif
        &TorchConvolutionBackward::footprint)
{
}

void TorchConvolutionBackward::cpu(void *buffers[], void *cl_args) noexcept
{
    try
    {
        auto *args = reinterpret_cast<TorchDispatchArgs *>(cl_args);
        const bool need_gi = args->iargs[12] != 0;
        const bool need_gw = args->iargs[13] != 0;
        const bool need_gb = args->iargs[14] != 0;
        Index buf = 0;
        at::Tensor grad_out =
            in_fp32(buf_as<float>(buffers, buf++), *args, 0);
        at::Tensor input = in_fp32(buf_as<float>(buffers, buf++), *args, 1);
        at::Tensor weight = in_fp32(buf_as<float>(buffers, buf++), *args, 2);
        at::Tensor grad_input;
        at::Tensor grad_weight;
        at::Tensor grad_bias;
        if (need_gi)
        {
            grad_input = out_fp32(buf_as<float>(buffers, buf++), *args, 0);
        }
        if (need_gw)
        {
            grad_weight = out_fp32(buf_as<float>(buffers, buf++), *args, 1);
        }
        if (need_gb)
        {
            grad_bias = out_fp32(buf_as<float>(buffers, buf++), *args, 2);
        }
        // The dispatcher's out wrapper requires defined tensors for every
        // output even when the output mask skips it.
        if (!need_gi)
        {
            grad_input = at::empty({0}, input.options());
        }
        if (!need_gw)
        {
            grad_weight = at::empty({0}, weight.options());
        }
        if (!need_gb)
        {
            grad_bias = at::empty({0}, input.options());
        }
        const Index ndim = args->iargs[0];
        std::vector<std::int64_t> bias_sizes_vec;
        at::OptionalIntArrayRef bias_sizes = c10::nullopt;
        if (need_gb)
        {
            bias_sizes_vec = sizes_of(*args, 2, true);
            bias_sizes = at::IntArrayRef(bias_sizes_vec);
        }
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        // The functional entry allocates its own outputs (the _out
        // wrapper demands uninitialized buffers), so the results are
        // copied into the caller's tiles - the accepted extra copying
        // of the simple port.
        auto grads = at::convolution_backward(
            grad_out,
            input,
            weight,
            bias_sizes,
            iarg_vec(*args, 3, ndim),
            iarg_vec(*args, 5, ndim),
            iarg_vec(*args, 7, ndim),
            args->iargs[2] != 0,
            iarg_vec(*args, 9, ndim),
            static_cast<std::int64_t>(args->iargs[1]),
            {need_gi, need_gw, need_gb});
        if (need_gi)
        {
            grad_input.copy_(std::get<0>(grads));
        }
        if (need_gw)
        {
            grad_weight.copy_(std::get<1>(grads));
        }
        if (need_gb)
        {
            grad_bias.copy_(std::get<2>(grads));
        }
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_convolution_backward failed: %s\n",
            ex.what());
        std::abort();
    }
}

#ifdef NNTILE_USE_CUDA
void TorchConvolutionBackward::cuda(void *buffers[], void *cl_args) noexcept
{
    try
    {
        HaulTorchCudaEnv cuda_env;
        (void)cuda_env;
        cpu(buffers, cl_args);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_convolution_backward CUDA failed: %s\n",
            ex.what());
        std::abort();
    }
}
#endif // NNTILE_USE_CUDA

void TorchConvolutionBackward::submit(
    int worker_hint,
    args_t const &meta,
    TorchHandle const &grad_out,
    TorchHandle const &input,
    TorchHandle const &weight,
    TorchHandle const &grad_input,
    TorchHandle const &grad_weight,
    TorchHandle const &grad_bias,
    bool need_grad_input,
    bool need_grad_weight,
    bool need_grad_bias)
{
    args_t args = meta;
    args.kind = TorchKind::ConvolutionBackward;
    args.iargs[12] = need_grad_input ? 1 : 0;
    args.iargs[13] = need_grad_weight ? 1 : 0;
    args.iargs[14] = need_grad_bias ? 1 : 0;
    std::vector<BufSpec> bufs;
    bufs.push_back(BufSpec{STARPU_R, &grad_out.get()});
    bufs.push_back(BufSpec{STARPU_R, &input.get()});
    bufs.push_back(BufSpec{STARPU_R, &weight.get()});
    if (need_grad_input)
    {
        bufs.push_back(BufSpec{STARPU_W, &grad_input.get()});
    }
    if (need_grad_weight)
    {
        bufs.push_back(BufSpec{STARPU_W, &grad_weight.get()});
    }
    if (need_grad_bias)
    {
        bufs.push_back(BufSpec{STARPU_W, &grad_bias.get()});
    }
    torch_insert(codelet, worker_hint, args, bufs);
}

//! aten::max_pool2d_with_indices (out variant), 2-D only.
TorchMaxPool2dWithIndices::TorchMaxPool2dWithIndices():
    codelet(
        "nntile_torch_max_pool2d_with_indices",
        &TorchMaxPool2dWithIndices::cpu,
#ifdef NNTILE_USE_CUDA
        &TorchMaxPool2dWithIndices::cuda,
#else
        nullptr,
#endif
        &TorchMaxPool2dWithIndices::footprint)
{
}

void TorchMaxPool2dWithIndices::cpu(
    void *buffers[], void *cl_args) noexcept
{
    try
    {
        auto *args = reinterpret_cast<TorchDispatchArgs *>(cl_args);
        at::Tensor input = in_fp32(buf_as<float>(buffers, 0), *args, 0);
        at::Tensor out = out_fp32(buf_as<float>(buffers, 1), *args, 0);
        at::Tensor indices =
            out_i64(buf_as<std::int64_t>(buffers, 2), *args, 1);
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        at::max_pool2d_with_indices_out(
            out,
            indices,
            input,
            iarg_vec(*args, 0, 2),
            iarg_vec(*args, 2, 2),
            iarg_vec(*args, 4, 2),
            iarg_vec(*args, 6, 2),
            args->iargs[8] != 0);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_max_pool2d_with_indices failed: %s\n",
            ex.what());
        std::abort();
    }
}

#ifdef NNTILE_USE_CUDA
void TorchMaxPool2dWithIndices::cuda(
    void *buffers[], void *cl_args) noexcept
{
    try
    {
        HaulTorchCudaEnv cuda_env;
        (void)cuda_env;
        cpu(buffers, cl_args);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_max_pool2d_with_indices CUDA failed: %s\n",
            ex.what());
        std::abort();
    }
}
#endif // NNTILE_USE_CUDA

void TorchMaxPool2dWithIndices::submit(
    int worker_hint,
    args_t const &meta,
    TorchHandle const &input,
    TorchHandle const &out,
    TorchHandle const &indices)
{
    args_t args = meta;
    args.kind = TorchKind::MaxPool2dWithIndices;
    torch_insert(
        codelet,
        worker_hint,
        args,
        {BufSpec{STARPU_R, &input.get()},
         BufSpec{STARPU_W, &out.get()},
         BufSpec{STARPU_W, &indices.get()}});
}

//! aten::max_pool2d_with_indices_backward (out variant), 2-D only.
TorchMaxPool2dWithIndicesBackward::TorchMaxPool2dWithIndicesBackward():
    codelet(
        "nntile_torch_max_pool2d_with_indices_backward",
        &TorchMaxPool2dWithIndicesBackward::cpu,
#ifdef NNTILE_USE_CUDA
        &TorchMaxPool2dWithIndicesBackward::cuda,
#else
        nullptr,
#endif
        &TorchMaxPool2dWithIndicesBackward::footprint)
{
}

void TorchMaxPool2dWithIndicesBackward::cpu(
    void *buffers[], void *cl_args) noexcept
{
    try
    {
        auto *args = reinterpret_cast<TorchDispatchArgs *>(cl_args);
        at::Tensor grad_out =
            in_fp32(buf_as<float>(buffers, 0), *args, 0);
        at::Tensor input = in_fp32(buf_as<float>(buffers, 1), *args, 1);
        at::Tensor indices =
            in_i64(buf_as<std::int64_t>(buffers, 2), *args, 2);
        at::Tensor grad_input =
            out_fp32(buf_as<float>(buffers, 3), *args, 0);
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        at::max_pool2d_with_indices_backward_out(
            grad_input,
            grad_out,
            input,
            iarg_vec(*args, 0, 2),
            iarg_vec(*args, 2, 2),
            iarg_vec(*args, 4, 2),
            iarg_vec(*args, 6, 2),
            args->iargs[8] != 0,
            indices);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_max_pool2d_with_indices_backward failed: %s\n",
            ex.what());
        std::abort();
    }
}

#ifdef NNTILE_USE_CUDA
void TorchMaxPool2dWithIndicesBackward::cuda(
    void *buffers[], void *cl_args) noexcept
{
    try
    {
        HaulTorchCudaEnv cuda_env;
        (void)cuda_env;
        cpu(buffers, cl_args);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_max_pool2d_with_indices_backward CUDA failed: %s\n",
            ex.what());
        std::abort();
    }
}
#endif // NNTILE_USE_CUDA

void TorchMaxPool2dWithIndicesBackward::submit(
    int worker_hint,
    args_t const &meta,
    TorchHandle const &grad_out,
    TorchHandle const &input,
    TorchHandle const &indices,
    TorchHandle const &grad_input)
{
    args_t args = meta;
    args.kind = TorchKind::MaxPool2dWithIndicesBackward;
    torch_insert(
        codelet,
        worker_hint,
        args,
        {BufSpec{STARPU_R, &grad_out.get()},
         BufSpec{STARPU_R, &input.get()},
         BufSpec{STARPU_R, &indices.get()},
         BufSpec{STARPU_W, &grad_input.get()}});
}

TorchStub const torch_sdpa_backward{"sdpa_backward"};


//! native_batch_norm forward (the layer_norm composite lands here).
TorchNativeBatchNorm::TorchNativeBatchNorm():
    codelet(
        "nntile_torch_native_batch_norm",
        &TorchNativeBatchNorm::cpu,
#ifdef NNTILE_USE_CUDA
        &TorchNativeBatchNorm::cuda,
#else
        nullptr,
#endif
        &TorchNativeBatchNorm::footprint)
{
}

void TorchNativeBatchNorm::cpu(void *buffers[], void *cl_args) noexcept
{
    try
    {
        auto *args = reinterpret_cast<TorchDispatchArgs *>(cl_args);
        const bool training = args->iargs[0] != 0;
        const bool has_w = args->iargs[1] != 0;
        const bool has_b = args->iargs[2] != 0;
        const bool has_rm = args->iargs[3] != 0;
        const bool has_rv = args->iargs[4] != 0;
        Index buf = 0;
        at::Tensor input = in_fp32(buf_as<float>(buffers, buf++), *args, 0);
        at::Tensor out = out_fp32(buf_as<float>(buffers, buf++), *args, 0);
        at::Tensor save_mean =
            out_fp32(buf_as<float>(buffers, buf++), *args, 1);
        at::Tensor save_invstd =
            out_fp32(buf_as<float>(buffers, buf++), *args, 2);
        at::Tensor weight;
        at::Tensor bias;
        at::Tensor running_mean;
        at::Tensor running_var;
        if (has_w)
        {
            weight = in_fp32(buf_as<float>(buffers, buf++), *args, 1);
        }
        if (has_b)
        {
            bias = in_fp32(buf_as<float>(buffers, buf++), *args, 2);
        }
        if (has_rm)
        {
            running_mean =
                in_fp32(buf_as<float>(buffers, buf++), *args, 3);
        }
        if (has_rv)
        {
            running_var =
                in_fp32(buf_as<float>(buffers, buf++), *args, 4);
        }
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        at::native_batch_norm_out(
            out,
            save_mean,
            save_invstd,
            input,
            has_w ? c10::optional<at::Tensor>(weight) : c10::nullopt,
            has_b ? c10::optional<at::Tensor>(bias) : c10::nullopt,
            has_rm ? c10::optional<at::Tensor>(running_mean) : c10::nullopt,
            has_rv ? c10::optional<at::Tensor>(running_var) : c10::nullopt,
            training,
            static_cast<double>(args->scalars[0]),
            static_cast<double>(args->scalars[1]));
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_native_batch_norm failed: %s\n",
            ex.what());
        std::abort();
    }
}

#ifdef NNTILE_USE_CUDA
void TorchNativeBatchNorm::cuda(void *buffers[], void *cl_args) noexcept
{
    try
    {
        HaulTorchCudaEnv cuda_env;
        (void)cuda_env;
        cpu(buffers, cl_args);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_native_batch_norm CUDA failed: %s\n",
            ex.what());
        std::abort();
    }
}
#endif // NNTILE_USE_CUDA

void TorchNativeBatchNorm::submit(
    int worker_hint,
    args_t const &meta,
    TorchHandle const &input,
    TorchHandle const &weight,
    TorchHandle const &bias,
    TorchHandle const &running_mean,
    TorchHandle const &running_var,
    TorchHandle const &out,
    TorchHandle const &save_mean,
    TorchHandle const &save_invstd,
    bool has_weight,
    bool has_bias,
    bool has_running_mean,
    bool has_running_var,
    bool training)
{
    args_t args = meta;
    args.kind = TorchKind::NativeBatchNorm;
    args.iargs[0] = training ? 1 : 0;
    args.iargs[1] = has_weight ? 1 : 0;
    args.iargs[2] = has_bias ? 1 : 0;
    args.iargs[3] = has_running_mean ? 1 : 0;
    args.iargs[4] = has_running_var ? 1 : 0;
    std::vector<BufSpec> bufs;
    bufs.push_back(BufSpec{STARPU_R, &input.get()});
    bufs.push_back(BufSpec{STARPU_W, &out.get()});
    bufs.push_back(BufSpec{STARPU_W, &save_mean.get()});
    bufs.push_back(BufSpec{STARPU_W, &save_invstd.get()});
    if (has_weight)
    {
        bufs.push_back(BufSpec{STARPU_R, &weight.get()});
    }
    if (has_bias)
    {
        bufs.push_back(BufSpec{STARPU_R, &bias.get()});
    }
    if (has_running_mean)
    {
        bufs.push_back(BufSpec{
            training ? STARPU_RW : STARPU_R, &running_mean.get()});
    }
    if (has_running_var)
    {
        bufs.push_back(BufSpec{
            training ? STARPU_RW : STARPU_R, &running_var.get()});
    }
    torch_insert(codelet, worker_hint, args, bufs);
}

//! native_batch_norm backward.
TorchNativeBatchNormBackward::TorchNativeBatchNormBackward():
    codelet(
        "nntile_torch_native_batch_norm_backward",
        &TorchNativeBatchNormBackward::cpu,
#ifdef NNTILE_USE_CUDA
        &TorchNativeBatchNormBackward::cuda,
#else
        nullptr,
#endif
        &TorchNativeBatchNormBackward::footprint)
{
}

void TorchNativeBatchNormBackward::cpu(
    void *buffers[],
    void *cl_args) noexcept
{
    try
    {
        auto *args = reinterpret_cast<TorchDispatchArgs *>(cl_args);
        const bool training = args->iargs[0] != 0;
        const bool has_w = args->iargs[1] != 0;
        const bool has_rm = args->iargs[3] != 0;
        const bool has_rv = args->iargs[4] != 0;
        const bool has_sm = args->iargs[5] != 0;
        const bool has_si = args->iargs[6] != 0;
        const bool need_gi = args->iargs[7] != 0;
        const bool need_gw = args->iargs[8] != 0;
        const bool need_gb = args->iargs[9] != 0;
        Index buf = 0;
        at::Tensor grad_out =
            in_fp32(buf_as<float>(buffers, buf++), *args, 0);
        at::Tensor input =
            in_fp32(buf_as<float>(buffers, buf++), *args, 1);
        at::Tensor weight;
        at::Tensor running_mean;
        at::Tensor running_var;
        at::Tensor save_mean;
        at::Tensor save_invstd;
        if (has_w)
        {
            weight = in_fp32(buf_as<float>(buffers, buf++), *args, 2);
        }
        if (has_rm)
        {
            running_mean =
                in_fp32(buf_as<float>(buffers, buf++), *args, 3);
        }
        if (has_rv)
        {
            running_var =
                in_fp32(buf_as<float>(buffers, buf++), *args, 4);
        }
        if (has_sm)
        {
            save_mean = in_fp32(buf_as<float>(buffers, buf++), *args, 5);
        }
        if (has_si)
        {
            save_invstd =
                in_fp32(buf_as<float>(buffers, buf++), *args, 6);
        }
        at::Tensor grad_input;
        at::Tensor grad_weight;
        at::Tensor grad_bias;
        if (need_gi)
        {
            grad_input = out_fp32(buf_as<float>(buffers, buf++), *args, 0);
        }
        if (need_gw)
        {
            grad_weight = out_fp32(buf_as<float>(buffers, buf++), *args, 1);
        }
        if (need_gb)
        {
            grad_bias = out_fp32(buf_as<float>(buffers, buf++), *args, 2);
        }
        std::array<bool, 3> output_mask = {need_gi, need_gw, need_gb};
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        auto result = at::native_batch_norm_backward(
            grad_out,
            input,
            has_w ? c10::optional<at::Tensor>(weight) : c10::nullopt,
            has_rm
                ? c10::optional<at::Tensor>(running_mean)
                : c10::nullopt,
            has_rv
                ? c10::optional<at::Tensor>(running_var)
                : c10::nullopt,
            has_sm ? c10::optional<at::Tensor>(save_mean) : c10::nullopt,
            has_si
                ? c10::optional<at::Tensor>(save_invstd)
                : c10::nullopt,
            training,
            static_cast<double>(args->scalars[1]),
            output_mask);
        if (need_gi)
        {
            grad_input.copy_(std::get<0>(result));
        }
        if (need_gw)
        {
            grad_weight.copy_(std::get<1>(result));
        }
        if (need_gb)
        {
            grad_bias.copy_(std::get<2>(result));
        }
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_native_batch_norm_backward failed: %s\n",
            ex.what());
        std::abort();
    }
}

#ifdef NNTILE_USE_CUDA
void TorchNativeBatchNormBackward::cuda(
    void *buffers[],
    void *cl_args) noexcept
{
    try
    {
        HaulTorchCudaEnv cuda_env;
        (void)cuda_env;
        cpu(buffers, cl_args);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_native_batch_norm_backward CUDA failed: %s\n",
            ex.what());
        std::abort();
    }
}
#endif // NNTILE_USE_CUDA

void TorchNativeBatchNormBackward::submit(
    int worker_hint,
    args_t const &meta,
    TorchHandle const &grad_out,
    TorchHandle const &input,
    TorchHandle const &weight,
    TorchHandle const &running_mean,
    TorchHandle const &running_var,
    TorchHandle const &save_mean,
    TorchHandle const &save_invstd,
    TorchHandle const &grad_input,
    TorchHandle const &grad_weight,
    TorchHandle const &grad_bias,
    bool has_weight,
    bool has_running_mean,
    bool has_running_var,
    bool has_save_mean,
    bool has_save_invstd,
    bool need_grad_input,
    bool need_grad_weight,
    bool need_grad_bias)
{
    args_t args = meta;
    args.kind = TorchKind::NativeBatchNormBackward;
    args.iargs[1] = has_weight ? 1 : 0;
    args.iargs[3] = has_running_mean ? 1 : 0;
    args.iargs[4] = has_running_var ? 1 : 0;
    args.iargs[5] = has_save_mean ? 1 : 0;
    args.iargs[6] = has_save_invstd ? 1 : 0;
    args.iargs[7] = need_grad_input ? 1 : 0;
    args.iargs[8] = need_grad_weight ? 1 : 0;
    args.iargs[9] = need_grad_bias ? 1 : 0;
    std::vector<BufSpec> bufs;
    bufs.push_back(BufSpec{STARPU_R, &grad_out.get()});
    bufs.push_back(BufSpec{STARPU_R, &input.get()});
    if (has_weight)
    {
        bufs.push_back(BufSpec{STARPU_R, &weight.get()});
    }
    if (has_running_mean)
    {
        bufs.push_back(BufSpec{STARPU_R, &running_mean.get()});
    }
    if (has_running_var)
    {
        bufs.push_back(BufSpec{STARPU_R, &running_var.get()});
    }
    if (has_save_mean)
    {
        bufs.push_back(BufSpec{STARPU_R, &save_mean.get()});
    }
    if (has_save_invstd)
    {
        bufs.push_back(BufSpec{STARPU_R, &save_invstd.get()});
    }
    if (need_grad_input)
    {
        bufs.push_back(BufSpec{STARPU_W, &grad_input.get()});
    }
    if (need_grad_weight)
    {
        bufs.push_back(BufSpec{STARPU_W, &grad_weight.get()});
    }
    if (need_grad_bias)
    {
        bufs.push_back(BufSpec{STARPU_W, &grad_bias.get()});
    }
    torch_insert(codelet, worker_hint, args, bufs);
}

TorchNativeBatchNorm torch_native_batch_norm;
TorchNativeBatchNormBackward torch_native_batch_norm_backward;
TorchEmbedding torch_embedding;
TorchEmbeddingDenseBackward torch_embedding_dense_backward;
TorchNllLossForward torch_nll_loss_forward;
TorchNllLossBackward torch_nll_loss_backward;
TorchCat torch_cat;
TorchWhere torch_where;
TorchArange torch_arange;
TorchGt torch_gt;
TorchI64Unary torch_i64_unary;
TorchCast torch_cast;
TorchConvolution torch_convolution;
TorchConvolutionBackward torch_convolution_backward;
TorchMaxPool2dWithIndices torch_max_pool2d_with_indices;
TorchMaxPool2dWithIndicesBackward torch_max_pool2d_with_indices_backward;

} // namespace nntile::haul
