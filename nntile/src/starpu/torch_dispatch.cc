/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file src/starpu/torch_dispatch.cc
 * Torch-native family StarPU codelets (CPU/CUDA aten *_out).
 *
 * @version 1.1.0
 */

#include "nntile/starpu/torch_dispatch.hh"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/core/grad_mode.h>
#include <ATen/ops/_adaptive_avg_pool2d.h>
#include <ATen/ops/_adaptive_avg_pool2d_backward.h>
#include <ATen/ops/add.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/eq.h>
#include <ATen/ops/gt.h>
#include <ATen/ops/log.h>
#include <ATen/ops/lt.h>
#include <ATen/ops/minimum.h>
#include <ATen/ops/sub.h>
#include <ATen/ops/triu.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/avg_pool2d.h>
#include <ATen/ops/avg_pool2d_backward.h>
#include <ATen/ops/bmm.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/convolution.h>
#include <ATen/ops/convolution_backward.h>
#include <ATen/ops/cos.h>
#include <ATen/ops/_log_softmax.h>
#include <ATen/ops/_log_softmax_backward_data.h>
#include <ATen/ops/_scaled_dot_product_flash_attention_for_cpu.h>
#include <ATen/ops/_scaled_dot_product_flash_attention_for_cpu_backward.h>
#ifdef NNTILE_USE_CUDA
#include <ATen/ops/_scaled_dot_product_efficient_attention.h>
#include <ATen/ops/_scaled_dot_product_efficient_attention_backward.h>
#endif
#include <ATen/ops/embedding.h>
#include <ATen/ops/embedding_dense_backward.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/gelu.h>
#include <ATen/ops/gelu_backward.h>
#include <ATen/ops/hypot.h>
#include <ATen/ops/linalg_vector_norm.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/masked_fill.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/max_pool2d_with_indices.h>
#include <ATen/ops/max_pool2d_with_indices_backward.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/native_batch_norm.h>
#include <ATen/ops/native_batch_norm_backward.h>
#include <ATen/ops/native_layer_norm.h>
#include <ATen/ops/native_layer_norm_backward.h>
#include <ATen/ops/neg.h>
#include <ATen/ops/nll_loss_backward.h>
#include <ATen/ops/nll_loss_forward.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/repeat.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <ATen/ops/silu.h>
#include <ATen/ops/silu_backward.h>
#include <ATen/ops/sin.h>
#include <ATen/ops/softmax.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/threshold_backward.h>
#include <ATen/ops/tril.h>
#include <ATen/ops/upsample_bilinear2d.h>
#include <ATen/ops/upsample_bilinear2d_backward.h>
#include <ATen/ops/upsample_nearest2d.h>
#include <ATen/ops/upsample_nearest2d_backward.h>
#include <ATen/ops/where.h>
#include <ATen/ops/zeros.h>

#include "nntile/starpu/torch_blob.hh"
#ifdef NNTILE_USE_CUDA
#include "nntile/starpu/torch_cuda_env.hh"
#endif

namespace nntile::starpu
{

namespace
{

using torch_blob::blob_bool;
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

//! Packed dtype tags for Cast / Where: 0=fp32, 1=i64, 2=bool.
at::Tensor in_tagged(
    VariableInterface *iface,
    const TorchDispatchArgs &args,
    Index slot,
    Index tag)
{
    switch (tag)
    {
    case 0:
        return in_fp32(iface->get_ptr<float>(), args, slot);
    case 1:
        return in_i64(iface->get_ptr<std::int64_t>(), args, slot);
    case 2:
        return in_bool(
            reinterpret_cast<bool *>(iface->get_ptr<bool_t>()),
            args,
            slot);
    default:
        throw std::runtime_error("torch_cast: bad src dtype tag");
    }
}

at::Tensor out_tagged(
    VariableInterface *iface,
    const TorchDispatchArgs &args,
    Index slot,
    Index tag)
{
    switch (tag)
    {
    case 0:
        return out_fp32(iface->get_ptr<float>(), args, slot);
    case 1:
        return out_i64(iface->get_ptr<std::int64_t>(), args, slot);
    case 2:
        return out_bool(
            reinterpret_cast<bool *>(iface->get_ptr<bool_t>()),
            args,
            slot);
    default:
        throw std::runtime_error("torch_cast: bad dst dtype tag");
    }
}

//! CopyIntoView may submit a single STARPU_RW buffer when src/dst alias.
bool copy_into_view_aliases_in(const TorchDispatchArgs *args)
{
    return args->kind == TorchKind::CopyIntoView &&
        args->iargs[7] != 0;
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
    Index has_slot,
    Index value_slot)
{
    if (args.iargs[has_slot] == 0)
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
        // works on CPU and CUDA (StarPU worker stream).
        at::Tensor src = self.narrow(dim, start, length);
        if (src.sizes() != result.sizes())
        {
            auto fmt = [](at::IntArrayRef dims) {
                std::string s = "[";
                for (size_t i = 0; i < dims.size(); ++i)
                {
                    if (i != 0)
                    {
                        s += ",";
                    }
                    s += std::to_string(dims[i]);
                }
                return s + "]";
            };
            throw std::runtime_error(
                "torch NarrowCopy: size mismatch after narrow "
                "narrowed=" +
                fmt(src.sizes()) +
                " out=" +
                fmt(result.sizes()));
        }
        result.copy_(src);
        break;
    }
    case TorchKind::Copy:
    case TorchKind::CopyIntoView:
    {
        // Copy: densify a view into contiguous out.
        // CopyIntoView: packed out layout is a view of the parent
        // StarPU buffer (STARPU_RW); copy_ writes only that region.
        if (self.sizes() != result.sizes())
        {
            auto fmt = [](at::IntArrayRef dims) {
                std::string s = "[";
                for (size_t i = 0; i < dims.size(); ++i)
                {
                    if (i != 0)
                    {
                        s += ",";
                    }
                    s += std::to_string(dims[i]);
                }
                return s + "]";
            };
            throw std::runtime_error(
                "torch Copy: in/out size mismatch in=" +
                fmt(self.sizes()) +
                " out=" +
                fmt(result.sizes()) +
                " (packed layout meta must match logical "
                "tensor sizes; unpacked slots fall back to "
                "storage tile shape)");
        }
        result.copy_(self);
        break;
    }
    case TorchKind::Repeat:
    {
        // Factors are stored for the *output* rank (tensor_repeat_fp32).
        // Do not use in_ndim: the StarPU tile may still be the parent 1D
        // bias storage while factors pad a leading dim (addmm / linear
        // bias broadcast). Using in_ndim alone made aten::repeat_out
        // target a flat [numel] shape and resize_output-warn on the
        // preallocated 2D out.
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
            auto fmt = [](at::IntArrayRef dims) {
                std::string s = "[";
                for (size_t i = 0; i < dims.size(); ++i)
                {
                    if (i != 0)
                    {
                        s += ",";
                    }
                    s += std::to_string(dims[i]);
                }
                return s + "]";
            };
            throw std::runtime_error(
                "torch TransposeCopy: size mismatch after "
                "transpose transposed=" +
                fmt(src.sizes()) +
                " out=" +
                fmt(result.sizes()) +
                " in=" +
                fmt(self.sizes()));
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
    case TorchKind::AvgPool2dBackward:
        at::avg_pool2d_backward_out(
            result,
            ta,
            tb,
            iarg_vec(*args, 0, 2),
            iarg_vec(*args, 2, 2),
            iarg_vec(*args, 4, 2),
            args->iargs[6] != 0,
            args->iargs[7] != 0,
            optional_iarg(*args, 8, 9));
        break;
    case TorchKind::AdaptiveAvgPool2dBackward:
        at::_adaptive_avg_pool2d_backward_out(result, ta, tb);
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

//! Math SDPA backward via matmul/softmax (fallback; materializes SxS).
void run_sdpa_math_backward(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &grad_out,
    const c10::optional<at::Tensor> &attn_mask,
    bool is_causal,
    at::Tensor &grad_q,
    at::Tensor &grad_k,
    at::Tensor &grad_v)
{
    const double scale =
        1.0 / std::sqrt(static_cast<double>(q.size(-1)));
    at::Tensor scores = at::matmul(q, k.transpose(-2, -1));
    scores = scores.mul(scale);
    if (is_causal)
    {
        const auto L = scores.size(-2);
        const auto S = scores.size(-1);
        at::Tensor keep = at::ones(
            {L, S},
            q.options().dtype(at::kBool)).tril();
        scores.masked_fill_(
            ~keep,
            -std::numeric_limits<float>::infinity());
    }
    if (attn_mask.has_value())
    {
        scores = scores.add(*attn_mask);
    }
    at::Tensor attn = at::softmax(scores, /*dim=*/-1);
    at::Tensor d_attn =
        at::matmul(grad_out, v.transpose(-2, -1));
    at::Tensor d_scores = attn.mul(
        d_attn.sub((attn.mul(d_attn)).sum(-1, /*keepdim=*/true)));
    d_scores = d_scores.mul(scale);
    grad_q.copy_(at::matmul(d_scores, k));
    grad_k.copy_(at::matmul(d_scores.transpose(-2, -1), q));
    grad_v.copy_(at::matmul(attn.transpose(-2, -1), grad_out));
}

#ifdef NNTILE_USE_CUDA
//! CUDA mem-efficient SDPA backward (fp32-capable; no SxS materialize).
void run_sdpa_efficient_backward(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &grad_out,
    const c10::optional<at::Tensor> &attn_mask,
    bool is_causal,
    at::Tensor &grad_q,
    at::Tensor &grad_k,
    at::Tensor &grad_v)
{
    at::Tensor qc = q.contiguous();
    at::Tensor kc = k.contiguous();
    at::Tensor vc = v.contiguous();
    at::Tensor goc = grad_out.contiguous();
    c10::optional<at::Tensor> bias = c10::nullopt;
    if (attn_mask.has_value())
    {
        bias = attn_mask->contiguous();
    }
    auto fwd = at::_scaled_dot_product_efficient_attention(
        qc,
        kc,
        vc,
        bias,
        /*compute_log_sumexp=*/true,
        /*dropout_p=*/0.0,
        is_causal,
        /*scale=*/c10::nullopt);
    at::Tensor bias_tensor =
        bias.has_value() ? *bias : at::Tensor();
    auto bwd = at::_scaled_dot_product_efficient_attention_backward(
        goc,
        qc,
        kc,
        vc,
        bias_tensor,
        std::get<0>(fwd),
        std::get<1>(fwd),
        std::get<2>(fwd),
        std::get<3>(fwd),
        /*dropout_p=*/0.0,
        std::array<bool, 4>{true, true, true, false},
        is_causal,
        /*scale=*/c10::nullopt);
    grad_q.copy_(std::get<0>(bwd));
    grad_k.copy_(std::get<1>(bwd));
    grad_v.copy_(std::get<2>(bwd));
}

//! CUDA SDPA backward: efficient first, math fallback.
void run_sdpa_cuda_backward(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &grad_out,
    const c10::optional<at::Tensor> &attn_mask,
    bool is_causal,
    at::Tensor &grad_q,
    at::Tensor &grad_k,
    at::Tensor &grad_v)
{
    try
    {
        run_sdpa_efficient_backward(
            q,
            k,
            v,
            grad_out,
            attn_mask,
            is_causal,
            grad_q,
            grad_k,
            grad_v);
    }
    catch (const c10::Error &)
    {
        run_sdpa_math_backward(
            q,
            k,
            v,
            grad_out,
            attn_mask,
            is_causal,
            grad_q,
            grad_k,
            grad_v);
    }
}
#endif // NNTILE_USE_CUDA

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

int submit_accesses(
    Codelet *codelet,
    int starpu_worker_hint,
    TorchDispatchArgs *args,
    const std::vector<std::pair<enum starpu_data_access_mode, Handle>>
        &handles)
{
    if (handles.empty() || handles.size() > 10)
    {
        std::free(args);
        throw std::runtime_error("torch_dispatch.submit: bad handle count");
    }
    starpu_data_handle_t h[10];
    enum starpu_data_access_mode m[10];
    const Index n = static_cast<Index>(handles.size());
    for (Index i = 0; i < n; ++i)
    {
        m[static_cast<size_t>(i)] = handles[static_cast<size_t>(i)].first;
        h[static_cast<size_t>(i)] =
            handles[static_cast<size_t>(i)].second.get();
    }
    switch (n)
    {
    case 1:
        return nntile_starpu_task_insert(
            codelet, starpu_worker_hint, m[0], h[0],
            STARPU_CL_ARGS, args, sizeof(*args), 0);
    case 2:
        return nntile_starpu_task_insert(
            codelet, starpu_worker_hint, m[0], h[0], m[1], h[1],
            STARPU_CL_ARGS, args, sizeof(*args), 0);
    case 3:
        return nntile_starpu_task_insert(
            codelet, starpu_worker_hint, m[0], h[0], m[1], h[1],
            m[2], h[2], STARPU_CL_ARGS, args, sizeof(*args), 0);
    case 4:
        return nntile_starpu_task_insert(
            codelet, starpu_worker_hint, m[0], h[0], m[1], h[1],
            m[2], h[2], m[3], h[3], STARPU_CL_ARGS, args,
            sizeof(*args), 0);
    case 5:
        return nntile_starpu_task_insert(
            codelet, starpu_worker_hint, m[0], h[0], m[1], h[1],
            m[2], h[2], m[3], h[3], m[4], h[4],
            STARPU_CL_ARGS, args, sizeof(*args), 0);
    case 6:
        return nntile_starpu_task_insert(
            codelet, starpu_worker_hint, m[0], h[0], m[1], h[1],
            m[2], h[2], m[3], h[3], m[4], h[4], m[5], h[5],
            STARPU_CL_ARGS, args, sizeof(*args), 0);
    case 7:
        return nntile_starpu_task_insert(
            codelet, starpu_worker_hint, m[0], h[0], m[1], h[1],
            m[2], h[2], m[3], h[3], m[4], h[4], m[5], h[5],
            m[6], h[6], STARPU_CL_ARGS, args, sizeof(*args), 0);
    case 8:
        return nntile_starpu_task_insert(
            codelet, starpu_worker_hint, m[0], h[0], m[1], h[1],
            m[2], h[2], m[3], h[3], m[4], h[4], m[5], h[5],
            m[6], h[6], m[7], h[7], STARPU_CL_ARGS, args,
            sizeof(*args), 0);
    case 9:
        return nntile_starpu_task_insert(
            codelet, starpu_worker_hint, m[0], h[0], m[1], h[1],
            m[2], h[2], m[3], h[3], m[4], h[4], m[5], h[5],
            m[6], h[6], m[7], h[7], m[8], h[8],
            STARPU_CL_ARGS, args, sizeof(*args), 0);
    case 10:
        return nntile_starpu_task_insert(
            codelet, starpu_worker_hint, m[0], h[0], m[1], h[1],
            m[2], h[2], m[3], h[3], m[4], h[4], m[5], h[5],
            m[6], h[6], m[7], h[7], m[8], h[8], m[9], h[9],
            STARPU_CL_ARGS, args, sizeof(*args), 0);
    default:
        std::free(args);
        throw std::runtime_error("torch_dispatch.submit: bad handle count");
    }
}

} // namespace

template<typename T>
TorchUnary<std::tuple<T>>::TorchUnary():
    codelet("nntile_torch_unary", footprint, cpu_funcs, cuda_funcs)
{
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
        float *out = copy_into_view_aliases_in(args)
            ? in
            : ifaces[1]->get_ptr<float>();
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        run_unary(args, in, out, at::kCPU);
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


#ifdef NNTILE_USE_CUDA
template<>
void TorchUnary<std::tuple<fp32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        TorchCudaEnv cuda_env;
        auto *args = reinterpret_cast<args_t *>(cl_args);
        auto **ifaces =
            reinterpret_cast<VariableInterface **>(buffers);
        float *in = ifaces[0]->get_ptr<float>();
        float *out = copy_into_view_aliases_in(args)
            ? in
            : ifaces[1]->get_ptr<float>();
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
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
#endif
}
#endif // NNTILE_USE_CUDA

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
    int ret = 0;
    if (args->kind == TorchKind::CopyIntoView)
    {
        // Preserve parent values outside the packed view (RW, not W).
        const bool out_aliases_in = (out.get() == in.get());
        args->iargs[7] = out_aliases_in ? 1 : 0;
        if (out_aliases_in)
        {
            ret = nntile_starpu_task_insert(
                &codelet,
                starpu_worker_hint,
                STARPU_RW,
                in.get(),
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
                in.get(),
                STARPU_CL_ARGS,
                args,
                sizeof(*args),
                STARPU_RW,
                out.get(),
                0);
        }
    }
    else
    {
        ret = nntile_starpu_task_insert(
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
    }
    if (ret != 0)
    {
        throw std::runtime_error("torch_unary.submit failed");
    }
}

template<typename T>
TorchBinary<std::tuple<T>>::TorchBinary():
    codelet("nntile_torch_binary", footprint, cpu_funcs, cuda_funcs)
{
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
        run_binary(args, a, b, out, at::kCPU);
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


#ifdef NNTILE_USE_CUDA
template<>
void TorchBinary<std::tuple<fp32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        TorchCudaEnv cuda_env;
        auto *args = reinterpret_cast<args_t *>(cl_args);
        auto **ifaces =
            reinterpret_cast<VariableInterface **>(buffers);
        float *a = ifaces[0]->get_ptr<float>();
        float *b = ifaces[1]->get_ptr<float>();
        float *out = ifaces[2]->get_ptr<float>();
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        run_binary(args, a, b, out, cuda_env.device());
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_binary CUDA failed: %s\n",
            ex.what());
        std::abort();
    }
#endif
}
#endif // NNTILE_USE_CUDA

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
        run_ternary(args, a, b, c, out, at::kCPU);
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


#ifdef NNTILE_USE_CUDA
template<>
void TorchTernary<std::tuple<fp32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        TorchCudaEnv cuda_env;
        auto *args = reinterpret_cast<args_t *>(cl_args);
        auto **ifaces =
            reinterpret_cast<VariableInterface **>(buffers);
        float *a = ifaces[0]->get_ptr<float>();
        float *b = ifaces[1]->get_ptr<float>();
        float *c = ifaces[2]->get_ptr<float>();
        // Match CPU: iargs[7] means out aliases first input (3 buffers).
        float *out = (args->iargs[7] != 0)
            ? a
            : ifaces[3]->get_ptr<float>();
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        run_ternary(args, a, b, c, out, cuda_env.device());
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_ternary CUDA failed: %s\n",
            ex.what());
        std::abort();
    }
#endif
}
#endif // NNTILE_USE_CUDA

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
        at::Tensor idx = in_i64(indices, *args, 1);
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


#ifdef NNTILE_USE_CUDA
void TorchEmbedding::cuda(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        // StarPU stream + cuBLAS; TLS blob device = CUDA.
        TorchCudaEnv cuda_env;
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
#endif
}
#endif // NNTILE_USE_CUDA

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

TorchWhere::TorchWhere():
    codelet("nntile_torch_where", footprint, cpu_funcs, cuda_funcs)
{
}

void TorchWhere::cpu(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        auto *args = reinterpret_cast<args_t *>(cl_args);
        auto **ifaces =
            reinterpret_cast<VariableInterface **>(buffers);
        bool_t *cond_ptr = ifaces[0]->get_ptr<bool_t>();
        at::Tensor cond = in_bool(
            reinterpret_cast<bool *>(cond_ptr),
            *args,
            0);
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        if (args->iargs[15] != 0)
        {
            std::int64_t *self_ptr =
                ifaces[1]->get_ptr<std::int64_t>();
            std::int64_t *other_ptr =
                ifaces[2]->get_ptr<std::int64_t>();
            std::int64_t *out_ptr =
                ifaces[3]->get_ptr<std::int64_t>();
            at::Tensor self = in_i64(self_ptr, *args, 1);
            at::Tensor other = in_i64(other_ptr, *args, 2);
            at::Tensor result = out_i64(out_ptr, *args, 0);
            at::where_out(result, cond, self, other);
        }
        else
        {
            float *self_ptr = ifaces[1]->get_ptr<float>();
            float *other_ptr = ifaces[2]->get_ptr<float>();
            float *out_ptr = ifaces[3]->get_ptr<float>();
            at::Tensor self = in_fp32(self_ptr, *args, 1);
            at::Tensor other = in_fp32(other_ptr, *args, 2);
            at::Tensor result = out_fp32(out_ptr, *args, 0);
            at::where_out(result, cond, self, other);
        }
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_where failed: %s\n",
            ex.what());
        std::abort();
    }
#endif
}


#ifdef NNTILE_USE_CUDA
void TorchWhere::cuda(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        // StarPU stream + cuBLAS; TLS blob device = CUDA.
        TorchCudaEnv cuda_env;
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
#endif
}
#endif // NNTILE_USE_CUDA

uint32_t TorchWhere::footprint(struct starpu_task *task)
{
    return args_footprint(
        reinterpret_cast<args_t *>(task->cl_arg));
}

void TorchWhere::submit(
    int starpu_worker_hint,
    const args_t &meta,
    Handle condition,
    Handle self,
    Handle other,
    Handle out
)
{
    args_t *args = clone_args(meta);
    int ret = nntile_starpu_task_insert(
        &codelet,
        starpu_worker_hint,
        STARPU_R,
        condition.get(),
        STARPU_R,
        self.get(),
        STARPU_R,
        other.get(),
        STARPU_CL_ARGS,
        args,
        sizeof(*args),
        STARPU_W,
        out.get(),
        0);
    if (ret != 0)
    {
        throw std::runtime_error("torch_where.submit failed");
    }
}

TorchArange::TorchArange():
    codelet("nntile_torch_arange", footprint, cpu_funcs, cuda_funcs)
{
}

void TorchArange::cpu(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        auto *args = reinterpret_cast<args_t *>(cl_args);
        auto **ifaces =
            reinterpret_cast<VariableInterface **>(buffers);
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        if (args->kind == TorchKind::ArangeFp32)
        {
            float *out_ptr = ifaces[0]->get_ptr<float>();
            at::Tensor result = out_fp32(out_ptr, *args, 0);
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
            std::int64_t *out_ptr =
                ifaces[0]->get_ptr<std::int64_t>();
            at::Tensor result = out_i64(out_ptr, *args, 0);
            result.fill_(
                static_cast<std::int64_t>(args->iargs[0]));
        }
        else
        {
            std::int64_t *out_ptr =
                ifaces[0]->get_ptr<std::int64_t>();
            at::Tensor result = out_i64(out_ptr, *args, 0);
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
        std::fprintf(
            stderr,
            "nntile_torch_arange failed: %s\n",
            ex.what());
        std::abort();
    }
#endif
}


#ifdef NNTILE_USE_CUDA
void TorchArange::cuda(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        TorchCudaEnv cuda_env;
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
#endif
}
#endif // NNTILE_USE_CUDA

uint32_t TorchArange::footprint(struct starpu_task *task)
{
    return args_footprint(
        reinterpret_cast<args_t *>(task->cl_arg));
}

void TorchArange::submit(
    int starpu_worker_hint,
    const args_t &meta,
    Handle out
)
{
    args_t *args = clone_args(meta);
    int ret = nntile_starpu_task_insert(
        &codelet,
        starpu_worker_hint,
        STARPU_CL_ARGS,
        args,
        sizeof(*args),
        STARPU_W,
        out.get(),
        0);
    if (ret != 0)
    {
        throw std::runtime_error("torch_arange.submit failed");
    }
}

TorchGt::TorchGt():
    codelet("nntile_torch_gt", footprint, cpu_funcs, cuda_funcs)
{
}

void TorchGt::cpu(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        auto *args = reinterpret_cast<args_t *>(cl_args);
        auto **ifaces =
            reinterpret_cast<VariableInterface **>(buffers);
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        if (args->kind == TorchKind::Eq)
        {
            float *a_ptr = ifaces[0]->get_ptr<float>();
            float *b_ptr = ifaces[1]->get_ptr<float>();
            bool_t *out_ptr = ifaces[2]->get_ptr<bool_t>();
            at::Tensor ta = in_fp32(a_ptr, *args, 0);
            at::Tensor tb = in_fp32(b_ptr, *args, 1);
            at::Tensor result = out_bool(
                reinterpret_cast<bool *>(out_ptr),
                *args,
                0);
            at::eq_out(result, ta, tb);
        }
        else
        {
            std::int64_t *a_ptr = ifaces[0]->get_ptr<std::int64_t>();
            std::int64_t *b_ptr = ifaces[1]->get_ptr<std::int64_t>();
            at::Tensor ta = in_i64(a_ptr, *args, 0);
            at::Tensor tb = in_i64(b_ptr, *args, 1);
            switch (args->kind)
            {
        case TorchKind::Lt:
        {
            bool_t *out_ptr = ifaces[2]->get_ptr<bool_t>();
            at::Tensor result = out_bool(
                reinterpret_cast<bool *>(out_ptr),
                *args,
                0);
            at::lt_out(result, ta, tb);
            break;
        }
        case TorchKind::Sub:
        {
            std::int64_t *out_ptr =
                ifaces[2]->get_ptr<std::int64_t>();
            at::Tensor result = out_i64(out_ptr, *args, 0);
            at::sub_out(result, ta, tb, /*alpha=*/1);
            break;
        }
        case TorchKind::Add:
        {
            std::int64_t *out_ptr =
                ifaces[2]->get_ptr<std::int64_t>();
            at::Tensor result = out_i64(out_ptr, *args, 0);
            at::add_out(result, ta, tb, /*alpha=*/1);
            break;
        }
        case TorchKind::Mul:
        {
            std::int64_t *out_ptr =
                ifaces[2]->get_ptr<std::int64_t>();
            at::Tensor result = out_i64(out_ptr, *args, 0);
            at::mul_out(result, ta, tb);
            break;
        }
        case TorchKind::Minimum:
        {
            std::int64_t *out_ptr =
                ifaces[2]->get_ptr<std::int64_t>();
            at::Tensor result = out_i64(out_ptr, *args, 0);
            at::minimum_out(result, ta, tb);
            break;
        }
        default:
        {
            bool_t *out_ptr = ifaces[2]->get_ptr<bool_t>();
            at::Tensor result = out_bool(
                reinterpret_cast<bool *>(out_ptr),
                *args,
                0);
            at::gt_out(result, ta, tb);
            break;
        }
            }
        }
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_gt failed: %s\n",
            ex.what());
        std::abort();
    }
#endif
}


#ifdef NNTILE_USE_CUDA
void TorchGt::cuda(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        TorchCudaEnv cuda_env;
        (void)cuda_env;
        cpu(buffers, cl_args);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_gt CUDA failed: %s\n",
            ex.what());
        std::abort();
    }
#endif
}
#endif // NNTILE_USE_CUDA

uint32_t TorchGt::footprint(struct starpu_task *task)
{
    return args_footprint(
        reinterpret_cast<args_t *>(task->cl_arg));
}

void TorchGt::submit(
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
        throw std::runtime_error("torch_gt.submit failed");
    }
}

TorchI64Unary::TorchI64Unary():
    codelet("nntile_torch_i64_unary", footprint, cpu_funcs, cuda_funcs)
{
}

void TorchI64Unary::cpu(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        auto *args = reinterpret_cast<args_t *>(cl_args);
        auto **ifaces =
            reinterpret_cast<VariableInterface **>(buffers);
        std::int64_t *in_ptr = ifaces[0]->get_ptr<std::int64_t>();
        std::int64_t *out_ptr = copy_into_view_aliases_in(args)
            ? in_ptr
            : ifaces[1]->get_ptr<std::int64_t>();
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
#endif
}


#ifdef NNTILE_USE_CUDA
void TorchI64Unary::cuda(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        TorchCudaEnv cuda_env;
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
#endif
}
#endif // NNTILE_USE_CUDA

uint32_t TorchI64Unary::footprint(struct starpu_task *task)
{
    return args_footprint(
        reinterpret_cast<args_t *>(task->cl_arg));
}

void TorchI64Unary::submit(
    int starpu_worker_hint,
    const args_t &meta,
    Handle in,
    Handle out
)
{
    args_t *args = clone_args(meta);
    int ret = 0;
    if (args->kind == TorchKind::CopyIntoView)
    {
        const bool out_aliases_in = (out.get() == in.get());
        args->iargs[7] = out_aliases_in ? 1 : 0;
        if (out_aliases_in)
        {
            ret = nntile_starpu_task_insert(
                &codelet,
                starpu_worker_hint,
                STARPU_RW,
                in.get(),
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
                in.get(),
                STARPU_CL_ARGS,
                args,
                sizeof(*args),
                STARPU_RW,
                out.get(),
                0);
        }
    }
    else
    {
        ret = nntile_starpu_task_insert(
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
    }
    if (ret != 0)
    {
        throw std::runtime_error("torch_i64_unary.submit failed");
    }
}

TorchCast::TorchCast():
    codelet("nntile_torch_cast", footprint, cpu_funcs, cuda_funcs)
{
}

void TorchCast::cpu(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        auto *args = reinterpret_cast<args_t *>(cl_args);
        auto **ifaces =
            reinterpret_cast<VariableInterface **>(buffers);
        at::Tensor self = in_tagged(
            ifaces[0],
            *args,
            0,
            args->iargs[0]);
        at::Tensor result = out_tagged(
            ifaces[1],
            *args,
            0,
            args->iargs[1]);
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        result.copy_(self);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_cast failed: %s\n",
            ex.what());
        std::abort();
    }
#endif
}


#ifdef NNTILE_USE_CUDA
void TorchCast::cuda(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        TorchCudaEnv cuda_env;
        (void)cuda_env;
        cpu(buffers, cl_args);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_cast CUDA failed: %s\n",
            ex.what());
        std::abort();
    }
#endif
}
#endif // NNTILE_USE_CUDA

uint32_t TorchCast::footprint(struct starpu_task *task)
{
    return args_footprint(
        reinterpret_cast<args_t *>(task->cl_arg));
}

void TorchCast::submit(
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
        throw std::runtime_error("torch_cast.submit failed");
    }
}

TorchCat::TorchCat():
    codelet("nntile_torch_cat", footprint, cpu_funcs, cuda_funcs)
{
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


#ifdef NNTILE_USE_CUDA
void TorchCat::cuda(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        // StarPU stream + cuBLAS; TLS blob device = CUDA.
        TorchCudaEnv cuda_env;
        (void)cuda_env;
        cpu(buffers, cl_args);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_cat CUDA failed: %s\n",
            ex.what());
        std::abort();
    }
#endif
}
#endif // NNTILE_USE_CUDA

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
    // starpu_task_insert is varargs — fixed switch covers 1..max_tensors.
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
    case 5:
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
            STARPU_R,
            handles[4],
            STARPU_CL_ARGS,
            args,
            sizeof(*args),
            STARPU_W,
            out.get(),
            0);
        break;
    case 6:
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
            STARPU_R,
            handles[4],
            STARPU_R,
            handles[5],
            STARPU_CL_ARGS,
            args,
            sizeof(*args),
            STARPU_W,
            out.get(),
            0);
        break;
    case 7:
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
            STARPU_R,
            handles[4],
            STARPU_R,
            handles[5],
            STARPU_R,
            handles[6],
            STARPU_CL_ARGS,
            args,
            sizeof(*args),
            STARPU_W,
            out.get(),
            0);
        break;
    case 8:
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
            STARPU_R,
            handles[4],
            STARPU_R,
            handles[5],
            STARPU_R,
            handles[6],
            STARPU_R,
            handles[7],
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
            "torch_cat.submit: unexpected input count");
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
        // ATen may keep normalized dims as size-1; NNTile stores reduced
        // mean/rstd without those axes.
        mean.copy_(std::get<1>(ln).reshape(mean.sizes()));
        rstd.copy_(std::get<2>(ln).reshape(rstd.sizes()));
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


#ifdef NNTILE_USE_CUDA
void TorchLayerNorm::cuda(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        // StarPU stream + cuBLAS; TLS blob device = CUDA.
        TorchCudaEnv cuda_env;
        (void)cuda_env;
        cpu(buffers, cl_args);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_layer_norm CUDA failed: %s\n",
            ex.what());
        std::abort();
    }
#endif
}
#endif // NNTILE_USE_CUDA

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

TorchLayerNormBackward::TorchLayerNormBackward():
    codelet(
        "nntile_torch_layer_norm_backward",
        footprint,
        cpu_funcs,
        cuda_funcs)
{
}

void TorchLayerNormBackward::cpu(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        auto *args = reinterpret_cast<args_t *>(cl_args);
        auto **ifaces =
            reinterpret_cast<VariableInterface **>(buffers);
        const bool has_w = args->iargs[1] != 0;
        const bool has_b = args->iargs[2] != 0;
        const bool need_gi = args->iargs[3] != 0;
        const bool need_gw = args->iargs[4] != 0;
        const bool need_gb = args->iargs[5] != 0;
        Index buf = 0;
        float *grad_out_ptr = ifaces[buf++]->get_ptr<float>();
        float *input_ptr = ifaces[buf++]->get_ptr<float>();
        float *mean_ptr = ifaces[buf++]->get_ptr<float>();
        float *rstd_ptr = ifaces[buf++]->get_ptr<float>();
        float *gi_ptr = need_gi ? ifaces[buf++]->get_ptr<float>()
            : nullptr;
        float *gw_ptr = need_gw ? ifaces[buf++]->get_ptr<float>()
            : nullptr;
        float *gb_ptr = need_gb ? ifaces[buf++]->get_ptr<float>()
            : nullptr;
        at::Tensor weight;
        at::Tensor bias;
        if (has_w)
        {
            weight = in_fp32(
                ifaces[buf++]->get_ptr<float>(),
                *args,
                4);
        }
        if (has_b)
        {
            bias = in_fp32(
                ifaces[buf++]->get_ptr<float>(),
                *args,
                5);
        }
        at::Tensor grad_out = in_fp32(grad_out_ptr, *args, 0);
        at::Tensor input = in_fp32(input_ptr, *args, 1);
        at::Tensor mean = in_fp32(mean_ptr, *args, 2);
        at::Tensor rstd = in_fp32(rstd_ptr, *args, 3);
        // ATen may expect keepdim stats; reshape reduced buffers.
        const std::int64_t n =
            static_cast<std::int64_t>(args->iargs[0]);
        std::vector<std::int64_t> normalized_shape;
        for (std::int64_t i = 0; i < n; ++i)
        {
            normalized_shape.push_back(
                input.size(input.dim() - n + i));
        }
        if (mean.dim() + n == input.dim())
        {
            std::vector<std::int64_t> stats = mean.sizes().vec();
            for (std::int64_t i = 0; i < n; ++i)
            {
                stats.push_back(1);
            }
            mean = mean.reshape(stats);
            rstd = rstd.reshape(stats);
        }
        at::Tensor grad_input;
        at::Tensor grad_weight;
        at::Tensor grad_bias;
        if (need_gi)
        {
            grad_input = out_fp32(gi_ptr, *args, 0);
        }
        if (need_gw)
        {
            grad_weight = out_fp32(gw_ptr, *args, 1);
        }
        if (need_gb)
        {
            grad_bias = out_fp32(gb_ptr, *args, 2);
        }
        std::array<bool, 3> output_mask = {
            need_gi,
            need_gw,
            need_gb};
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        // ATen empty_like(bias/weight) requires defined affine tensors when
        // the corresponding output_mask bit is set.
        const bool use_out = need_gi && need_gw && need_gb && has_w
            && has_b;
        if (use_out)
        {
            at::native_layer_norm_backward_out(
                grad_input,
                grad_weight,
                grad_bias,
                grad_out,
                input,
                normalized_shape,
                mean,
                rstd,
                weight,
                bias,
                output_mask);
        }
        else
        {
            auto grads = at::native_layer_norm_backward(
                grad_out,
                input,
                normalized_shape,
                mean,
                rstd,
                has_w ? c10::optional<at::Tensor>(weight)
                    : c10::nullopt,
                has_b ? c10::optional<at::Tensor>(bias)
                    : c10::nullopt,
                output_mask);
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
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_layer_norm_backward failed: %s\n",
            ex.what());
        std::abort();
    }
#endif
}


#ifdef NNTILE_USE_CUDA
void TorchLayerNormBackward::cuda(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        // StarPU stream + cuBLAS; TLS blob device = CUDA.
        TorchCudaEnv cuda_env;
        (void)cuda_env;
        cpu(buffers, cl_args);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_layer_norm_backward CUDA failed: %s\n",
            ex.what());
        std::abort();
    }
#endif
}
#endif // NNTILE_USE_CUDA

uint32_t TorchLayerNormBackward::footprint(struct starpu_task *task)
{
    return args_footprint(
        reinterpret_cast<args_t *>(task->cl_arg));
}

void TorchLayerNormBackward::submit(
    int starpu_worker_hint,
    const args_t &meta,
    Handle grad_out,
    Handle input,
    Handle mean,
    Handle rstd,
    Handle weight,
    Handle bias,
    Handle grad_input,
    Handle grad_weight,
    Handle grad_bias,
    bool has_weight,
    bool has_bias,
    bool need_grad_input,
    bool need_grad_weight,
    bool need_grad_bias
)
{
    args_t *args = clone_args(meta);
    args->kind = TorchKind::NativeLayerNormBackward;
    args->iargs[1] = has_weight ? 1 : 0;
    args->iargs[2] = has_bias ? 1 : 0;
    args->iargs[3] = need_grad_input ? 1 : 0;
    args->iargs[4] = need_grad_weight ? 1 : 0;
    args->iargs[5] = need_grad_bias ? 1 : 0;
    // Build task with a fixed set of common cases.
    std::vector<std::pair<starpu_data_access_mode, Handle>> handles;
    handles.push_back({STARPU_R, grad_out});
    handles.push_back({STARPU_R, input});
    handles.push_back({STARPU_R, mean});
    handles.push_back({STARPU_R, rstd});
    if (need_grad_input)
    {
        handles.push_back({STARPU_W, grad_input});
    }
    if (need_grad_weight)
    {
        handles.push_back({STARPU_W, grad_weight});
    }
    if (need_grad_bias)
    {
        handles.push_back({STARPU_W, grad_bias});
    }
    if (has_weight)
    {
        handles.push_back({STARPU_R, weight});
    }
    if (has_bias)
    {
        handles.push_back({STARPU_R, bias});
    }
    // Use starpu_task_insert via packed helper for up to 9 handles.
    starpu_data_handle_t h[9];
    enum starpu_data_access_mode m[9];
    const Index n = static_cast<Index>(handles.size());
    if (n > 9)
    {
        std::free(args);
        throw std::runtime_error(
            "torch_layer_norm_backward.submit: too many handles");
    }
    for (Index i = 0; i < n; ++i)
    {
        m[static_cast<size_t>(i)] = handles[static_cast<size_t>(i)].first;
        h[static_cast<size_t>(i)] =
            handles[static_cast<size_t>(i)].second.get();
    }
    int ret = 0;
    switch (n)
    {
    case 4:
        ret = nntile_starpu_task_insert(
            &codelet,
            starpu_worker_hint,
            m[0],
            h[0],
            m[1],
            h[1],
            m[2],
            h[2],
            m[3],
            h[3],
            STARPU_CL_ARGS,
            args,
            sizeof(*args),
            0);
        break;
    case 5:
        ret = nntile_starpu_task_insert(
            &codelet,
            starpu_worker_hint,
            m[0],
            h[0],
            m[1],
            h[1],
            m[2],
            h[2],
            m[3],
            h[3],
            m[4],
            h[4],
            STARPU_CL_ARGS,
            args,
            sizeof(*args),
            0);
        break;
    case 6:
        ret = nntile_starpu_task_insert(
            &codelet,
            starpu_worker_hint,
            m[0],
            h[0],
            m[1],
            h[1],
            m[2],
            h[2],
            m[3],
            h[3],
            m[4],
            h[4],
            m[5],
            h[5],
            STARPU_CL_ARGS,
            args,
            sizeof(*args),
            0);
        break;
    case 7:
        ret = nntile_starpu_task_insert(
            &codelet,
            starpu_worker_hint,
            m[0],
            h[0],
            m[1],
            h[1],
            m[2],
            h[2],
            m[3],
            h[3],
            m[4],
            h[4],
            m[5],
            h[5],
            m[6],
            h[6],
            STARPU_CL_ARGS,
            args,
            sizeof(*args),
            0);
        break;
    case 8:
        ret = nntile_starpu_task_insert(
            &codelet,
            starpu_worker_hint,
            m[0],
            h[0],
            m[1],
            h[1],
            m[2],
            h[2],
            m[3],
            h[3],
            m[4],
            h[4],
            m[5],
            h[5],
            m[6],
            h[6],
            m[7],
            h[7],
            STARPU_CL_ARGS,
            args,
            sizeof(*args),
            0);
        break;
    case 9:
        ret = nntile_starpu_task_insert(
            &codelet,
            starpu_worker_hint,
            m[0],
            h[0],
            m[1],
            h[1],
            m[2],
            h[2],
            m[3],
            h[3],
            m[4],
            h[4],
            m[5],
            h[5],
            m[6],
            h[6],
            m[7],
            h[7],
            m[8],
            h[8],
            STARPU_CL_ARGS,
            args,
            sizeof(*args),
            0);
        break;
    default:
        std::free(args);
        throw std::runtime_error(
            "torch_layer_norm_backward.submit: bad handle count");
    }
    if (ret != 0)
    {
        throw std::runtime_error(
            "torch_layer_norm_backward.submit failed");
    }
}

TorchEmbeddingDenseBackward::TorchEmbeddingDenseBackward():
    codelet(
        "nntile_torch_embedding_dense_backward",
        footprint,
        cpu_funcs,
        cuda_funcs)
{
}

void TorchEmbeddingDenseBackward::cpu(
    void *buffers[],
    void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        auto *args = reinterpret_cast<args_t *>(cl_args);
        auto **ifaces =
            reinterpret_cast<VariableInterface **>(buffers);
        float *grad_ptr = ifaces[0]->get_ptr<float>();
        auto *idx_ptr = ifaces[1]->get_ptr<std::int64_t>();
        float *gw_ptr = ifaces[2]->get_ptr<float>();
        at::Tensor grad = in_fp32(grad_ptr, *args, 0);
        at::Tensor indices = in_i64(idx_ptr, *args, 1);
        at::Tensor grad_weight = out_fp32(gw_ptr, *args, 0);
        const std::int64_t num_weights =
            static_cast<std::int64_t>(args->iargs[0]);
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        at::embedding_dense_backward_out(
            grad_weight,
            grad,
            indices,
            num_weights,
            /*padding_idx=*/-1,
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
#endif
}


#ifdef NNTILE_USE_CUDA
void TorchEmbeddingDenseBackward::cuda(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        // StarPU stream + cuBLAS; TLS blob device = CUDA.
        TorchCudaEnv cuda_env;
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
#endif
}
#endif // NNTILE_USE_CUDA

uint32_t TorchEmbeddingDenseBackward::footprint(
    struct starpu_task *task)
{
    return args_footprint(
        reinterpret_cast<args_t *>(task->cl_arg));
}

void TorchEmbeddingDenseBackward::submit(
    int starpu_worker_hint,
    const args_t &meta,
    Handle grad,
    Handle indices,
    Handle grad_weight
)
{
    args_t *args = clone_args(meta);
    args->kind = TorchKind::EmbeddingDenseBackward;
    int ret = nntile_starpu_task_insert(
        &codelet,
        starpu_worker_hint,
        STARPU_R,
        grad.get(),
        STARPU_R,
        indices.get(),
        STARPU_CL_ARGS,
        args,
        sizeof(*args),
        STARPU_W,
        grad_weight.get(),
        0);
    if (ret != 0)
    {
        throw std::runtime_error(
            "torch_embedding_dense_backward.submit failed");
    }
}

TorchConvolution::TorchConvolution():
    codelet("nntile_torch_convolution", footprint, cpu_funcs, cuda_funcs)
{
}

void TorchConvolution::cpu(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        auto *args = reinterpret_cast<args_t *>(cl_args);
        auto **ifaces =
            reinterpret_cast<VariableInterface **>(buffers);
        const bool has_bias = args->iargs[11] != 0;
        Index buf = 0;
        at::Tensor input = in_fp32(ifaces[buf++]->get_ptr<float>(), *args, 0);
        at::Tensor weight = in_fp32(ifaces[buf++]->get_ptr<float>(), *args, 1);
        at::Tensor bias;
        if (has_bias)
        {
            bias = in_fp32(ifaces[buf++]->get_ptr<float>(), *args, 2);
        }
        at::Tensor out = out_fp32(ifaces[buf++]->get_ptr<float>(), *args, 0);
        const Index ndim = args->iargs[0];
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        at::Tensor result = at::convolution(
            input,
            weight,
            has_bias ? c10::optional<at::Tensor>(bias) : c10::nullopt,
            iarg_vec(*args, 3, ndim),
            iarg_vec(*args, 5, ndim),
            iarg_vec(*args, 7, ndim),
            args->iargs[2] != 0,
            iarg_vec(*args, 9, ndim),
            static_cast<std::int64_t>(args->iargs[1]));
        out.copy_(result);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_convolution failed: %s\n",
            ex.what());
        std::abort();
    }
#endif
}

#ifdef NNTILE_USE_CUDA
void TorchConvolution::cuda(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        TorchCudaEnv cuda_env;
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
#endif
}
#endif // NNTILE_USE_CUDA

uint32_t TorchConvolution::footprint(struct starpu_task *task)
{
    return args_footprint(reinterpret_cast<args_t *>(task->cl_arg));
}

void TorchConvolution::submit(
    int starpu_worker_hint,
    const args_t &meta,
    Handle input,
    Handle weight,
    Handle bias,
    Handle out,
    bool has_bias
)
{
    args_t *args = clone_args(meta);
    args->kind = TorchKind::Convolution;
    args->iargs[11] = has_bias ? 1 : 0;
    std::vector<std::pair<enum starpu_data_access_mode, Handle>> handles;
    handles.push_back({STARPU_R, input});
    handles.push_back({STARPU_R, weight});
    if (has_bias)
    {
        handles.push_back({STARPU_R, bias});
    }
    handles.push_back({STARPU_W, out});
    if (submit_accesses(&codelet, starpu_worker_hint, args, handles) != 0)
    {
        throw std::runtime_error("torch_convolution.submit failed");
    }
}

TorchConvolutionBackward::TorchConvolutionBackward():
    codelet(
        "nntile_torch_convolution_backward",
        footprint,
        cpu_funcs,
        cuda_funcs)
{
}

void TorchConvolutionBackward::cpu(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        auto *args = reinterpret_cast<args_t *>(cl_args);
        auto **ifaces =
            reinterpret_cast<VariableInterface **>(buffers);
        const bool need_gi = args->iargs[12] != 0;
        const bool need_gw = args->iargs[13] != 0;
        const bool need_gb = args->iargs[14] != 0;
        Index buf = 0;
        at::Tensor grad_out =
            in_fp32(ifaces[buf++]->get_ptr<float>(), *args, 0);
        at::Tensor input = in_fp32(ifaces[buf++]->get_ptr<float>(), *args, 1);
        at::Tensor weight = in_fp32(ifaces[buf++]->get_ptr<float>(), *args, 2);
        at::Tensor grad_input;
        at::Tensor grad_weight;
        at::Tensor grad_bias;
        if (need_gi)
        {
            grad_input = out_fp32(ifaces[buf++]->get_ptr<float>(), *args, 0);
        }
        if (need_gw)
        {
            grad_weight = out_fp32(ifaces[buf++]->get_ptr<float>(), *args, 1);
        }
        if (need_gb)
        {
            grad_bias = out_fp32(ifaces[buf++]->get_ptr<float>(), *args, 2);
        }
        const Index ndim = args->iargs[0];
        std::vector<std::int64_t> bias_sizes_vec;
        at::OptionalIntArrayRef bias_sizes = c10::nullopt;
        if (need_gb)
        {
            bias_sizes_vec = sizes_of(*args, 2, true);
            bias_sizes = at::IntArrayRef(bias_sizes_vec);
        }
        std::array<bool, 3> output_mask = {need_gi, need_gw, need_gb};
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
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
            output_mask);
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
#endif
}

#ifdef NNTILE_USE_CUDA
void TorchConvolutionBackward::cuda(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        TorchCudaEnv cuda_env;
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
#endif
}
#endif // NNTILE_USE_CUDA

uint32_t TorchConvolutionBackward::footprint(struct starpu_task *task)
{
    return args_footprint(reinterpret_cast<args_t *>(task->cl_arg));
}

void TorchConvolutionBackward::submit(
    int starpu_worker_hint,
    const args_t &meta,
    Handle grad_out,
    Handle input,
    Handle weight,
    Handle grad_input,
    Handle grad_weight,
    Handle grad_bias,
    bool need_grad_input,
    bool need_grad_weight,
    bool need_grad_bias
)
{
    args_t *args = clone_args(meta);
    args->kind = TorchKind::ConvolutionBackward;
    args->iargs[12] = need_grad_input ? 1 : 0;
    args->iargs[13] = need_grad_weight ? 1 : 0;
    args->iargs[14] = need_grad_bias ? 1 : 0;
    std::vector<std::pair<enum starpu_data_access_mode, Handle>> handles;
    handles.push_back({STARPU_R, grad_out});
    handles.push_back({STARPU_R, input});
    handles.push_back({STARPU_R, weight});
    if (need_grad_input)
    {
        handles.push_back({STARPU_W, grad_input});
    }
    if (need_grad_weight)
    {
        handles.push_back({STARPU_W, grad_weight});
    }
    if (need_grad_bias)
    {
        handles.push_back({STARPU_W, grad_bias});
    }
    if (submit_accesses(&codelet, starpu_worker_hint, args, handles) != 0)
    {
        throw std::runtime_error("torch_convolution_backward.submit failed");
    }
}

TorchMaxPool2dWithIndices::TorchMaxPool2dWithIndices():
    codelet(
        "nntile_torch_max_pool2d_with_indices",
        footprint,
        cpu_funcs,
        cuda_funcs)
{
}

void TorchMaxPool2dWithIndices::cpu(
    void *buffers[],
    void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        auto *args = reinterpret_cast<args_t *>(cl_args);
        auto **ifaces =
            reinterpret_cast<VariableInterface **>(buffers);
        at::Tensor input = in_fp32(ifaces[0]->get_ptr<float>(), *args, 0);
        at::Tensor out = out_fp32(ifaces[1]->get_ptr<float>(), *args, 0);
        at::Tensor indices =
            out_i64(ifaces[2]->get_ptr<std::int64_t>(), *args, 1);
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
#endif
}

#ifdef NNTILE_USE_CUDA
void TorchMaxPool2dWithIndices::cuda(
    void *buffers[],
    void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        TorchCudaEnv cuda_env;
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
#endif
}
#endif // NNTILE_USE_CUDA

uint32_t TorchMaxPool2dWithIndices::footprint(struct starpu_task *task)
{
    return args_footprint(reinterpret_cast<args_t *>(task->cl_arg));
}

void TorchMaxPool2dWithIndices::submit(
    int starpu_worker_hint,
    const args_t &meta,
    Handle input,
    Handle out,
    Handle indices
)
{
    args_t *args = clone_args(meta);
    args->kind = TorchKind::MaxPool2dWithIndices;
    std::vector<std::pair<enum starpu_data_access_mode, Handle>> handles = {
        {STARPU_R, input},
        {STARPU_W, out},
        {STARPU_W, indices}};
    if (submit_accesses(&codelet, starpu_worker_hint, args, handles) != 0)
    {
        throw std::runtime_error(
            "torch_max_pool2d_with_indices.submit failed");
    }
}

TorchMaxPool2dWithIndicesBackward::TorchMaxPool2dWithIndicesBackward():
    codelet(
        "nntile_torch_max_pool2d_with_indices_backward",
        footprint,
        cpu_funcs,
        cuda_funcs)
{
}

void TorchMaxPool2dWithIndicesBackward::cpu(
    void *buffers[],
    void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        auto *args = reinterpret_cast<args_t *>(cl_args);
        auto **ifaces =
            reinterpret_cast<VariableInterface **>(buffers);
        at::Tensor grad_out =
            in_fp32(ifaces[0]->get_ptr<float>(), *args, 0);
        at::Tensor input = in_fp32(ifaces[1]->get_ptr<float>(), *args, 1);
        at::Tensor indices =
            in_i64(ifaces[2]->get_ptr<std::int64_t>(), *args, 2);
        at::Tensor grad_input =
            out_fp32(ifaces[3]->get_ptr<float>(), *args, 0);
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
#endif
}

#ifdef NNTILE_USE_CUDA
void TorchMaxPool2dWithIndicesBackward::cuda(
    void *buffers[],
    void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        TorchCudaEnv cuda_env;
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
#endif
}
#endif // NNTILE_USE_CUDA

uint32_t TorchMaxPool2dWithIndicesBackward::footprint(
    struct starpu_task *task)
{
    return args_footprint(reinterpret_cast<args_t *>(task->cl_arg));
}

void TorchMaxPool2dWithIndicesBackward::submit(
    int starpu_worker_hint,
    const args_t &meta,
    Handle grad_out,
    Handle input,
    Handle indices,
    Handle grad_input
)
{
    args_t *args = clone_args(meta);
    args->kind = TorchKind::MaxPool2dWithIndicesBackward;
    std::vector<std::pair<enum starpu_data_access_mode, Handle>> handles = {
        {STARPU_R, grad_out},
        {STARPU_R, input},
        {STARPU_R, indices},
        {STARPU_W, grad_input}};
    if (submit_accesses(&codelet, starpu_worker_hint, args, handles) != 0)
    {
        throw std::runtime_error(
            "torch_max_pool2d_with_indices_backward.submit failed");
    }
}

TorchNativeBatchNorm::TorchNativeBatchNorm():
    codelet(
        "nntile_torch_native_batch_norm",
        footprint,
        cpu_funcs,
        cuda_funcs)
{
}

void TorchNativeBatchNorm::cpu(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        auto *args = reinterpret_cast<args_t *>(cl_args);
        auto **ifaces =
            reinterpret_cast<VariableInterface **>(buffers);
        const bool training = args->iargs[0] != 0;
        const bool has_w = args->iargs[1] != 0;
        const bool has_b = args->iargs[2] != 0;
        const bool has_rm = args->iargs[3] != 0;
        const bool has_rv = args->iargs[4] != 0;
        Index buf = 0;
        at::Tensor input = in_fp32(ifaces[buf++]->get_ptr<float>(), *args, 0);
        at::Tensor out = out_fp32(ifaces[buf++]->get_ptr<float>(), *args, 0);
        at::Tensor save_mean =
            out_fp32(ifaces[buf++]->get_ptr<float>(), *args, 1);
        at::Tensor save_invstd =
            out_fp32(ifaces[buf++]->get_ptr<float>(), *args, 2);
        at::Tensor weight;
        at::Tensor bias;
        at::Tensor running_mean;
        at::Tensor running_var;
        if (has_w)
        {
            weight = in_fp32(ifaces[buf++]->get_ptr<float>(), *args, 1);
        }
        if (has_b)
        {
            bias = in_fp32(ifaces[buf++]->get_ptr<float>(), *args, 2);
        }
        if (has_rm)
        {
            running_mean =
                in_fp32(ifaces[buf++]->get_ptr<float>(), *args, 3);
        }
        if (has_rv)
        {
            running_var =
                in_fp32(ifaces[buf++]->get_ptr<float>(), *args, 4);
        }
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        auto result = at::native_batch_norm(
            input,
            has_w ? c10::optional<at::Tensor>(weight) : c10::nullopt,
            has_b ? c10::optional<at::Tensor>(bias) : c10::nullopt,
            has_rm ? c10::optional<at::Tensor>(running_mean) : c10::nullopt,
            has_rv ? c10::optional<at::Tensor>(running_var) : c10::nullopt,
            training,
            static_cast<double>(args->scalars[0]),
            static_cast<double>(args->scalars[1]));
        out.copy_(std::get<0>(result));
        save_mean.copy_(std::get<1>(result).reshape(save_mean.sizes()));
        save_invstd.copy_(std::get<2>(result).reshape(save_invstd.sizes()));
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_native_batch_norm failed: %s\n",
            ex.what());
        std::abort();
    }
#endif
}

#ifdef NNTILE_USE_CUDA
void TorchNativeBatchNorm::cuda(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        TorchCudaEnv cuda_env;
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
#endif
}
#endif // NNTILE_USE_CUDA

uint32_t TorchNativeBatchNorm::footprint(struct starpu_task *task)
{
    return args_footprint(reinterpret_cast<args_t *>(task->cl_arg));
}

void TorchNativeBatchNorm::submit(
    int starpu_worker_hint,
    const args_t &meta,
    Handle input,
    Handle weight,
    Handle bias,
    Handle running_mean,
    Handle running_var,
    Handle out,
    Handle save_mean,
    Handle save_invstd,
    bool has_weight,
    bool has_bias,
    bool has_running_mean,
    bool has_running_var,
    bool training
)
{
    args_t *args = clone_args(meta);
    args->kind = TorchKind::NativeBatchNorm;
    args->iargs[0] = training ? 1 : 0;
    args->iargs[1] = has_weight ? 1 : 0;
    args->iargs[2] = has_bias ? 1 : 0;
    args->iargs[3] = has_running_mean ? 1 : 0;
    args->iargs[4] = has_running_var ? 1 : 0;
    std::vector<std::pair<enum starpu_data_access_mode, Handle>> handles;
    handles.push_back({STARPU_R, input});
    handles.push_back({STARPU_W, out});
    handles.push_back({STARPU_W, save_mean});
    handles.push_back({STARPU_W, save_invstd});
    if (has_weight)
    {
        handles.push_back({STARPU_R, weight});
    }
    if (has_bias)
    {
        handles.push_back({STARPU_R, bias});
    }
    if (has_running_mean)
    {
        handles.push_back({training ? STARPU_RW : STARPU_R, running_mean});
    }
    if (has_running_var)
    {
        handles.push_back({training ? STARPU_RW : STARPU_R, running_var});
    }
    if (submit_accesses(&codelet, starpu_worker_hint, args, handles) != 0)
    {
        throw std::runtime_error("torch_native_batch_norm.submit failed");
    }
}

TorchNativeBatchNormBackward::TorchNativeBatchNormBackward():
    codelet(
        "nntile_torch_native_batch_norm_backward",
        footprint,
        cpu_funcs,
        cuda_funcs)
{
}

void TorchNativeBatchNormBackward::cpu(
    void *buffers[],
    void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        auto *args = reinterpret_cast<args_t *>(cl_args);
        auto **ifaces =
            reinterpret_cast<VariableInterface **>(buffers);
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
            in_fp32(ifaces[buf++]->get_ptr<float>(), *args, 0);
        at::Tensor input = in_fp32(ifaces[buf++]->get_ptr<float>(), *args, 1);
        at::Tensor weight;
        at::Tensor running_mean;
        at::Tensor running_var;
        at::Tensor save_mean;
        at::Tensor save_invstd;
        if (has_w)
        {
            weight = in_fp32(ifaces[buf++]->get_ptr<float>(), *args, 2);
        }
        if (has_rm)
        {
            running_mean =
                in_fp32(ifaces[buf++]->get_ptr<float>(), *args, 3);
        }
        if (has_rv)
        {
            running_var =
                in_fp32(ifaces[buf++]->get_ptr<float>(), *args, 4);
        }
        if (has_sm)
        {
            save_mean = in_fp32(ifaces[buf++]->get_ptr<float>(), *args, 5);
        }
        if (has_si)
        {
            save_invstd =
                in_fp32(ifaces[buf++]->get_ptr<float>(), *args, 6);
        }
        at::Tensor grad_input;
        at::Tensor grad_weight;
        at::Tensor grad_bias;
        if (need_gi)
        {
            grad_input = out_fp32(ifaces[buf++]->get_ptr<float>(), *args, 0);
        }
        if (need_gw)
        {
            grad_weight = out_fp32(ifaces[buf++]->get_ptr<float>(), *args, 1);
        }
        if (need_gb)
        {
            grad_bias = out_fp32(ifaces[buf++]->get_ptr<float>(), *args, 2);
        }
        std::array<bool, 3> output_mask = {need_gi, need_gw, need_gb};
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        auto grads = at::native_batch_norm_backward(
            grad_out,
            input,
            has_w ? c10::optional<at::Tensor>(weight) : c10::nullopt,
            has_rm ? c10::optional<at::Tensor>(running_mean) : c10::nullopt,
            has_rv ? c10::optional<at::Tensor>(running_var) : c10::nullopt,
            has_sm ? c10::optional<at::Tensor>(save_mean) : c10::nullopt,
            has_si ? c10::optional<at::Tensor>(save_invstd) : c10::nullopt,
            training,
            static_cast<double>(args->scalars[1]),
            output_mask);
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
            "nntile_torch_native_batch_norm_backward failed: %s\n",
            ex.what());
        std::abort();
    }
#endif
}

#ifdef NNTILE_USE_CUDA
void TorchNativeBatchNormBackward::cuda(
    void *buffers[],
    void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        TorchCudaEnv cuda_env;
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
#endif
}
#endif // NNTILE_USE_CUDA

uint32_t TorchNativeBatchNormBackward::footprint(
    struct starpu_task *task)
{
    return args_footprint(reinterpret_cast<args_t *>(task->cl_arg));
}

void TorchNativeBatchNormBackward::submit(
    int starpu_worker_hint,
    const args_t &meta,
    Handle grad_out,
    Handle input,
    Handle weight,
    Handle running_mean,
    Handle running_var,
    Handle save_mean,
    Handle save_invstd,
    Handle grad_input,
    Handle grad_weight,
    Handle grad_bias,
    bool has_weight,
    bool has_running_mean,
    bool has_running_var,
    bool has_save_mean,
    bool has_save_invstd,
    bool need_grad_input,
    bool need_grad_weight,
    bool need_grad_bias
)
{
    args_t *args = clone_args(meta);
    args->kind = TorchKind::NativeBatchNormBackward;
    args->iargs[1] = has_weight ? 1 : 0;
    args->iargs[3] = has_running_mean ? 1 : 0;
    args->iargs[4] = has_running_var ? 1 : 0;
    args->iargs[5] = has_save_mean ? 1 : 0;
    args->iargs[6] = has_save_invstd ? 1 : 0;
    args->iargs[7] = need_grad_input ? 1 : 0;
    args->iargs[8] = need_grad_weight ? 1 : 0;
    args->iargs[9] = need_grad_bias ? 1 : 0;
    std::vector<std::pair<enum starpu_data_access_mode, Handle>> handles;
    handles.push_back({STARPU_R, grad_out});
    handles.push_back({STARPU_R, input});
    if (has_weight)
    {
        handles.push_back({STARPU_R, weight});
    }
    if (has_running_mean)
    {
        handles.push_back({STARPU_R, running_mean});
    }
    if (has_running_var)
    {
        handles.push_back({STARPU_R, running_var});
    }
    if (has_save_mean)
    {
        handles.push_back({STARPU_R, save_mean});
    }
    if (has_save_invstd)
    {
        handles.push_back({STARPU_R, save_invstd});
    }
    if (need_grad_input)
    {
        handles.push_back({STARPU_W, grad_input});
    }
    if (need_grad_weight)
    {
        handles.push_back({STARPU_W, grad_weight});
    }
    if (need_grad_bias)
    {
        handles.push_back({STARPU_W, grad_bias});
    }
    if (submit_accesses(&codelet, starpu_worker_hint, args, handles) != 0)
    {
        throw std::runtime_error(
            "torch_native_batch_norm_backward.submit failed");
    }
}

TorchSdpaBackward::TorchSdpaBackward():
    codelet(
        "nntile_torch_sdpa_backward",
        footprint,
        cpu_funcs,
        cuda_funcs)
{
}

void TorchSdpaBackward::cpu(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        auto *args = reinterpret_cast<args_t *>(cl_args);
        auto **ifaces =
            reinterpret_cast<VariableInterface **>(buffers);
        const bool has_mask = args->iargs[0] != 0;
        const bool is_causal = args->iargs[1] != 0;
        float *q_ptr = ifaces[0]->get_ptr<float>();
        float *k_ptr = ifaces[1]->get_ptr<float>();
        float *v_ptr = ifaces[2]->get_ptr<float>();
        float *go_ptr = ifaces[3]->get_ptr<float>();
        Index buf = 4;
        at::Tensor mask_bool;
        if (has_mask)
        {
            bool_t *m_ptr = ifaces[buf++]->get_ptr<bool_t>();
            mask_bool = blob_bool(
                reinterpret_cast<bool *>(m_ptr),
                sizes_of(*args, 4, false),
                strides_of(*args, 4, false),
                static_cast<std::int64_t>(args->in_offset[4]));
        }
        float *gq_ptr = ifaces[buf++]->get_ptr<float>();
        float *gk_ptr = ifaces[buf++]->get_ptr<float>();
        float *gv_ptr = ifaces[buf++]->get_ptr<float>();
        at::Tensor q = in_fp32(q_ptr, *args, 0);
        at::Tensor k = in_fp32(k_ptr, *args, 1);
        at::Tensor v = in_fp32(v_ptr, *args, 2);
        at::Tensor grad_out = in_fp32(go_ptr, *args, 3);
        at::Tensor grad_q = out_fp32(gq_ptr, *args, 0);
        at::Tensor grad_k = out_fp32(gk_ptr, *args, 1);
        at::Tensor grad_v = out_fp32(gv_ptr, *args, 2);
        c10::optional<at::Tensor> attn_mask = c10::nullopt;
        if (has_mask)
        {
            // Bool keep-mask → additive float mask (flash CPU).
            at::Tensor float_mask = at::zeros(
                mask_bool.sizes(),
                q.options());
            float_mask.masked_fill_(
                ~mask_bool,
                -std::numeric_limits<float>::infinity());
            attn_mask = float_mask;
        }
        at::AutoDispatchBelowADInplaceOrView guard;
        at::NoGradGuard no_grad;
        // flash_*_for_cpu is CPU-only; CUDA uses mem-efficient
        // (math fallback). Prefer is_causal over dense SxS masks.
#ifdef NNTILE_USE_CUDA
        if (q.is_cuda())
        {
            run_sdpa_cuda_backward(
                q,
                k,
                v,
                grad_out,
                attn_mask,
                is_causal,
                grad_q,
                grad_k,
                grad_v);
        }
        else
#endif
        {
            auto fwd =
                at::_scaled_dot_product_flash_attention_for_cpu(
                    q,
                    k,
                    v,
                    /*dropout_p=*/0.0,
                    is_causal,
                    attn_mask,
                    /*scale=*/c10::nullopt);
            auto bwd = at::
                _scaled_dot_product_flash_attention_for_cpu_backward(
                    grad_out,
                    q,
                    k,
                    v,
                    std::get<0>(fwd),
                    std::get<1>(fwd),
                    /*dropout_p=*/0.0,
                    is_causal,
                    attn_mask,
                    /*scale=*/c10::nullopt);
            grad_q.copy_(std::get<0>(bwd));
            grad_k.copy_(std::get<1>(bwd));
            grad_v.copy_(std::get<2>(bwd));
        }
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_sdpa_backward failed: %s\n",
            ex.what());
        std::abort();
    }
#endif
}


#ifdef NNTILE_USE_CUDA
void TorchSdpaBackward::cuda(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        // StarPU stream + cuBLAS; TLS blob device = CUDA.
        TorchCudaEnv cuda_env;
        (void)cuda_env;
        cpu(buffers, cl_args);
    }
    catch (const std::exception &ex)
    {
        std::fprintf(
            stderr,
            "nntile_torch_sdpa_backward CUDA failed: %s\n",
            ex.what());
        std::abort();
    }
#endif
}
#endif // NNTILE_USE_CUDA

uint32_t TorchSdpaBackward::footprint(struct starpu_task *task)
{
    return args_footprint(
        reinterpret_cast<args_t *>(task->cl_arg));
}

void TorchSdpaBackward::submit(
    int starpu_worker_hint,
    const args_t &meta,
    Handle q,
    Handle k,
    Handle v,
    Handle grad_out,
    Handle mask,
    Handle grad_q,
    Handle grad_k,
    Handle grad_v,
    bool has_mask
)
{
    args_t *args = clone_args(meta);
    args->kind = TorchKind::SdpaBackward;
    args->iargs[0] = has_mask ? 1 : 0;
    int ret = 0;
    if (has_mask)
    {
        ret = nntile_starpu_task_insert(
            &codelet,
            starpu_worker_hint,
            STARPU_R,
            q.get(),
            STARPU_R,
            k.get(),
            STARPU_R,
            v.get(),
            STARPU_R,
            grad_out.get(),
            STARPU_R,
            mask.get(),
            STARPU_CL_ARGS,
            args,
            sizeof(*args),
            STARPU_W,
            grad_q.get(),
            STARPU_W,
            grad_k.get(),
            STARPU_W,
            grad_v.get(),
            0);
    }
    else
    {
        ret = nntile_starpu_task_insert(
            &codelet,
            starpu_worker_hint,
            STARPU_R,
            q.get(),
            STARPU_R,
            k.get(),
            STARPU_R,
            v.get(),
            STARPU_R,
            grad_out.get(),
            STARPU_CL_ARGS,
            args,
            sizeof(*args),
            STARPU_W,
            grad_q.get(),
            STARPU_W,
            grad_k.get(),
            STARPU_W,
            grad_v.get(),
            0);
    }
    if (ret != 0)
    {
        throw std::runtime_error("torch_sdpa_backward.submit failed");
    }
}

TorchNllLossForward::TorchNllLossForward():
    codelet(
        "nntile_torch_nll_loss_forward",
        footprint,
        cpu_funcs,
        cuda_funcs)
{
}

void TorchNllLossForward::cpu(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        auto *args = reinterpret_cast<args_t *>(cl_args);
        auto **ifaces =
            reinterpret_cast<VariableInterface **>(buffers);
        float *lp_ptr = ifaces[0]->get_ptr<float>();
        auto *tgt_ptr = ifaces[1]->get_ptr<std::int64_t>();
        float *loss_ptr = ifaces[2]->get_ptr<float>();
        float *tw_ptr = ifaces[3]->get_ptr<float>();
        at::Tensor log_probs = in_fp32(lp_ptr, *args, 0);
        at::Tensor target = in_i64(tgt_ptr, *args, 1);
        at::Tensor loss = out_fp32(loss_ptr, *args, 0);
        at::Tensor total_weight = out_fp32(tw_ptr, *args, 1);
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
#endif
}


#ifdef NNTILE_USE_CUDA
void TorchNllLossForward::cuda(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        // StarPU stream + cuBLAS; TLS blob device = CUDA.
        TorchCudaEnv cuda_env;
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
#endif
}
#endif // NNTILE_USE_CUDA

uint32_t TorchNllLossForward::footprint(struct starpu_task *task)
{
    return args_footprint(
        reinterpret_cast<args_t *>(task->cl_arg));
}

void TorchNllLossForward::submit(
    int starpu_worker_hint,
    const args_t &meta,
    Handle log_probs,
    Handle target,
    Handle loss,
    Handle total_weight
)
{
    args_t *args = clone_args(meta);
    args->kind = TorchKind::NllLossForward;
    int ret = nntile_starpu_task_insert(
        &codelet,
        starpu_worker_hint,
        STARPU_R,
        log_probs.get(),
        STARPU_R,
        target.get(),
        STARPU_CL_ARGS,
        args,
        sizeof(*args),
        STARPU_W,
        loss.get(),
        STARPU_W,
        total_weight.get(),
        0);
    if (ret != 0)
    {
        throw std::runtime_error(
            "torch_nll_loss_forward.submit failed");
    }
}

TorchNllLossBackward::TorchNllLossBackward():
    codelet(
        "nntile_torch_nll_loss_backward",
        footprint,
        cpu_funcs,
        cuda_funcs)
{
}

void TorchNllLossBackward::cpu(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        auto *args = reinterpret_cast<args_t *>(cl_args);
        auto **ifaces =
            reinterpret_cast<VariableInterface **>(buffers);
        float *go_ptr = ifaces[0]->get_ptr<float>();
        float *lp_ptr = ifaces[1]->get_ptr<float>();
        auto *tgt_ptr = ifaces[2]->get_ptr<std::int64_t>();
        float *tw_ptr = ifaces[3]->get_ptr<float>();
        float *gi_ptr = ifaces[4]->get_ptr<float>();
        at::Tensor grad_output = in_fp32(go_ptr, *args, 0);
        at::Tensor log_probs = in_fp32(lp_ptr, *args, 1);
        at::Tensor target = in_i64(tgt_ptr, *args, 2);
        at::Tensor total_weight = in_fp32(tw_ptr, *args, 3);
        at::Tensor grad_input = out_fp32(gi_ptr, *args, 0);
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
#endif
}


#ifdef NNTILE_USE_CUDA
void TorchNllLossBackward::cuda(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID
    try
    {
        // StarPU stream + cuBLAS; TLS blob device = CUDA.
        TorchCudaEnv cuda_env;
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
#endif
}
#endif // NNTILE_USE_CUDA

uint32_t TorchNllLossBackward::footprint(struct starpu_task *task)
{
    return args_footprint(
        reinterpret_cast<args_t *>(task->cl_arg));
}

void TorchNllLossBackward::submit(
    int starpu_worker_hint,
    const args_t &meta,
    Handle grad_output,
    Handle log_probs,
    Handle target,
    Handle total_weight,
    Handle grad_input
)
{
    args_t *args = clone_args(meta);
    args->kind = TorchKind::NllLossBackward;
    int ret = nntile_starpu_task_insert(
        &codelet,
        starpu_worker_hint,
        STARPU_R,
        grad_output.get(),
        STARPU_R,
        log_probs.get(),
        STARPU_R,
        target.get(),
        STARPU_R,
        total_weight.get(),
        STARPU_CL_ARGS,
        args,
        sizeof(*args),
        STARPU_W,
        grad_input.get(),
        0);
    if (ret != 0)
    {
        throw std::runtime_error(
            "torch_nll_loss_backward.submit failed");
    }
}

torch_unary_pack_t torch_unary;
torch_binary_pack_t torch_binary;
torch_ternary_pack_t torch_ternary;
TorchEmbedding torch_embedding;
TorchEmbeddingDenseBackward torch_embedding_dense_backward;
TorchConvolution torch_convolution;
TorchConvolutionBackward torch_convolution_backward;
TorchMaxPool2dWithIndices torch_max_pool2d_with_indices;
TorchMaxPool2dWithIndicesBackward torch_max_pool2d_with_indices_backward;
TorchNativeBatchNorm torch_native_batch_norm;
TorchNativeBatchNormBackward torch_native_batch_norm_backward;
TorchLayerNorm torch_layer_norm;
TorchLayerNormBackward torch_layer_norm_backward;
TorchSdpaBackward torch_sdpa_backward;
TorchNllLossForward torch_nll_loss_forward;
TorchNllLossBackward torch_nll_loss_backward;
TorchCat torch_cat;
TorchWhere torch_where;
TorchArange torch_arange;
TorchGt torch_gt;
TorchI64Unary torch_i64_unary;
TorchCast torch_cast;

} // namespace nntile::starpu
