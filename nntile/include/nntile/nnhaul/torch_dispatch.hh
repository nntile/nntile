/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file include/nntile/nnhaul/torch_dispatch.hh
 * Family NNHaul codelets for torch-native aten kernels (CPU/CUDA).
 *
 * The graph IR is backend-neutral: TorchKind and TorchDispatchArgs live
 * in nntile/starpu/torch_dispatch.hh and are shared by both builds. The
 * classes here mirror the StarPU families of the same name so
 * nntile/starpu/torch_dispatch.hh can forward to them under
 * NNTILE_USE_NNHAUL; the kernel bodies are the same at::*_out calls,
 * bound to the NNHaul CUDA worker stream by HaulTorchCudaEnv.
 *
 * @version 1.1.0
 */

#pragma once

#include <nntile/defs.h>

#ifndef NNTILE_TORCH_NATIVE_OPS
#error "nntile/nnhaul/torch_dispatch.hh requires NNTILE_TORCH_NATIVE_OPS"
#endif

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <nnhaul/nnhaul.hh>

#include <nntile/base_types.hh>
#include <nntile/core/tile.hh>
#include <nntile/core/torch_args.hh>
#include <nntile/core/torch_meta.hh>
#include <nntile/data_access.hh>
#include <nntile/nnhaul/codelet.hh>

namespace nntile::haul
{

//! The handle type the torch-native core bridge passes. Under NNHaul
//! it wraps a registered ::nnhaul::Handle (any core::Tile converts),
//! carries the StarPU-side MPI stubs the bridge calls, and converts to
//! ::nnhaul::Handle& for the codelet substrate.
class TorchHandle
{
public:
    TorchHandle() : handle_(nullptr)
    {
    }

    TorchHandle(const TorchHandle &) = default;

    template<typename U>
    TorchHandle(const core::Tile<U> &tile) : handle_(&tile.handle())
    {
    }

    int mpi_get_rank() const
    {
        return 0;
    }

    void mpi_transfer(int, int) const
    {
    }

    operator ::nnhaul::Handle &() const
    {
        return *handle_;
    }

    ::nnhaul::Handle &get() const
    {
        return *handle_;
    }

private:
    ::nnhaul::Handle *handle_;
};

//! Footprint over the packed TorchDispatchArgs blob.
std::uint64_t torch_args_footprint(
    void const *cl_args, std::size_t cl_arg_size) noexcept;

//! Shared submit: copy args into the runtime bag (nnhaul::insert takes
//! a synchronous copy, so ``meta`` may die before the task runs) and
//! map the access modes onto nnhaul::insert.
void torch_insert(
    Codelet &codelet,
    int worker,
    starpu::TorchDispatchArgs const &meta,
    std::vector<BufSpec> const &bufs);

//! fp32 unary + reduce family: one R in, one W out.
template<typename T>
class TorchUnary;

template<typename T>
class TorchUnary<std::tuple<T>>
{
public:
    CodeletTyped<T> codelet;
    using args_t = starpu::TorchDispatchArgs;

    TorchUnary();

    static void cpu(void *buffers[], void *cl_args) noexcept;
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
#endif
    static std::uint64_t footprint(
        void const *cl_args, std::size_t cl_arg_size) noexcept
    {
        return torch_args_footprint(cl_args, cl_arg_size);
    }

    void submit(
        int worker_hint,
        starpu::TorchDispatchArgs const &meta,
        TorchHandle const &in,
        TorchHandle const &out);
};

//! fp32 binary family: two R in, one W out.
template<typename T>
class TorchBinary;

template<typename T>
class TorchBinary<std::tuple<T>>
{
public:
    CodeletTyped<T> codelet;
    using args_t = starpu::TorchDispatchArgs;

    TorchBinary();

    static void cpu(void *buffers[], void *cl_args) noexcept;
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
#endif
    static std::uint64_t footprint(
        void const *cl_args, std::size_t cl_arg_size) noexcept
    {
        return torch_args_footprint(cl_args, cl_arg_size);
    }

    void submit(
        int worker_hint,
        starpu::TorchDispatchArgs const &meta,
        TorchHandle const &a,
        TorchHandle const &b,
        TorchHandle const &out);
};

//! fp32 ternary family: three R in, one W out.
template<typename T>
class TorchTernary;

template<typename T>
class TorchTernary<std::tuple<T>>
{
public:
    CodeletTyped<T> codelet;
    using args_t = starpu::TorchDispatchArgs;

    TorchTernary();

    static void cpu(void *buffers[], void *cl_args) noexcept;
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
#endif
    static std::uint64_t footprint(
        void const *cl_args, std::size_t cl_arg_size) noexcept
    {
        return torch_args_footprint(cl_args, cl_arg_size);
    }

    void submit(
        int worker_hint,
        starpu::TorchDispatchArgs const &meta,
        TorchHandle const &a,
        TorchHandle const &b,
        TorchHandle const &c,
        TorchHandle const &out);
};

//! Embedding gather: i64 indices R, vocab R, out W.
class TorchEmbedding
{
public:
    Codelet codelet;
    using args_t = starpu::TorchDispatchArgs;

    TorchEmbedding();

    static void cpu(void *buffers[], void *cl_args) noexcept;
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
#endif
    static std::uint64_t footprint(
        void const *cl_args, std::size_t cl_arg_size) noexcept
    {
        return torch_args_footprint(cl_args, cl_arg_size);
    }

    void submit(
        int worker_hint,
        starpu::TorchDispatchArgs const &meta,
        TorchHandle const &weight,
        TorchHandle const &indices,
        TorchHandle const &out);
};

//! Embedding dense backward: grad R, indices R, grad_weight W.
class TorchEmbeddingDenseBackward
{
public:
    Codelet codelet;
    using args_t = starpu::TorchDispatchArgs;

    TorchEmbeddingDenseBackward();

    static void cpu(void *buffers[], void *cl_args) noexcept;
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
#endif
    static std::uint64_t footprint(
        void const *cl_args, std::size_t cl_arg_size) noexcept
    {
        return torch_args_footprint(cl_args, cl_arg_size);
    }

    void submit(
        int worker_hint,
        starpu::TorchDispatchArgs const &meta,
        TorchHandle const &grad,
        TorchHandle const &indices,
        TorchHandle const &grad_weight);
};

//! NLL loss forward: log_probs R, target R, loss W, total_weight W.
class TorchNllLossForward
{
public:
    Codelet codelet;
    using args_t = starpu::TorchDispatchArgs;

    TorchNllLossForward();

    static void cpu(void *buffers[], void *cl_args) noexcept;
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
#endif
    static std::uint64_t footprint(
        void const *cl_args, std::size_t cl_arg_size) noexcept
    {
        return torch_args_footprint(cl_args, cl_arg_size);
    }

    void submit(
        int worker_hint,
        starpu::TorchDispatchArgs const &meta,
        TorchHandle const &log_probs,
        TorchHandle const &target,
        TorchHandle const &loss,
        TorchHandle const &total_weight);
};

//! NLL loss backward: grad_loss R, log_probs R, target R,
//! total_weight R, grad_input W.
class TorchNllLossBackward
{
public:
    Codelet codelet;
    using args_t = starpu::TorchDispatchArgs;

    TorchNllLossBackward();

    static void cpu(void *buffers[], void *cl_args) noexcept;
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
#endif
    static std::uint64_t footprint(
        void const *cl_args, std::size_t cl_arg_size) noexcept
    {
        return torch_args_footprint(cl_args, cl_arg_size);
    }

    void submit(
        int worker_hint,
        starpu::TorchDispatchArgs const &meta,
        TorchHandle const &grad_output,
        TorchHandle const &log_probs,
        TorchHandle const &target,
        TorchHandle const &total_weight,
        TorchHandle const &grad_input);
};

//! Variable-arity cat: up to torch_dispatch_max_tensors fp32 inputs.
class TorchCat
{
public:
    Codelet codelet;
    using args_t = starpu::TorchDispatchArgs;

    TorchCat();

    static void cpu(void *buffers[], void *cl_args) noexcept;
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
#endif
    static std::uint64_t footprint(
        void const *cl_args, std::size_t cl_arg_size) noexcept
    {
        return torch_args_footprint(cl_args, cl_arg_size);
    }

    void submit(
        int worker_hint,
        starpu::TorchDispatchArgs const &meta,
        std::vector<TorchHandle> const &inputs,
        TorchHandle const &out);
};

//! Where: condition (bool) R, x R, y R, out W.
class TorchWhere
{
public:
    Codelet codelet;
    using args_t = starpu::TorchDispatchArgs;

    TorchWhere();

    static void cpu(void *buffers[], void *cl_args) noexcept;
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
#endif
    static std::uint64_t footprint(
        void const *cl_args, std::size_t cl_arg_size) noexcept
    {
        return torch_args_footprint(cl_args, cl_arg_size);
    }

    void submit(
        int worker_hint,
        starpu::TorchDispatchArgs const &meta,
        TorchHandle const &condition,
        TorchHandle const &self,
        TorchHandle const &other,
        TorchHandle const &out);
};

//! Write-only int64/fp32/bool fill family (arange, full).
class TorchArange
{
public:
    Codelet codelet;
    using args_t = starpu::TorchDispatchArgs;

    TorchArange();

    static void cpu(void *buffers[], void *cl_args) noexcept;
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
#endif
    static std::uint64_t footprint(
        void const *cl_args, std::size_t cl_arg_size) noexcept
    {
        return torch_args_footprint(cl_args, cl_arg_size);
    }

    void submit(
        int worker_hint,
        starpu::TorchDispatchArgs const &meta,
        TorchHandle const &out);
};

//! int64 elementwise: gt/lt → bool, add/sub/mul/minimum → i64.
class TorchGt
{
public:
    Codelet codelet;
    using args_t = starpu::TorchDispatchArgs;

    TorchGt();

    static void cpu(void *buffers[], void *cl_args) noexcept;
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
#endif
    static std::uint64_t footprint(
        void const *cl_args, std::size_t cl_arg_size) noexcept
    {
        return torch_args_footprint(cl_args, cl_arg_size);
    }

    void submit(
        int worker_hint,
        starpu::TorchDispatchArgs const &meta,
        TorchHandle const &a,
        TorchHandle const &b,
        TorchHandle const &out);
};

//! int64 unary: abs → i64.
class TorchI64Unary
{
public:
    Codelet codelet;
    using args_t = starpu::TorchDispatchArgs;

    TorchI64Unary();

    static void cpu(void *buffers[], void *cl_args) noexcept;
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
#endif
    static std::uint64_t footprint(
        void const *cl_args, std::size_t cl_arg_size) noexcept
    {
        return torch_args_footprint(cl_args, cl_arg_size);
    }

    void submit(
        int worker_hint,
        starpu::TorchDispatchArgs const &meta,
        TorchHandle const &in,
        TorchHandle const &out);
};

//! dtype cast: fp32/i64/bool source to fp32/i64/bool destination.
class TorchCast
{
public:
    Codelet codelet;
    using args_t = starpu::TorchDispatchArgs;

    TorchCast();

    static void cpu(void *buffers[], void *cl_args) noexcept;
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
#endif
    static std::uint64_t footprint(
        void const *cl_args, std::size_t cl_arg_size) noexcept
    {
        return torch_args_footprint(cl_args, cl_arg_size);
    }

    void submit(
        int worker_hint,
        starpu::TorchDispatchArgs const &meta,
        TorchHandle const &in,
        TorchHandle const &out);
};

//! Families the GPT-2/Neo/NeoX models never record. The core bridge
//! still names them; a submit throws instead of linking a ported
//! codelet, so the missing substrate cannot fail silently. Each later
//! model PR ports the families its models need.
class TorchStub
{
public:
    explicit TorchStub(const char *name) : name_(name)
    {
    }

    template<typename... Args>
    void submit(Args &&...) const
    {
        throw std::runtime_error(
            std::string("torch family not ported to NNHaul: ") + name_);
    }

private:
    const char *name_;
};

extern TorchStub const torch_sdpa_backward;

//! aten::convolution through the public dispatcher entry. The simple
//! port: at::convolution_out lets aten pick the backend (cuDNN on
//! CUDA), accepting the extra copying a hand-tuned backend switch
//! would avoid.
class TorchConvolution
{
public:
    Codelet codelet;
    using args_t = starpu::TorchDispatchArgs;

    TorchConvolution();

    static void cpu(void *buffers[], void *cl_args) noexcept;
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
#endif
    static std::uint64_t footprint(
        void const *cl_args, std::size_t cl_arg_size) noexcept
    {
        return torch_args_footprint(cl_args, cl_arg_size);
    }

    void submit(
        int worker_hint,
        starpu::TorchDispatchArgs const &meta,
        TorchHandle const &input,
        TorchHandle const &weight,
        TorchHandle const &bias, // aliases input when the op has none
        TorchHandle const &out,
        bool has_bias);
};

//! aten::convolution_backward through the public dispatcher entry
//! (at::convolution_backward_out writes caller-provided buffers).
class TorchConvolutionBackward
{
public:
    Codelet codelet;
    using args_t = starpu::TorchDispatchArgs;

    TorchConvolutionBackward();

    static void cpu(void *buffers[], void *cl_args) noexcept;
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
#endif
    static std::uint64_t footprint(
        void const *cl_args, std::size_t cl_arg_size) noexcept
    {
        return torch_args_footprint(cl_args, cl_arg_size);
    }

    void submit(
        int worker_hint,
        starpu::TorchDispatchArgs const &meta,
        TorchHandle const &grad_out,
        TorchHandle const &input,
        TorchHandle const &weight,
        TorchHandle const &grad_input, // aliases inputs when not needed
        TorchHandle const &grad_weight,
        TorchHandle const &grad_bias,
        bool need_grad_input,
        bool need_grad_weight,
        bool need_grad_bias);
};

//! aten::max_pool2d_with_indices (out variant), 2-D only.
class TorchMaxPool2dWithIndices
{
public:
    Codelet codelet;
    using args_t = starpu::TorchDispatchArgs;

    TorchMaxPool2dWithIndices();

    static void cpu(void *buffers[], void *cl_args) noexcept;
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
#endif
    static std::uint64_t footprint(
        void const *cl_args, std::size_t cl_arg_size) noexcept
    {
        return torch_args_footprint(cl_args, cl_arg_size);
    }

    void submit(
        int worker_hint,
        starpu::TorchDispatchArgs const &meta,
        TorchHandle const &input,
        TorchHandle const &out,
        TorchHandle const &indices);
};

//! aten::max_pool2d_with_indices_backward (out variant), 2-D only.
class TorchMaxPool2dWithIndicesBackward
{
public:
    Codelet codelet;
    using args_t = starpu::TorchDispatchArgs;

    TorchMaxPool2dWithIndicesBackward();

    static void cpu(void *buffers[], void *cl_args) noexcept;
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
#endif
    static std::uint64_t footprint(
        void const *cl_args, std::size_t cl_arg_size) noexcept
    {
        return torch_args_footprint(cl_args, cl_arg_size);
    }

    void submit(
        int worker_hint,
        starpu::TorchDispatchArgs const &meta,
        TorchHandle const &grad_out,
        TorchHandle const &input,
        TorchHandle const &indices,
        TorchHandle const &grad_input);
};

extern TorchConvolution torch_convolution;
extern TorchConvolutionBackward torch_convolution_backward;
extern TorchMaxPool2dWithIndices torch_max_pool2d_with_indices;
extern TorchMaxPool2dWithIndicesBackward torch_max_pool2d_with_indices_backward;

//! native_batch_norm forward. The layer_norm composite decomposes to
//! this family (training=true, no running stats), so GPT-2 records it
//! through every LayerNorm.
class TorchNativeBatchNorm
{
public:
    Codelet codelet;
    using args_t = starpu::TorchDispatchArgs;

    TorchNativeBatchNorm();

    static void cpu(void *buffers[], void *cl_args) noexcept;
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
#endif
    static std::uint64_t footprint(
        void const *cl_args, std::size_t cl_arg_size) noexcept
    {
        return torch_args_footprint(cl_args, cl_arg_size);
    }

    void submit(
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
        bool training);
};

//! native_batch_norm backward.
class TorchNativeBatchNormBackward
{
public:
    Codelet codelet;
    using args_t = starpu::TorchDispatchArgs;

    TorchNativeBatchNormBackward();

    static void cpu(void *buffers[], void *cl_args) noexcept;
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
#endif
    static std::uint64_t footprint(
        void const *cl_args, std::size_t cl_arg_size) noexcept
    {
        return torch_args_footprint(cl_args, cl_arg_size);
    }

    void submit(
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
        bool need_grad_bias);
};

using torch_unary_pack_t = OperationPack<
    TorchUnary,
    std::tuple<nntile::fp32_t>
>;
using torch_binary_pack_t = OperationPack<
    TorchBinary,
    std::tuple<nntile::fp32_t>
>;
using torch_ternary_pack_t = OperationPack<
    TorchTernary,
    std::tuple<nntile::fp32_t>
>;

extern TorchNativeBatchNorm torch_native_batch_norm;
extern TorchNativeBatchNormBackward torch_native_batch_norm_backward;
extern torch_unary_pack_t torch_unary;
extern torch_binary_pack_t torch_binary;
extern torch_ternary_pack_t torch_ternary;
extern TorchEmbedding torch_embedding;
extern TorchEmbeddingDenseBackward torch_embedding_dense_backward;
extern TorchNllLossForward torch_nll_loss_forward;
extern TorchNllLossBackward torch_nll_loss_backward;
extern TorchCat torch_cat;
extern TorchWhere torch_where;
extern TorchArange torch_arange;
extern TorchGt torch_gt;
extern TorchI64Unary torch_i64_unary;
extern TorchCast torch_cast;

} // namespace nntile::haul
