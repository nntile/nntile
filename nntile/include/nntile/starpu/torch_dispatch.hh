/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file include/nntile/starpu/torch_dispatch.hh
 * Family StarPU codelets for torch-native aten kernels (CPU/CUDA).
 *
 * @version 1.1.0
 */

#pragma once

#include <nntile/defs.h>

#ifndef NNTILE_TORCH_NATIVE_OPS
#error "nntile/starpu/torch_dispatch.hh requires NNTILE_TORCH_NATIVE_OPS"
#endif

#include <cstdint>
#include <tuple>
#include <vector>

#include <nntile/core/torch_args.hh>
#ifndef NNTILE_USE_NNHAUL
#include <nntile/starpu/codelet.hh>
#include <nntile/starpu/handle.hh>
#endif

namespace nntile::starpu
{

// StarPU codelet subclasses. TorchKind and TorchDispatchArgs above stay
// available in both builds. NNHaul does not compile these classes.
#ifndef NNTILE_USE_NNHAUL
template<typename T>
class TorchUnary;

template<typename T>
class TorchUnary<std::tuple<T>>
{
public:
    CodeletTyped<T> codelet;
    TorchUnary();
    using args_t = TorchDispatchArgs;
    static uint32_t footprint(struct starpu_task *task);
    static void cpu(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cpu_funcs = {cpu};
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cuda_funcs = {cuda};
#else
    static constexpr func_array cuda_funcs = {};
#endif
    void submit(
        int starpu_worker_hint,
        const args_t &meta,
        Handle in,
        Handle out
    );
};

template<typename T>
class TorchBinary;

template<typename T>
class TorchBinary<std::tuple<T>>
{
public:
    CodeletTyped<T> codelet;
    TorchBinary();
    using args_t = TorchDispatchArgs;
    static uint32_t footprint(struct starpu_task *task);
    static void cpu(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cpu_funcs = {cpu};
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cuda_funcs = {cuda};
#else
    static constexpr func_array cuda_funcs = {};
#endif
    void submit(
        int starpu_worker_hint,
        const args_t &meta,
        Handle a,
        Handle b,
        Handle out
    );
};

template<typename T>
class TorchTernary;

template<typename T>
class TorchTernary<std::tuple<T>>
{
public:
    CodeletTyped<T> codelet;
    TorchTernary();
    using args_t = TorchDispatchArgs;
    static uint32_t footprint(struct starpu_task *task);
    static void cpu(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cpu_funcs = {cpu};
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cuda_funcs = {cuda};
#else
    static constexpr func_array cuda_funcs = {};
#endif
    void submit(
        int starpu_worker_hint,
        const args_t &meta,
        Handle a,
        Handle b,
        Handle c,
        Handle out
    );
};

//! Where: condition bool + self fp32 + other fp32 → out fp32.
//!
//! ``other`` may be a scalar tile; aten::where broadcasts. Avoids the
//! host gather/scatter path that leaked StarPU buffers on GPT-Neo eager
//! attention (``torch.where(mask, scores, finfo.min)`` every layer).
class TorchWhere
{
public:
    Codelet codelet;
    TorchWhere();
    using args_t = TorchDispatchArgs;
    static uint32_t footprint(struct starpu_task *task);
    static void cpu(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cpu_funcs = {cpu};
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cuda_funcs = {cuda};
#else
    static constexpr func_array cuda_funcs = {};
#endif
    void submit(
        int starpu_worker_hint,
        const args_t &meta,
        Handle condition,
        Handle self,
        Handle other,
        Handle out
    );
};

//! Write-only int64 arange (no host copy into nntile).
class TorchArange
{
public:
    Codelet codelet;
    TorchArange();
    using args_t = TorchDispatchArgs;
    static uint32_t footprint(struct starpu_task *task);
    static void cpu(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cpu_funcs = {cpu};
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cuda_funcs = {cuda};
#else
    static constexpr func_array cuda_funcs = {};
#endif
    void submit(
        int starpu_worker_hint,
        const args_t &meta,
        Handle out
    );
};

//! int64 elementwise: ``gt``/``lt`` → bool, or add/sub/mul/minimum
//! → int64 (broadcast layouts packed in ``args``).
class TorchGt
{
public:
    Codelet codelet;
    TorchGt();
    using args_t = TorchDispatchArgs;
    static uint32_t footprint(struct starpu_task *task);
    static void cpu(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cpu_funcs = {cpu};
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cuda_funcs = {cuda};
#else
    static constexpr func_array cuda_funcs = {};
#endif
    void submit(
        int starpu_worker_hint,
        const args_t &meta,
        Handle a,
        Handle b,
        Handle out
    );
};

//! int64 unary (``abs``). Layouts packed in ``args``.
class TorchI64Unary
{
public:
    Codelet codelet;
    TorchI64Unary();
    using args_t = TorchDispatchArgs;
    static uint32_t footprint(struct starpu_task *task);
    static void cpu(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cpu_funcs = {cpu};
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cuda_funcs = {cuda};
#else
    static constexpr func_array cuda_funcs = {};
#endif
    void submit(
        int starpu_worker_hint,
        const args_t &meta,
        Handle in,
        Handle out
    );
};

//! Same-shape copy with a dtype change (bool/i64/fp32).
class TorchCast
{
public:
    Codelet codelet;
    TorchCast();
    using args_t = TorchDispatchArgs;
    static uint32_t footprint(struct starpu_task *task);
    static void cpu(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cpu_funcs = {cpu};
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cuda_funcs = {cuda};
#else
    static constexpr func_array cuda_funcs = {};
#endif
    void submit(
        int starpu_worker_hint,
        const args_t &meta,
        Handle in,
        Handle out
    );
};

//! Embedding: weight fp32 + indices i64 + out fp32 (mixed handles).
class TorchEmbedding
{
public:
    Codelet codelet;
    TorchEmbedding();
    using args_t = TorchDispatchArgs;
    static uint32_t footprint(struct starpu_task *task);
    static void cpu(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cpu_funcs = {cpu};
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cuda_funcs = {cuda};
#else
    static constexpr func_array cuda_funcs = {};
#endif
    void submit(
        int starpu_worker_hint,
        const args_t &meta,
        Handle weight,
        Handle indices,
        Handle out
    );
};

//! Embedding dense backward: grad + indices → grad_weight.
class TorchEmbeddingDenseBackward
{
public:
    Codelet codelet;
    TorchEmbeddingDenseBackward();
    using args_t = TorchDispatchArgs;
    static uint32_t footprint(struct starpu_task *task);
    static void cpu(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cpu_funcs = {cpu};
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cuda_funcs = {cuda};
#else
    static constexpr func_array cuda_funcs = {};
#endif
    void submit(
        int starpu_worker_hint,
        const args_t &meta,
        Handle grad,
        Handle indices,
        Handle grad_weight
    );
};

//! Convolution: input + weight + optional bias → out.
class TorchConvolution
{
public:
    Codelet codelet;
    TorchConvolution();
    using args_t = TorchDispatchArgs;
    static uint32_t footprint(struct starpu_task *task);
    static void cpu(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cpu_funcs = {cpu};
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cuda_funcs = {cuda};
#else
    static constexpr func_array cuda_funcs = {};
#endif
    void submit(
        int starpu_worker_hint,
        const args_t &meta,
        Handle input,
        Handle weight,
        Handle bias,
        Handle out,
        bool has_bias
    );
};

//! Convolution backward: grad_out + input + weight → optional grad outs.
class TorchConvolutionBackward
{
public:
    Codelet codelet;
    TorchConvolutionBackward();
    using args_t = TorchDispatchArgs;
    static uint32_t footprint(struct starpu_task *task);
    static void cpu(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cpu_funcs = {cpu};
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cuda_funcs = {cuda};
#else
    static constexpr func_array cuda_funcs = {};
#endif
    void submit(
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
    );
};

//! MaxPool2d with indices: input fp32 → output fp32 + indices i64.
class TorchMaxPool2dWithIndices
{
public:
    Codelet codelet;
    TorchMaxPool2dWithIndices();
    using args_t = TorchDispatchArgs;
    static uint32_t footprint(struct starpu_task *task);
    static void cpu(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cpu_funcs = {cpu};
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cuda_funcs = {cuda};
#else
    static constexpr func_array cuda_funcs = {};
#endif
    void submit(
        int starpu_worker_hint,
        const args_t &meta,
        Handle input,
        Handle out,
        Handle indices
    );
};

//! MaxPool2d backward: grad_out + self + indices i64 → grad_input.
class TorchMaxPool2dWithIndicesBackward
{
public:
    Codelet codelet;
    TorchMaxPool2dWithIndicesBackward();
    using args_t = TorchDispatchArgs;
    static uint32_t footprint(struct starpu_task *task);
    static void cpu(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cpu_funcs = {cpu};
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cuda_funcs = {cuda};
#else
    static constexpr func_array cuda_funcs = {};
#endif
    void submit(
        int starpu_worker_hint,
        const args_t &meta,
        Handle grad_out,
        Handle input,
        Handle indices,
        Handle grad_input
    );
};

//! Native batch norm: input + optional affine/running stats → out, stats.
class TorchNativeBatchNorm
{
public:
    Codelet codelet;
    TorchNativeBatchNorm();
    using args_t = TorchDispatchArgs;
    static uint32_t footprint(struct starpu_task *task);
    static void cpu(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cpu_funcs = {cpu};
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cuda_funcs = {cuda};
#else
    static constexpr func_array cuda_funcs = {};
#endif
    void submit(
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
    );
};

//! Native batch norm backward: inputs R → optional grad outs W.
class TorchNativeBatchNormBackward
{
public:
    Codelet codelet;
    TorchNativeBatchNormBackward();
    using args_t = TorchDispatchArgs;
    static uint32_t footprint(struct starpu_task *task);
    static void cpu(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cpu_funcs = {cpu};
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cuda_funcs = {cuda};
#else
    static constexpr func_array cuda_funcs = {};
#endif
    void submit(
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
    );
};

//! SDPA backward: q,k,v,grad_out,(mask) → grad_q,k,v.
class TorchSdpaBackward
{
public:
    Codelet codelet;
    TorchSdpaBackward();
    using args_t = TorchDispatchArgs;
    static uint32_t footprint(struct starpu_task *task);
    static void cpu(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cpu_funcs = {cpu};
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cuda_funcs = {cuda};
#else
    static constexpr func_array cuda_funcs = {};
#endif
    void submit(
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
    );
};

//! NLL loss forward: log_probs + target → loss, total_weight.
class TorchNllLossForward
{
public:
    Codelet codelet;
    TorchNllLossForward();
    using args_t = TorchDispatchArgs;
    static uint32_t footprint(struct starpu_task *task);
    static void cpu(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cpu_funcs = {cpu};
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cuda_funcs = {cuda};
#else
    static constexpr func_array cuda_funcs = {};
#endif
    void submit(
        int starpu_worker_hint,
        const args_t &meta,
        Handle log_probs,
        Handle target,
        Handle loss,
        Handle total_weight
    );
};

//! NLL loss backward: grad_loss + log_probs + target + tw → grad.
class TorchNllLossBackward
{
public:
    Codelet codelet;
    TorchNllLossBackward();
    using args_t = TorchDispatchArgs;
    static uint32_t footprint(struct starpu_task *task);
    static void cpu(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cpu_funcs = {cpu};
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cuda_funcs = {cuda};
#else
    static constexpr func_array cuda_funcs = {};
#endif
    void submit(
        int starpu_worker_hint,
        const args_t &meta,
        Handle grad_output,
        Handle log_probs,
        Handle target,
        Handle total_weight,
        Handle grad_input
    );
};

//! Variable-arity cat: up to torch_dispatch_max_tensors fp32 inputs.
class TorchCat
{
public:
    Codelet codelet;
    TorchCat();
    using args_t = TorchDispatchArgs;
    static uint32_t footprint(struct starpu_task *task);
    static void cpu(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cpu_funcs = {cpu};
#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args) noexcept;
    static constexpr func_array cuda_funcs = {cuda};
#else
    static constexpr func_array cuda_funcs = {};
#endif
    void submit(
        int starpu_worker_hint,
        const args_t &meta,
        const std::vector<Handle> &inputs,
        Handle out
    );
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

extern torch_unary_pack_t torch_unary;
extern torch_binary_pack_t torch_binary;
extern torch_ternary_pack_t torch_ternary;
extern TorchEmbedding torch_embedding;
extern TorchEmbeddingDenseBackward torch_embedding_dense_backward;
extern TorchConvolution torch_convolution;
extern TorchConvolutionBackward torch_convolution_backward;
extern TorchMaxPool2dWithIndices torch_max_pool2d_with_indices;
extern TorchMaxPool2dWithIndicesBackward
    torch_max_pool2d_with_indices_backward;
extern TorchNativeBatchNorm torch_native_batch_norm;
extern TorchNativeBatchNormBackward torch_native_batch_norm_backward;
extern TorchSdpaBackward torch_sdpa_backward;
extern TorchNllLossForward torch_nll_loss_forward;
extern TorchNllLossBackward torch_nll_loss_backward;
extern TorchCat torch_cat;
extern TorchWhere torch_where;
extern TorchArange torch_arange;
extern TorchGt torch_gt;
extern TorchI64Unary torch_i64_unary;
extern TorchCast torch_cast;
#else // NNTILE_USE_NNHAUL
} // namespace nntile::starpu

// The codelet substrate runs on the NNHaul backend; the haul families
// carry the same names so every starpu:: call site keeps compiling.
// The include sits at global scope: the haul header defines its own
// namespaces.
#include <nntile/nnhaul/torch_dispatch.hh>

namespace nntile::starpu
{

using namespace nntile::haul;
using Handle = ::nntile::haul::TorchHandle;
#endif // NNTILE_USE_NNHAUL

} // namespace nntile::starpu
