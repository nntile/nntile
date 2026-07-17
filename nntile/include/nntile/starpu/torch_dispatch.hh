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

#include <nntile/core/torch_meta.hh>
#include <nntile/starpu/codelet.hh>
#include <nntile/starpu/handle.hh>

namespace nntile::starpu
{

//! Which aten kernel the CPU codelet should call.
//!
//! Names match torch aten schemas (not NNTile classic kernels). TensorGraph
//! records the same kind; StarPU CPU/CUDA wrappers call the matching
//! ``at::*_out`` / ``*_copy_out`` on ``device=CPU`` / ``device=CUDA``
//! under ``NoGradGuard``. CUDA uses StarPU stream + cuBLAS handle.
//!
//! Access modes (out-of-place ``*_out`` unless noted): each tensor is
//! read-only (``STARPU_R``), write-only (``STARPU_W``), read-write
//! (``STARPU_RW``), or workspace (``STARPU_SCRATCH``). See
//! ``docs/dev/torch_starpu_kernels.md`` for the full table. Family
//! codelets below implement the common out-of-place pattern
//! ``R… + W``; ``Addmm`` may use ``RW`` when ``out`` aliases the first
//! input (accumulate).
enum class TorchKind : std::int32_t
{
    Mul = 1,                 // R,R → W  aten::mul.out
    Hypot = 2,               // R,R → W  aten::hypot.out
    MulScalar = 3,           // R → W    aten::mul.Scalar_out
    Add = 4,                 // R,R → W  aten::add.out (alpha in scalars[0])
    Relu = 10,               // R → W    aten::relu.out
    Silu = 11,               // R → W    aten::silu.out
    Gelu = 12,               // R → W    aten::gelu.out
    ThresholdBackward = 20,  // R,R → W  aten::threshold_backward
    SiluBackward = 21,       // R,R → W  aten::silu_backward
    GeluBackward = 22,       // R,R → W  aten::gelu_backward
    Softmax = 30,            // R → W    aten::_softmax.out
    SoftmaxBackward = 31,   // R,R → W  aten::_softmax_backward_data
    LogSoftmax = 32,         // R → W    aten::_log_softmax.out
    LogSoftmaxBackward = 33,// R,R → W  aten::_log_softmax_backward_data
    NllLossForward = 34,     // R,R → W,W  aten::nll_loss_forward
    NllLossBackward = 35,    // R,R,R,R → W  aten::nll_loss_backward
    Sum = 40,                // R → W    aten::sum.IntList_out
    VectorNorm = 41,         // R → W    aten::linalg_vector_norm.out
    Mm = 50,                 // R,R → W  aten::mm.out
    Bmm = 51,                // R,R → W  aten::bmm.out
    Addmm = 52,              // R,R,R→W or RW,R,R  aten::addmm.out
    Matmul = 53,             // R,R → W  aten::matmul.out
    Linear = 54,             // R,R → W or R,R,R→W  aten::linear.out
    Cat = 60,                // R… → W   aten::cat.out
    NarrowCopy = 61,         // R → W    aten::narrow_copy.out
    Repeat = 62,             // R → W    aten::repeat.out
    NativeLayerNorm = 70,    // R,(R),(R) → W,W,W  native_layer_norm
    NativeLayerNormBackward = 71, // R… → W…  native_layer_norm_backward
    Embedding = 80,          // R,R → W  aten::embedding.out
    EmbeddingDenseBackward = 81, // R,R → W  embedding_dense_backward
    Sdpa = 90,               // R,R,R → W  scaled_dot_product_attention
    SdpaBackward = 91,       // R… → W,W,W  SDPA backward
    TransposeCopy = 100,     // R → W    aten::transpose_copy.int_out
    Copy = 101,              // R → W    densify / contiguous (copy_)
};

inline constexpr Index torch_dispatch_max_ndim = core::torch_native_max_ndim;
inline constexpr Index torch_dispatch_max_tensors = 8;

//! Shared packed meta (used by unary/binary/reduce/mm families).
struct TorchDispatchArgs
{
    TorchKind kind = TorchKind::Mul;
    Index n_in = 0;
    Index n_out = 1;
    Scalar scalars[4] = {0, 0, 0, 0};
    Index iargs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    // iargs layout (per kind):
    // Softmax/SoftmaxBackward/LogSoftmax*: dim
    // Sum/VectorNorm: n_dims, keepdim, dim0..
    // Gelu*: approximate_tanh in iargs[0]
    // NarrowCopy: dim, start, length
    // Repeat: repeat counts in iargs[0..ndim-1]
    // Cat: dim, n_tensors
    // NativeLayerNorm: normalized_ndim, has_weight, has_bias;
    //   eps in scalars[0]
    // NativeLayerNormBackward: normalized_ndim, has_weight,
    //   has_bias, need_gi, need_gw, need_gb
    // NllLoss*: reduction in iargs[0], ignore_index in iargs[1]
    // Add: torch alpha in scalars[0] (out = a + alpha * b)
    // Addmm: beta in scalars[0], alpha in scalars[1];
    //   iargs[7]=1 when out aliases first input (STARPU_RW)
    // Sdpa/SdpaBackward: has_mask in iargs[0], is_causal in
    //   iargs[1]
    // EmbeddingDenseBackward: num_weights in iargs[0]
    // TransposeCopy: dim0, dim1
    char sarg[16] = {};
    Index in_ndim[torch_dispatch_max_tensors] = {};
    Index out_ndim[torch_dispatch_max_tensors] = {};
    Index in_sizes[torch_dispatch_max_tensors][torch_dispatch_max_ndim] = {};
    Index out_sizes[torch_dispatch_max_tensors][torch_dispatch_max_ndim] = {};
    Index in_strides[torch_dispatch_max_tensors][torch_dispatch_max_ndim] =
        {};
    Index out_strides[torch_dispatch_max_tensors][torch_dispatch_max_ndim] =
        {};
    //! Element offsets into StarPU buffers (views); 0 for dense tiles.
    Index in_offset[torch_dispatch_max_tensors] = {};
    Index out_offset[torch_dispatch_max_tensors] = {};
};

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

//! LayerNorm: input + optional weight/bias → out, mean, rstd.
class TorchLayerNorm
{
public:
    Codelet codelet;
    TorchLayerNorm();
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
        Handle mean,
        Handle rstd,
        bool has_weight,
        bool has_bias
    );
};

//! LayerNorm backward: inputs R → optional grad outs W.
class TorchLayerNormBackward
{
public:
    Codelet codelet;
    TorchLayerNormBackward();
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
extern TorchLayerNorm torch_layer_norm;
extern TorchLayerNormBackward torch_layer_norm_backward;
extern TorchSdpaBackward torch_sdpa_backward;
extern TorchNllLossForward torch_nll_loss_forward;
extern TorchNllLossBackward torch_nll_loss_backward;
extern TorchCat torch_cat;

} // namespace nntile::starpu
