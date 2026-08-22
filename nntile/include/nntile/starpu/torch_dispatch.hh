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
    Sub = 5,                 // R,R → W  aten::sub.out (alpha in scalars[0])
    Relu = 10,               // R → W    aten::relu.out
    Silu = 11,               // R → W    aten::silu.out
    Gelu = 12,               // R → W    aten::gelu.out
    Cos = 13,                // R → W    aten::cos.out
    Sin = 14,                // R → W    aten::sin.out
    Neg = 15,                // R → W    aten::neg.out
    Rsqrt = 16,              // R → W    aten::rsqrt.out
    Exp = 17,                // R → W    aten::exp.out
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
    Mean = 42,               // R → W    aten::mean.out
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
    Sdpa = 90,               // D8 unused fused SDPA; F.sdpa uses MATH
    SdpaBackward = 91,       // D8 unused fused SDPA backward
    TransposeCopy = 100,     // R → W    aten::transpose_copy.int_out
    Copy = 101,              // R → W    densify / contiguous (copy_)
    CopyIntoView = 180,      // R → RW   copy_ into packed parent view
    Triu = 102,              // R → W    aten::triu.out (diagonal iargs[0])
    AvgPool2d = 110,         // R → W    aten::avg_pool2d.out
    AvgPool2dBackward = 111, // R,R → W  aten::avg_pool2d_backward
    AdaptiveAvgPool2d = 112, // R → W    aten::_adaptive_avg_pool2d.out
    AdaptiveAvgPool2dBackward = 113, // R,R → W  adaptive avg pool bwd
    Convolution = 120,       // R,R,(R) → W  aten::convolution
    ConvolutionBackward = 121, // R,R,R → W...  convolution_backward
    MaxPool2dWithIndices = 130, // R → W,W(i64)  max_pool2d_with_indices
    MaxPool2dWithIndicesBackward = 131, // R,R,R(i64) → W
    NativeBatchNorm = 140,   // R,(R),(R),(RW),(RW) → W,W,W
    NativeBatchNormBackward = 141, // R... → W... native_batch_norm_backward
    UpsampleNearest2d = 150, // R → W    aten::upsample_nearest2d.out
    UpsampleNearest2dBackward = 151, // R → W  upsample_nearest2d_backward
    UpsampleBilinear2d = 152, // R → W   aten::upsample_bilinear2d.out
    UpsampleBilinear2dBackward = 153, // R → W upsample_bilinear2d_backward
    Where = 160,             // R(bool),R,R → W  aten::where.out
    Arange = 170,            // → W(i64) aten::arange.out
    ArangeFp32 = 179,        // → W(fp32) aten::arange.out
    Gt = 171,                // R(i64),R(i64) → W(bool) aten::gt.out
    Lt = 172,                // R(i64),R(i64) → W(bool) aten::lt.out
    Minimum = 173,           // R(i64),R(i64) → W(i64) aten::minimum.out
    Abs = 174,               // R(i64) → W(i64) aten::abs.out
    Log = 175,               // R → W    aten::log.out (fp32 unary)
    Cast = 176,              // R → W    copy_ with dtype change
    FillI64 = 177,           // → W(i64) aten fill_ (arange codelet)
    Eq = 178,                // R(fp32),R(fp32) → W(bool) aten::eq.out
    FillBool = 182,          // → W(bool) aten fill_ (arange codelet)
    Tril = 183,              // R(bool) → W(bool) aten::tril.out
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
    Index iargs[16] = {};
    // iargs layout (per kind):
    // Softmax/SoftmaxBackward/LogSoftmax*: dim
    // Sum/Mean/VectorNorm: n_dims, keepdim, dim0..
    // Gelu*: approximate_tanh in iargs[0]
    // NarrowCopy: dim, start, length
    // Repeat: repeat counts in iargs[0..out_ndim-1] (output rank; may
    //   pad leading dims when the input tile is still the 1D parent)
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
    // CopyIntoView: iargs[7]=1 when out aliases in (one STARPU_RW)
    // Triu: diagonal in iargs[0]
    // Arange: start/end/step in iargs[0..2] (int64)
    // ArangeFp32: start/end/step in scalars[0..2]
    // FillI64: value in iargs[0] (int64); same write-only
    //   codelet as Arange
    // FillBool: value in iargs[0] (0/1); same codelet as Arange
    // Tril: diagonal in iargs[0]; bool unary (MATH SDPA mask)
    // Gt/Lt: none (broadcast via packed layouts)
    // Cast: src dtype tag iargs[0], dst tag iargs[1]
    //   (0=fp32, 1=i64, 2=bool)
    // Where value dtype: iargs[15] (0=fp32, 1=i64)
    // AvgPool2d: kernel [0..1], stride [2..3], padding [4..5],
    //   ceil_mode [6], count_include_pad [7], has_divisor [8],
    //   divisor [9]
    // AdaptiveAvgPool2d: output_size H,W in iargs[0..1]
    // Convolution: spatial_ndim [0], groups [1], transposed [2],
    //   stride [3..4], padding [5..6], dilation [7..8],
    //   output_padding [9..10], has_bias [11], output_mask [12..14]
    // MaxPool2d: kernel [0..1], stride [2..3], padding [4..5],
    //   dilation [6..7], ceil_mode [8]
    // NativeBatchNorm: training [0], has_weight [1], has_bias [2],
    //   has_running_mean [3], has_running_var [4], has_save_mean [5],
    //   has_save_invstd [6], output_mask [7..9]; momentum in
    //   scalars[0], eps in scalars[1]
    // UpsampleNearest2d forward: out_h[0], out_w[1], has_scales_h[2],
    //   has_scales_w[3]; scales in scalars[0..1]
    // UpsampleNearest2dBackward: out_h[0], out_w[1], in_n[2], in_c[3],
    //   in_h[4], in_w[5], has_scales_h[6], has_scales_w[7];
    //   scales in scalars[0..1]
    // UpsampleBilinear2d forward: out_h[0], out_w[1], align_corners[2],
    //   has_scales_h[3], has_scales_w[4]; scales in scalars[0..1]
    // UpsampleBilinear2dBackward: out_h[0], out_w[1], in_n[2], in_c[3],
    //   in_h[4], in_w[5], align_corners[6], has_scales_h[7],
    //   has_scales_w[8]; scales in scalars[0..1]
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
    //! 1 if sizes/strides/offset were packed for this slot.
    //! Distinguishes a packed scalar (``ndim == 0``) from an unpacked
    //! slot that should fall back to the contiguous tile shape.
    Index in_layout_set[torch_dispatch_max_tensors] = {};
    Index out_layout_set[torch_dispatch_max_tensors] = {};
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
extern TorchLayerNorm torch_layer_norm;
extern TorchLayerNormBackward torch_layer_norm_backward;
extern TorchSdpaBackward torch_sdpa_backward;
extern TorchNllLossForward torch_nll_loss_forward;
extern TorchNllLossBackward torch_nll_loss_backward;
extern TorchCat torch_cat;
extern TorchWhere torch_where;
extern TorchArange torch_arange;
extern TorchGt torch_gt;
extern TorchI64Unary torch_i64_unary;
extern TorchCast torch_cast;

} // namespace nntile::starpu
