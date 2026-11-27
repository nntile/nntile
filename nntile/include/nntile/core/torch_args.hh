/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file include/nntile/core/torch_args.hh
 * Backend-neutral IR for the torch-native dispatch: the aten kind enum
 * and the packed per-op arguments shared by the StarPU and NNHaul
 * codelet substrates.
 *
 * @version 1.1.0
 */

#pragma once

#include <cstdint>

#include <nntile/base_types.hh>
#include <nntile/core/torch_meta.hh>

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
    PowScalar = 184,         // R → W    aten::pow.Tensor_Scalar_out
    Div = 185,               // R,R → W  aten::div.out
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
    // Where value dtype: iargs[15] (0=fp32, 1=i64, 2=bool binary, 3=fp32*bool)
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

} // namespace nntile::starpu
