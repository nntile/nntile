/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file include/nntile/starpu/torch_dispatch.hh
 * Family StarPU codelets for torch-native aten kernels (CPU).
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
//! records the same kind; StarPU CPU wrappers call the matching
//! ``at::*_out`` / ``*_copy_out`` on ``device=CPU`` under ``NoGradGuard``.
enum class TorchKind : std::int32_t
{
    Mul = 1,                 // aten::mul.out / mul.Scalar_out
    Hypot = 2,               // aten::hypot.out
    MulScalar = 3,           // aten::mul.Scalar_out
    Relu = 10,               // aten::relu.out
    Silu = 11,               // aten::silu.out
    Gelu = 12,               // aten::gelu.out
    ThresholdBackward = 20,  // aten::threshold_backward.grad_input
    SiluBackward = 21,       // aten::silu_backward.grad_input
    GeluBackward = 22,       // aten::gelu_backward.grad_input
    Softmax = 30,            // aten::_softmax.out
    SoftmaxBackward = 31,   // aten::_softmax_backward_data.out
    Sum = 40,                // aten::sum.IntList_out
    VectorNorm = 41,         // aten::linalg_vector_norm.out
    Mm = 50,                 // aten::mm.out
    Bmm = 51,                // aten::bmm.out
    Addmm = 52,              // aten::addmm.out
    Matmul = 53,             // aten::matmul.out
    Linear = 54,             // aten::linear.out
    Cat = 60,                // aten::cat.out
    NarrowCopy = 61,         // aten::narrow_copy.out
    Repeat = 62,             // aten::repeat.out
    NativeLayerNorm = 70,    // aten::native_layer_norm
    Embedding = 80,          // aten::embedding.out
    Sdpa = 90,               // aten::scaled_dot_product_attention
    TransposeCopy = 100,     // aten::transpose_copy.int_out
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
    // Softmax/SoftmaxBackward: dim
    // Sum/VectorNorm: n_dims, keepdim, dim0..
    // Gelu*: approximate_tanh in iargs[0]
    // NarrowCopy: dim, start, length
    // Repeat: repeat counts in iargs[0..ndim-1]
    // Cat: dim, n_tensors
    // NativeLayerNorm: normalized_ndim, has_weight, has_bias;
    //   eps in scalars[0]
    // Addmm: beta in scalars[0], alpha in scalars[1]
    // Sdpa: has_mask in iargs[0], is_causal in iargs[1]
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
    static constexpr func_array cuda_funcs = {};
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
    static constexpr func_array cuda_funcs = {};
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
    static constexpr func_array cuda_funcs = {};
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
    static constexpr func_array cuda_funcs = {};
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
    static constexpr func_array cuda_funcs = {};
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
    static constexpr func_array cuda_funcs = {};
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
extern TorchLayerNorm torch_layer_norm;
extern TorchCat torch_cat;

} // namespace nntile::starpu
