/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_executor_torch_native.cpp
 * Minimal executor for NNTILE_TORCH_NATIVE_OPS (add + fill).
 */

#include "nntile_executor.h"

#include "nntile_graph_recorder.h"
#include "nntile_graph_recorder_impl.h"
#include "nntile_tensor_meta.h"

#include <c10/util/Exception.h>

#include <nntile/base_types.hh>
#include <nntile/tensor/ops/fill.hh>
#include <nntile/tensor/ops/torch_add.hh>

#include <vector>

namespace torch_nntile
{

namespace
{

std::vector<nntile::Index> pytorch_shape_to_graph(c10::IntArrayRef shape)
{
    std::vector<nntile::Index> graph_shape;
    graph_shape.reserve(shape.size());
    for (const auto dim : shape)
    {
        graph_shape.push_back(static_cast<nntile::Index>(dim));
    }
    return graph_shape;
}

bool mark_as_input_for_operand(const at::Tensor &tensor)
{
    if (tensor.device().is_cpu())
    {
        return true;
    }
    return false;
}

[[noreturn]] void throw_op_disabled(const char *name)
{
    TORCH_CHECK(
        false,
        "torch_nntile: operation '",
        name,
        "' is disabled under NNTILE_TORCH_NATIVE_OPS "
        "(only out-of-place add is available)");
}

} // namespace

void tensor_add_fp32(
    float alpha,
    const at::Tensor &x,
    float beta,
    const at::Tensor &y,
    at::Tensor &out)
{
    // NNTile historical API: z = alpha * x + beta * y.
    // Torch add.out is out = self + alpha * other. Require alpha==1 and
    // map beta → torch alpha.
    TORCH_CHECK(
        alpha == 1.0f,
        "torch_nntile torch_add: only alpha=1 on the left "
        "operand is supported (z = x + beta * y)");
    TORCH_CHECK(
        x.sizes().equals(y.sizes()) && x.sizes().equals(out.sizes()),
        "torch_nntile torch_add: same-shape tensors only");
    TORCH_CHECK(
        x.scalar_type() == at::kFloat &&
            y.scalar_type() == at::kFloat &&
            out.scalar_type() == at::kFloat,
        "torch_nntile torch_add: float32 only");

    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(x.sizes());

    auto *x_node = get_or_create_data_node(
        x,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(x));
    auto *y_node = get_or_create_data_node(
        y,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(y));

    auto *z_node = nntile::tensor::torch_add(
        x_node,
        y_node,
        static_cast<nntile::Scalar>(beta))->set_name("z");
    register_data_node(out, z_node);
}

void tensor_fill_fp32(at::Tensor &self, float value)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(self.sizes());

    auto *self_node = get_or_create_data_node(
        self,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(self));
    nntile::tensor::fill(static_cast<nntile::Scalar>(value), self_node);
    register_data_node(self, self_node);
}

void tensor_swap_two_axes_fp32(
    const at::Tensor &,
    at::Tensor &,
    int64_t,
    int64_t)
{
    throw_op_disabled("swap_two_axes / transpose");
}

void tensor_add_inplace_fp32(float, const at::Tensor &, float, at::Tensor &)
{
    throw_op_disabled("add_");
}

void tensor_model_transpose_forward_fp32(
    const at::Tensor &,
    at::Tensor &,
    int64_t)
{
    throw_op_disabled("model_transpose");
}

void tensor_model_transpose_backward_fp32(
    const at::Tensor &,
    at::Tensor &,
    int64_t)
{
    throw_op_disabled("model_transpose_backward");
}

void tensor_mul_fp32(const at::Tensor &, const at::Tensor &, at::Tensor &)
{
    throw_op_disabled("mul");
}

void tensor_mul_inplace_fp32(const at::Tensor &, at::Tensor &)
{
    throw_op_disabled("mul_");
}

void tensor_hypot_fp32(const at::Tensor &, const at::Tensor &, at::Tensor &)
{
    throw_op_disabled("hypot");
}

void tensor_linear_fp32(const at::Tensor &, const at::Tensor &, at::Tensor &)
{
    throw_op_disabled("linear");
}

void tensor_relu_fp32(const at::Tensor &, at::Tensor &)
{
    throw_op_disabled("relu");
}

void tensor_relu_backward_fp32(
    const at::Tensor &,
    const at::Tensor &,
    at::Tensor &)
{
    throw_op_disabled("relu_backward");
}

void tensor_silu_fp32(const at::Tensor &, at::Tensor &)
{
    throw_op_disabled("silu");
}

void tensor_silu_inplace_fp32(at::Tensor &)
{
    throw_op_disabled("silu_");
}

void tensor_silu_backward_fp32(
    const at::Tensor &,
    const at::Tensor &,
    at::Tensor &)
{
    throw_op_disabled("silu_backward");
}

void tensor_gelu_fp32(const at::Tensor &, at::Tensor &, bool)
{
    throw_op_disabled("gelu");
}

void tensor_gelu_inplace_fp32(at::Tensor &, bool)
{
    throw_op_disabled("gelu_");
}

void tensor_gelu_backward_fp32(
    const at::Tensor &,
    const at::Tensor &,
    at::Tensor &,
    bool)
{
    throw_op_disabled("gelu_backward");
}

void tensor_gemm_fp32(
    const GemmParams &,
    const at::Tensor &,
    c10::IntArrayRef,
    const at::Tensor &,
    c10::IntArrayRef,
    at::Tensor &,
    c10::IntArrayRef)
{
    throw_op_disabled("gemm");
}

void tensor_gemm_accumulate_fp32(
    const GemmParams &,
    const at::Tensor &,
    c10::IntArrayRef,
    const at::Tensor &,
    c10::IntArrayRef,
    const at::Tensor &,
    c10::IntArrayRef,
    at::Tensor &,
    c10::IntArrayRef)
{
    throw_op_disabled("gemm_accumulate");
}

void tensor_mm_fp32(const at::Tensor &, const at::Tensor &, at::Tensor &)
{
    throw_op_disabled("mm");
}

void tensor_linear_backward_input_fp32(
    const at::Tensor &,
    const at::Tensor &,
    at::Tensor &)
{
    throw_op_disabled("linear_backward_input");
}

void tensor_linear_backward_weight_fp32(
    const at::Tensor &,
    const at::Tensor &,
    at::Tensor &)
{
    throw_op_disabled("linear_backward_weight");
}

void tensor_linear_add_bias_fp32(at::Tensor &, const at::Tensor &)
{
    throw_op_disabled("linear_add_bias");
}

void tensor_linear_grad_bias_fp32(const at::Tensor &, at::Tensor &)
{
    throw_op_disabled("linear_grad_bias");
}

void tensor_add_fiber_fp32(
    float,
    const at::Tensor &,
    float,
    const at::Tensor &,
    at::Tensor &,
    int64_t,
    int64_t)
{
    throw_op_disabled("add_fiber");
}


void tensor_sum_fiber_fp32(
    const at::Tensor &, at::Tensor &, int64_t, int64_t, float)
{ throw_op_disabled("sum_fiber"); }

void tensor_sum_slice_fp32(
    const at::Tensor &, at::Tensor &, int64_t, float, float)
{ throw_op_disabled("sum_slice"); }

void tensor_add_slice_fp32(
    float, const at::Tensor &, float, const at::Tensor &, at::Tensor &,
    int64_t)
{ throw_op_disabled("add_slice"); }

void tensor_cross_entropy_forward_fp32(
    const at::Tensor &, const at::Tensor &, std::int64_t, bool,
    at::Tensor &, at::Tensor &)
{ throw_op_disabled("cross_entropy_forward"); }

void tensor_cross_entropy_backward_fp32(
    const at::Tensor &, const at::Tensor &, const at::Tensor &,
    const at::Tensor &, at::Tensor &, at::Tensor &, std::int64_t, bool)
{ throw_op_disabled("cross_entropy_backward"); }

void tensor_softmax_fp32(const at::Tensor &, at::Tensor &, int64_t)
{ throw_op_disabled("softmax"); }

void tensor_softmax_backward_fp32(
    const at::Tensor &, const at::Tensor &, at::Tensor &, int64_t)
{ throw_op_disabled("softmax_backward"); }

void tensor_sgd_step_fp32(
    int64_t, float, float, float, float, bool, const at::Tensor &,
    at::Tensor &, at::Tensor &)
{ throw_op_disabled("sgd_step"); }

void tensor_adam_step_fp32(
    int64_t, float, float, float, float, float, const at::Tensor &,
    at::Tensor &, at::Tensor &, at::Tensor &)
{ throw_op_disabled("adam_step"); }

void tensor_adamw_step_fp32(
    int64_t, float, float, float, float, float, const at::Tensor &,
    at::Tensor &, at::Tensor &, at::Tensor &)
{ throw_op_disabled("adamw_step"); }

void tensor_layer_norm_forward_fp32(
    const at::Tensor &, const at::Tensor *, const at::Tensor *, bool, bool,
    at::Tensor &, at::Tensor &, at::Tensor &, int64_t, float)
{ throw_op_disabled("layer_norm_forward"); }

void tensor_layer_norm_backward_fp32(
    const at::Tensor &, const at::Tensor &, const at::Tensor &,
    const at::Tensor &, const at::Tensor *, bool, bool, at::Tensor *,
    at::Tensor *, at::Tensor *, bool, bool, bool, int64_t)
{ throw_op_disabled("layer_norm_backward"); }

void tensor_rms_norm_forward_fp32(
    const at::Tensor &, const at::Tensor *, bool, at::Tensor &,
    at::Tensor &, int64_t, float)
{ throw_op_disabled("rms_norm_forward"); }

void tensor_rms_norm_backward_fp32(
    const at::Tensor &, const at::Tensor &, const at::Tensor &,
    const at::Tensor *, bool, at::Tensor *, at::Tensor *, bool, bool,
    int64_t)
{ throw_op_disabled("rms_norm_backward"); }

void tensor_rope_fp32(
    const at::Tensor &, const at::Tensor &, const at::Tensor &, at::Tensor &)
{ throw_op_disabled("rope"); }

void tensor_rope_backward_fp32(
    const at::Tensor &, const at::Tensor &, const at::Tensor &, at::Tensor &)
{ throw_op_disabled("rope_backward"); }

void tensor_mse_loss_fp32(const at::Tensor &, float, at::Tensor &)
{ throw_op_disabled("mse_loss"); }

void tensor_mse_loss_backward_fp32(const at::Tensor &, float, at::Tensor &)
{ throw_op_disabled("mse_loss_backward"); }

void tensor_norm_fp32(const at::Tensor &, at::Tensor &)
{ throw_op_disabled("norm"); }

void tensor_norm_slice_fp32(const at::Tensor &, at::Tensor &, int64_t, bool)
{ throw_op_disabled("norm_slice"); }

void tensor_sum_dimlist_fp32(
    const at::Tensor &, at::Tensor &, at::OptionalIntArrayRef, bool)
{ throw_op_disabled("sum_dimlist"); }

void tensor_mul_scalar_fp32(const at::Tensor &, at::Tensor &, float)
{ throw_op_disabled("mul_scalar"); }

void tensor_cat_fp32(
    const std::vector<at::Tensor> &, at::Tensor &, int64_t)
{ throw_op_disabled("cat"); }

void tensor_narrow_fp32(
    const at::Tensor &, int64_t, int64_t, int64_t, at::Tensor &)
{ throw_op_disabled("narrow"); }

void tensor_split_with_sizes_fp32(
    const at::Tensor &, int64_t, const std::vector<int64_t> &,
    const std::vector<at::Tensor> &)
{ throw_op_disabled("split_with_sizes"); }

void tensor_embedding_forward_fp32(
    const at::Tensor &, const at::Tensor &, at::Tensor &, nntile::Index)
{ throw_op_disabled("embedding_forward"); }

void tensor_embedding_backward_fp32(
    const at::Tensor &, const at::Tensor &, at::Tensor &, nntile::Index, int)
{ throw_op_disabled("embedding_backward"); }

void tensor_sdpa_forward_fp32(
    const at::Tensor &, const at::Tensor &, const at::Tensor &,
    const at::Tensor *, at::Tensor &, int64_t)
{ throw_op_disabled("sdpa_forward"); }

void tensor_sdpa_backward_fp32(
    const at::Tensor &, const at::Tensor &, const at::Tensor &,
    const at::Tensor *, const at::Tensor &, at::Tensor &, at::Tensor &,
    at::Tensor &, int64_t)
{ throw_op_disabled("sdpa_backward"); }

} // namespace torch_nntile
