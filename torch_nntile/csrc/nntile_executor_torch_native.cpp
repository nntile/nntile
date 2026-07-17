/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_executor_torch_native.cpp
 * Torch-native TensorGraph executor for NNTILE_TORCH_NATIVE_OPS.
 */

#include "nntile_executor.h"

#include "nntile_broadcast.h"
#include "nntile_gemm_layout.h"
#include "nntile_graph_recorder.h"
#include "nntile_graph_recorder_impl.h"
#include "nntile_tensor_gc.h"
#include "nntile_tensor_meta.h"

#include <ATen/Functions.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include <nntile/base_types.hh>
#include <nntile/tensor/ops/fill.hh>
#include <nntile/tensor/ops/torch_dispatch.hh>

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

std::vector<nntile::Index> reduced_shape_along_axis(
    const std::vector<nntile::Index> &input_graph,
    nntile::Index axis)
{
    std::vector<nntile::Index> reduced;
    reduced.reserve(input_graph.size() - 1);
    for (nntile::Index i = 0;
         i < static_cast<nntile::Index>(input_graph.size());
         ++i)
    {
        if (i != axis)
        {
            reduced.push_back(input_graph[static_cast<std::size_t>(i)]);
        }
    }
    return reduced;
}

nntile::starpu::TorchKind torch_gemm_kind(c10::IntArrayRef a_shape,
    c10::IntArrayRef b_shape)
{
    if (a_shape.size() == 2 && b_shape.size() == 2)
    {
        return nntile::starpu::TorchKind::Mm;
    }
    if (a_shape.size() == 3 && b_shape.size() == 3)
    {
        return nntile::starpu::TorchKind::Bmm;
    }
    return nntile::starpu::TorchKind::Matmul;
}

//! Transpose the last two dims via aten::transpose_copy (for gemm trans flags).
nntile::TensorGraph::TensorNode *maybe_transpose_matrix_node(
    nntile::TensorGraph::TensorNode *node,
    std::vector<nntile::Index> shape,
    bool transpose)
{
    if (!transpose)
    {
        return node;
    }
    TORCH_CHECK(
        shape.size() >= 2,
        "torch_nntile gemm: transpose requires rank >= 2");
    const nntile::Index d0 =
        static_cast<nntile::Index>(shape.size()) - 2;
    const nntile::Index d1 =
        static_cast<nntile::Index>(shape.size()) - 1;
    std::swap(shape[static_cast<std::size_t>(d0)],
        shape[static_cast<std::size_t>(d1)]);
    nntile::starpu::TorchDispatchArgs extra{};
    extra.iargs[0] = d0;
    extra.iargs[1] = d1;
    return nntile::tensor::torch_unary(
        nntile::starpu::TorchKind::TransposeCopy,
        node,
        shape,
        extra);
}

void pack_sum_dims(
    nntile::starpu::TorchDispatchArgs &extra,
    const std::vector<int64_t> &dims,
    bool keepdim)
{
    TORCH_CHECK(
        dims.size() <= 6,
        "torch_nntile sum: at most 6 reduction dims supported");
    extra.iargs[0] = static_cast<nntile::Index>(dims.size());
    extra.iargs[1] = keepdim ? 1 : 0;
    for (std::size_t i = 0; i < dims.size(); ++i)
    {
        extra.iargs[2 + static_cast<nntile::Index>(i)] =
            static_cast<nntile::Index>(dims[i]);
    }
}

[[noreturn]] void throw_op_disabled(const char *name)
{
    TORCH_CHECK(
        false,
        "torch_nntile: operation '",
        name,
        "' is disabled under NNTILE_TORCH_NATIVE_OPS");
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
    // map beta → torch alpha (TorchKind::Add scalars[0]).
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

    nntile::starpu::TorchDispatchArgs extra;
    extra.scalars[0] = static_cast<nntile::Scalar>(beta);
    auto *z_node = nntile::tensor::torch_binary(
        nntile::starpu::TorchKind::Add,
        x_node,
        y_node,
        graph_shape,
        extra)->set_name("z");
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

void tensor_add_inplace_fp32(
    float alpha,
    const at::Tensor &other,
    float beta,
    at::Tensor &self)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(self.sizes());

    auto *self_node = get_or_create_data_node(
        self,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(self));
    auto *other_node = get_or_create_data_node(
        other,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(other));

    nntile::TensorGraph::TensorNode *lhs_node = self_node;
    if (beta != 1.0f)
    {
        nntile::starpu::TorchDispatchArgs scale_extra;
        scale_extra.scalars[0] = static_cast<nntile::Scalar>(beta);
        lhs_node = nntile::tensor::torch_unary(
            nntile::starpu::TorchKind::MulScalar,
            self_node,
            graph_shape,
            scale_extra);
    }

    nntile::starpu::TorchDispatchArgs extra;
    extra.scalars[0] = static_cast<nntile::Scalar>(alpha);
    auto *out_node = nntile::tensor::torch_binary(
        nntile::starpu::TorchKind::Add,
        lhs_node,
        other_node,
        graph_shape,
        extra);
    register_data_node(self, out_node);
}

void tensor_mul_fp32(
    const at::Tensor &self,
    const at::Tensor &other,
    at::Tensor &out)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(self.sizes());

    auto *self_node = get_or_create_data_node(
        self,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(self));
    auto *other_node = get_or_create_data_node(
        other,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(other));

    auto *out_node = nntile::tensor::torch_binary(
        nntile::starpu::TorchKind::Mul,
        self_node,
        other_node,
        graph_shape);
    register_data_node(out, out_node);
}

void tensor_mul_inplace_fp32(const at::Tensor &other, at::Tensor &self)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(self.sizes());

    auto *self_node = get_or_create_data_node(
        self,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(self));
    auto *other_node = get_or_create_data_node(
        other,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(other));

    auto *out_node = nntile::tensor::torch_binary(
        nntile::starpu::TorchKind::Mul,
        self_node,
        other_node,
        graph_shape);
    register_data_node(self, out_node);
}

void tensor_mul_scalar_fp32(
    const at::Tensor &input,
    at::Tensor &out,
    float scalar)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(input.sizes());

    auto *input_node = get_or_create_data_node(
        input,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(input));

    nntile::starpu::TorchDispatchArgs extra;
    extra.scalars[0] = static_cast<nntile::Scalar>(scalar);
    auto *out_node = nntile::tensor::torch_unary(
        nntile::starpu::TorchKind::MulScalar,
        input_node,
        graph_shape,
        extra);
    register_data_node(out, out_node);
}

void tensor_hypot_fp32(
    const at::Tensor &self,
    const at::Tensor &other,
    at::Tensor &out)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(self.sizes());

    auto *self_node = get_or_create_data_node(
        self,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(self));
    auto *other_node = get_or_create_data_node(
        other,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(other));

    auto *out_node = nntile::tensor::torch_binary(
        nntile::starpu::TorchKind::Hypot,
        self_node,
        other_node,
        graph_shape);
    register_data_node(out, out_node);
}

void tensor_relu_fp32(const at::Tensor &input, at::Tensor &out)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(input.sizes());

    auto *in_node = get_or_create_data_node(
        input,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(input));

    auto *out_node = nntile::tensor::torch_unary(
        nntile::starpu::TorchKind::Relu,
        in_node,
        graph_shape);
    register_data_node(out, out_node);
}

void tensor_relu_backward_fp32(
    const at::Tensor &x,
    const at::Tensor &dy,
    at::Tensor &dx)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(x.sizes());

    auto *x_node = get_or_create_data_node(
        x,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(x));
    auto *dy_node = get_or_create_data_node(
        dy,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(dy));
    auto *dx_node = get_or_create_data_node(
        dx,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(dx));

    nntile::starpu::TorchDispatchArgs extra;
    extra.scalars[0] = static_cast<nntile::Scalar>(0.0);
    nntile::tensor::torch_binary(
        nntile::starpu::TorchKind::ThresholdBackward,
        dy_node,
        x_node,
        dx_node,
        extra);
    register_data_node(dx, dx_node);
}

void tensor_silu_fp32(const at::Tensor &input, at::Tensor &out)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(input.sizes());

    auto *in_node = get_or_create_data_node(
        input,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(input));

    auto *out_node = nntile::tensor::torch_unary(
        nntile::starpu::TorchKind::Silu,
        in_node,
        graph_shape);
    register_data_node(out, out_node);
}

void tensor_silu_inplace_fp32(at::Tensor &self)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(self.sizes());

    auto *in_node = get_or_create_data_node(
        self,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(self));

    auto *out_node = nntile::tensor::torch_unary(
        nntile::starpu::TorchKind::Silu,
        in_node,
        graph_shape);
    register_data_node(self, out_node);
}

void tensor_silu_backward_fp32(
    const at::Tensor &x,
    const at::Tensor &dy,
    at::Tensor &dx)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(x.sizes());

    auto *x_node = get_or_create_data_node(
        x,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(x));
    auto *dy_node = get_or_create_data_node(
        dy,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(dy));
    auto *dx_node = get_or_create_data_node(
        dx,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(dx));

    nntile::tensor::torch_binary(
        nntile::starpu::TorchKind::SiluBackward,
        dy_node,
        x_node,
        dx_node);
    register_data_node(dx, dx_node);
}

void tensor_gelu_fp32(
    const at::Tensor &input,
    at::Tensor &out,
    bool approximate_tanh)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(input.sizes());

    auto *in_node = get_or_create_data_node(
        input,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(input));

    nntile::starpu::TorchDispatchArgs extra;
    extra.iargs[0] = approximate_tanh ? 1 : 0;
    auto *out_node = nntile::tensor::torch_unary(
        nntile::starpu::TorchKind::Gelu,
        in_node,
        graph_shape,
        extra);
    register_data_node(out, out_node);
}

void tensor_gelu_inplace_fp32(at::Tensor &self, bool approximate_tanh)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(self.sizes());

    auto *in_node = get_or_create_data_node(
        self,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(self));

    nntile::starpu::TorchDispatchArgs extra;
    extra.iargs[0] = approximate_tanh ? 1 : 0;
    auto *out_node = nntile::tensor::torch_unary(
        nntile::starpu::TorchKind::Gelu,
        in_node,
        graph_shape,
        extra);
    register_data_node(self, out_node);
}

void tensor_gelu_backward_fp32(
    const at::Tensor &x,
    const at::Tensor &dy,
    at::Tensor &dx,
    bool approximate_tanh)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(x.sizes());

    auto *x_node = get_or_create_data_node(
        x,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(x));
    auto *dy_node = get_or_create_data_node(
        dy,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(dy));
    auto *dx_node = get_or_create_data_node(
        dx,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(dx));

    nntile::starpu::TorchDispatchArgs extra;
    extra.iargs[0] = approximate_tanh ? 1 : 0;
    nntile::tensor::torch_binary(
        nntile::starpu::TorchKind::GeluBackward,
        dy_node,
        x_node,
        dx_node,
        extra);
    register_data_node(dx, dx_node);
}

void tensor_softmax_fp32(
    const at::Tensor &input,
    at::Tensor &out,
    int64_t dim)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(input.sizes());

    auto *in_node = get_or_create_data_node(
        input,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(input));

    nntile::starpu::TorchDispatchArgs extra;
    extra.iargs[0] = static_cast<nntile::Index>(dim);
    auto *out_node = nntile::tensor::torch_unary(
        nntile::starpu::TorchKind::Softmax,
        in_node,
        graph_shape,
        extra);
    register_data_node(out, out_node);
}

void tensor_softmax_backward_fp32(
    const at::Tensor &grad_output,
    const at::Tensor &output,
    at::Tensor &grad_input,
    int64_t dim)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(output.sizes());

    auto *grad_out_node = get_or_create_data_node(
        grad_output,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(grad_output));
    auto *out_node = get_or_create_data_node(
        output,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(output));
    auto *grad_in_node = get_or_create_data_node(
        grad_input,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(grad_input));

    nntile::starpu::TorchDispatchArgs extra;
    extra.iargs[0] = static_cast<nntile::Index>(dim);
    nntile::tensor::torch_binary(
        nntile::starpu::TorchKind::SoftmaxBackward,
        grad_out_node,
        out_node,
        grad_in_node,
        extra);
    register_data_node(grad_input, grad_in_node);
}

void tensor_gemm_fp32(
    const GemmParams &params,
    const at::Tensor &a,
    c10::IntArrayRef a_gemm_shape,
    const at::Tensor &b,
    c10::IntArrayRef b_gemm_shape,
    at::Tensor &out,
    c10::IntArrayRef /*out_shape*/)
{
    const std::vector<nntile::Index> a_graph =
        pytorch_shape_to_graph(a_gemm_shape);
    const std::vector<nntile::Index> b_graph =
        pytorch_shape_to_graph(b_gemm_shape);
    const std::vector<nntile::Index> out_graph =
        pytorch_shape_to_graph(out.sizes());

    auto *a_node = get_or_create_data_node(
        a,
        a_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(a));
    auto *b_node = get_or_create_data_node(
        b,
        b_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(b));

    // Honor classic gemm transpose flags by lowering to
    // aten::transpose_copy then mm/bmm/matmul (CPU, no grad).
    a_node = maybe_transpose_matrix_node(
        a_node,
        a_graph,
        params.trans_a);
    b_node = maybe_transpose_matrix_node(
        b_node,
        b_graph,
        params.trans_b);

    std::vector<nntile::Index> a_eff = a_graph;
    std::vector<nntile::Index> b_eff = b_graph;
    if (params.trans_a && a_eff.size() >= 2)
    {
        std::swap(a_eff[a_eff.size() - 2], a_eff[a_eff.size() - 1]);
    }
    if (params.trans_b && b_eff.size() >= 2)
    {
        std::swap(b_eff[b_eff.size() - 2], b_eff[b_eff.size() - 1]);
    }

    const std::vector<int64_t> a_i64(a_eff.begin(), a_eff.end());
    const std::vector<int64_t> b_i64(b_eff.begin(), b_eff.end());
    auto *out_node = nntile::tensor::torch_binary(
        torch_gemm_kind(a_i64, b_i64),
        a_node,
        b_node,
        out_graph);
    register_data_node(out, out_node);
}

void tensor_gemm_accumulate_fp32(
    const GemmParams &params,
    const at::Tensor &a,
    c10::IntArrayRef a_gemm_shape,
    const at::Tensor &b,
    c10::IntArrayRef b_gemm_shape,
    const at::Tensor &c,
    c10::IntArrayRef c_shape,
    at::Tensor &out,
    c10::IntArrayRef /*out_shape*/)
{
    const std::vector<nntile::Index> a_graph =
        pytorch_shape_to_graph(a_gemm_shape);
    const std::vector<nntile::Index> b_graph =
        pytorch_shape_to_graph(b_gemm_shape);
    const std::vector<nntile::Index> c_graph =
        pytorch_shape_to_graph(c_shape);

    auto *a_node = get_or_create_data_node(
        a,
        a_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(a));
    auto *b_node = get_or_create_data_node(
        b,
        b_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(b));
    auto *c_node = get_or_create_data_node(
        c,
        c_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(c));

    nntile::starpu::TorchDispatchArgs extra;
    extra.scalars[0] = static_cast<nntile::Scalar>(params.beta);
    extra.scalars[1] = static_cast<nntile::Scalar>(params.alpha);
    nntile::tensor::torch_ternary(
        nntile::starpu::TorchKind::Addmm,
        c_node,
        a_node,
        b_node,
        c_node,
        extra);
    register_data_node(out, c_node);
}

void tensor_mm_fp32(
    const at::Tensor &a,
    const at::Tensor &b,
    at::Tensor &out)
{
    const PreparedGemmOperands prepared = prepare_mm_operands(a, b);
    tensor_gemm_fp32(
        prepared.params,
        prepared.a,
        prepared.a_gemm_shape,
        prepared.b,
        prepared.b_gemm_shape,
        out,
        prepared.out_shape);
}

void tensor_linear_fp32(
    const at::Tensor &input,
    const at::Tensor &weight,
    at::Tensor &out)
{
    // aten::linear(input, weight) — same schema in the StarPU codelet.
    const std::vector<nntile::Index> in_graph =
        pytorch_shape_to_graph(input.sizes());
    const std::vector<nntile::Index> w_graph =
        pytorch_shape_to_graph(weight.sizes());
    const std::vector<nntile::Index> out_graph =
        pytorch_shape_to_graph(out.sizes());

    auto *in_node = get_or_create_data_node(
        input,
        in_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(input));
    auto *w_node = get_or_create_data_node(
        weight,
        w_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(weight));
    auto *out_node = nntile::tensor::torch_binary(
        nntile::starpu::TorchKind::Linear,
        in_node,
        w_node,
        out_graph);
    register_data_node(out, out_node);
}

void tensor_linear_bias_fp32(
    const at::Tensor &input,
    const at::Tensor &weight,
    const at::Tensor &bias,
    at::Tensor &out)
{
    // aten::linear(input, weight, bias).
    const std::vector<nntile::Index> in_graph =
        pytorch_shape_to_graph(input.sizes());
    const std::vector<nntile::Index> w_graph =
        pytorch_shape_to_graph(weight.sizes());
    const std::vector<nntile::Index> b_graph =
        pytorch_shape_to_graph(bias.sizes());
    const std::vector<nntile::Index> out_graph =
        pytorch_shape_to_graph(out.sizes());

    auto *in_node = get_or_create_data_node(
        input,
        in_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(input));
    auto *w_node = get_or_create_data_node(
        weight,
        w_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(weight));
    auto *b_node = get_or_create_data_node(
        bias,
        b_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(bias));
    auto *out_node = nntile::tensor::torch_ternary(
        nntile::starpu::TorchKind::Linear,
        in_node,
        w_node,
        b_node,
        out_graph);
    register_data_node(out, out_node);
}

void tensor_linear_backward_input_fp32(
    const at::Tensor &grad_out,
    const at::Tensor &weight,
    at::Tensor &grad_input)
{
    const PreparedGemmOperands forward =
        prepare_linear_operands(grad_input, weight);
    const GemmParams params =
        infer_linear_backward_grad_input_params(forward.params);
    const GemmMatrixLayout grad_out_layout =
        analyze_matrix_layout_for_nntile(grad_out);
    TORCH_CHECK(
        !grad_out_layout.needs_copy,
        "nntile linear_backward_input: grad_out must be contiguous or "
        "row/column-contiguous");
    tensor_gemm_fp32(
        params,
        grad_out,
        grad_out_layout.gemm_shape,
        weight,
        forward.b_gemm_shape,
        grad_input,
        forward.a_gemm_shape);
}

void tensor_linear_backward_weight_fp32(
    const at::Tensor &grad_out,
    const at::Tensor &input,
    at::Tensor &grad_weight)
{
    const PreparedGemmOperands forward =
        prepare_linear_operands(input, grad_weight);
    const GemmParams params =
        infer_linear_backward_grad_weight_params(forward.params);
    const GemmMatrixLayout grad_out_layout =
        analyze_matrix_layout_for_nntile(grad_out);
    TORCH_CHECK(
        !grad_out_layout.needs_copy,
        "nntile linear_backward_weight: grad_out must be contiguous or "
        "row/column-contiguous");
    TORCH_CHECK(
        forward.a.is_contiguous(),
        "nntile linear_backward_weight: input must be contiguous");
    tensor_gemm_fp32(
        params,
        grad_out,
        grad_out_layout.gemm_shape,
        forward.a,
        forward.a_gemm_shape,
        grad_weight,
        forward.b_gemm_shape);
}

void tensor_linear_add_bias_fp32(
    at::Tensor &output,
    const at::Tensor &bias)
{
    // Prefer recording aten::linear with bias when possible. Here output
    // already holds the matmul result; fold bias via add after repeat
    // using aten ops only (repeat + add).
    TORCH_CHECK(bias.dim() == 1, "linear_add_bias: bias must be 1D");
    TORCH_CHECK(
        output.size(-1) == bias.size(0),
        "linear_add_bias: trailing size mismatch");
    std::vector<int64_t> bshape(static_cast<size_t>(output.dim()), 1);
    bshape.back() = bias.size(0);
    at::Tensor bias_view = bias.reshape(bshape);
    std::vector<int64_t> repeats(static_cast<size_t>(output.dim()), 1);
    for (int64_t i = 0; i < output.dim() - 1; ++i)
    {
        repeats[static_cast<size_t>(i)] = output.size(i);
    }
    at::Tensor bias_b = at::empty(
        output.sizes(),
        output.options().memory_format(at::MemoryFormat::Contiguous));
    tensor_repeat_fp32(bias_view, bias_b, repeats);
    at::Tensor tmp = at::empty_like(output);
    tensor_add_fp32(1.0f, output, 1.0f, bias_b, tmp);
    const auto out_shape = pytorch_shape_to_graph(tmp.sizes());
    auto *node = get_or_create_data_node(
        tmp,
        out_shape,
        nntile::DataType::FP32,
        false);
    register_data_node(output, node);
}

void tensor_linear_grad_bias_fp32(
    const at::Tensor &grad_output,
    at::Tensor &grad_bias)
{
    TORCH_CHECK(
        grad_output.dim() >= 1,
        "nntile linear grad_bias: grad_output rank < 1");
    TORCH_CHECK(
        grad_bias.dim() == 1,
        "nntile linear grad_bias: grad_bias must be 1D");
    TORCH_CHECK(
        grad_bias.size(0) == grad_output.size(-1),
        "nntile linear grad_bias: size mismatch");

    const std::vector<nntile::Index> grad_out_graph =
        pytorch_shape_to_graph(grad_output.sizes());
    const std::vector<nntile::Index> grad_bias_graph =
        pytorch_shape_to_graph(grad_bias.sizes());

    auto *grad_out_node = get_or_create_data_node(
        grad_output,
        grad_out_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(grad_output));
    auto *grad_bias_node = get_or_create_data_node(
        grad_bias,
        grad_bias_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(grad_bias));

    std::vector<int64_t> dims;
    dims.reserve(static_cast<std::size_t>(grad_output.dim() - 1));
    for (int64_t i = 0; i < grad_output.dim() - 1; ++i)
    {
        dims.push_back(i);
    }

    nntile::starpu::TorchDispatchArgs extra;
    pack_sum_dims(extra, dims, false);
    nntile::tensor::torch_unary(
        nntile::starpu::TorchKind::Sum,
        grad_out_node,
        grad_bias_node,
        extra);
    register_data_node(grad_bias, grad_bias_node);
}

void tensor_norm_fp32(
    const at::Tensor &x,
    at::Tensor &out)
{
    const int64_t numel = x.numel();
    TORCH_CHECK(numel > 0, "torch_nntile norm: empty tensor");
    const std::vector<nntile::Index> flat_shape{
        static_cast<nntile::Index>(numel)};

    auto *x_node = get_or_create_data_node(
        x,
        flat_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(x));
    auto *out_node = get_or_create_data_node(
        out,
        std::vector<nntile::Index>{},
        nntile::DataType::FP32,
        mark_as_input_for_operand(out));

    nntile::starpu::TorchDispatchArgs extra;
    extra.iargs[0] = 1;
    extra.iargs[1] = 0;
    extra.iargs[2] = 0;
    nntile::tensor::torch_unary(
        nntile::starpu::TorchKind::VectorNorm,
        x_node,
        out_node,
        extra);
    register_data_node(out, out_node);
}

void tensor_norm_slice_fp32(
    const at::Tensor &x,
    at::Tensor &out,
    int64_t axis,
    bool keepdim)
{
    const std::vector<nntile::Index> input_graph =
        pytorch_shape_to_graph(x.sizes());
    const nntile::Index ax = static_cast<nntile::Index>(axis);

    auto *x_node = get_or_create_data_node(
        x,
        input_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(x));

    const std::vector<nntile::Index> out_graph =
        pytorch_shape_to_graph(out.sizes());
    auto *out_node = get_or_create_data_node(
        out,
        out_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(out));

    nntile::starpu::TorchDispatchArgs extra;
    extra.iargs[0] = 1;
    extra.iargs[1] = keepdim ? 1 : 0;
    extra.iargs[2] = ax;
    nntile::tensor::torch_unary(
        nntile::starpu::TorchKind::VectorNorm,
        x_node,
        out_node,
        extra);
    register_data_node(out, out_node);
}

void tensor_sum_dimlist_fp32(
    const at::Tensor &input,
    at::Tensor &out,
    at::OptionalIntArrayRef dim,
    bool keepdim)
{
    const c10::IntArrayRef input_shape = input.sizes();
    const int64_t rank = static_cast<int64_t>(input_shape.size());
    TORCH_CHECK(rank > 0, "nntile sum: cannot sum a 0-dim tensor");

    std::vector<int64_t> dims;
    if (!dim.has_value() || dim->empty())
    {
        dims.reserve(static_cast<std::size_t>(rank));
        for (int64_t i = 0; i < rank; ++i)
        {
            dims.push_back(i);
        }
    }
    else
    {
        dims.reserve(dim->size());
        for (const auto d : *dim)
        {
            const int64_t axis = d < 0 ? d + rank : d;
            TORCH_CHECK(
                axis >= 0 && axis < rank,
                "nntile sum: dimension out of range");
            dims.push_back(axis);
        }
    }
    std::sort(dims.begin(), dims.end(), std::greater<int64_t>());

    const std::vector<nntile::Index> input_graph =
        pytorch_shape_to_graph(input_shape);
    const std::vector<nntile::Index> out_graph =
        pytorch_shape_to_graph(out.sizes());

    auto *in_node = get_or_create_data_node(
        input,
        input_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(input));
    auto *out_node = get_or_create_data_node(
        out,
        out_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(out));

    nntile::starpu::TorchDispatchArgs extra;
    pack_sum_dims(extra, dims, keepdim);
    nntile::tensor::torch_unary(
        nntile::starpu::TorchKind::Sum,
        in_node,
        out_node,
        extra);
    register_data_node(out, out_node);
}

void tensor_cat_fp32(
    const std::vector<at::Tensor> &inputs,
    at::Tensor &out,
    int64_t dim)
{
    TORCH_CHECK(!inputs.empty(), "tensor_cat_fp32: expected non-empty inputs");
    const nntile::Index axis = static_cast<nntile::Index>(dim);
    const std::vector<nntile::Index> out_graph =
        pytorch_shape_to_graph(out.sizes());

    std::vector<nntile::TensorGraph::TensorNode *> nodes;
    nodes.reserve(inputs.size());
    for (const auto &tensor : inputs)
    {
        const std::vector<nntile::Index> shape_graph =
            pytorch_shape_to_graph(tensor.sizes());
        nodes.push_back(get_or_create_data_node(
            tensor,
            shape_graph,
            nntile::DataType::FP32,
            mark_as_input_for_operand(tensor)));
    }

    auto *out_node = nntile::tensor::torch_cat(axis, nodes, out_graph);
    register_data_node(out, out_node);
}

void tensor_narrow_fp32(
    const at::Tensor &input,
    int64_t dim,
    int64_t start,
    int64_t length,
    at::Tensor &out)
{
    const std::vector<nntile::Index> in_graph =
        pytorch_shape_to_graph(input.sizes());
    const std::vector<nntile::Index> out_graph =
        pytorch_shape_to_graph(out.sizes());

    auto *in_node = get_or_create_data_node(
        input,
        in_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(input));

    nntile::starpu::TorchDispatchArgs extra;
    extra.iargs[0] = static_cast<nntile::Index>(dim);
    extra.iargs[1] = static_cast<nntile::Index>(start);
    extra.iargs[2] = static_cast<nntile::Index>(length);
    auto *out_node = nntile::tensor::torch_unary(
        nntile::starpu::TorchKind::NarrowCopy,
        in_node,
        out_graph,
        extra);
    register_data_node(out, out_node);
}

void tensor_split_with_sizes_fp32(
    const at::Tensor &input,
    int64_t dim,
    const std::vector<int64_t> &split_sizes,
    const std::vector<at::Tensor> &outputs)
{
    const std::vector<nntile::Index> in_graph =
        pytorch_shape_to_graph(input.sizes());

    auto *in_node = get_or_create_data_node(
        input,
        in_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(input));

    nntile::Index offset = 0;
    for (std::size_t i = 0; i < split_sizes.size(); ++i)
    {
        const std::vector<nntile::Index> out_graph =
            pytorch_shape_to_graph(outputs[i].sizes());

        nntile::starpu::TorchDispatchArgs extra;
        extra.iargs[0] = static_cast<nntile::Index>(dim);
        extra.iargs[1] = offset;
        extra.iargs[2] =
            static_cast<nntile::Index>(split_sizes[i]);
        auto *out_node = nntile::tensor::torch_unary(
            nntile::starpu::TorchKind::NarrowCopy,
            in_node,
            out_graph,
            extra);
        register_data_node(outputs[i], out_node);
        offset += static_cast<nntile::Index>(split_sizes[i]);
    }
}

void tensor_embedding_forward_fp32(
    const at::Tensor &indices,
    const at::Tensor &weight,
    at::Tensor &out,
    nntile::Index /*axis*/)
{
    const std::vector<nntile::Index> index_graph =
        pytorch_shape_to_graph(indices.sizes());
    const std::vector<nntile::Index> weight_graph =
        pytorch_shape_to_graph(weight.sizes());
    const std::vector<nntile::Index> out_graph =
        pytorch_shape_to_graph(out.sizes());

    auto *weight_node = get_or_create_data_node(
        weight,
        weight_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(weight));
    auto *index_node = get_or_create_data_node(
        indices,
        index_graph,
        nntile::DataType::INT64,
        mark_as_input_for_operand(indices));

    auto *out_node = nntile::tensor::torch_embedding(
        weight_node,
        index_node,
        out_graph);
    register_data_node(out, out_node);
}

void tensor_layer_norm_forward_fp32(
    const at::Tensor &input,
    const at::Tensor *weight,
    const at::Tensor *bias,
    bool has_weight,
    bool has_bias,
    at::Tensor &output,
    at::Tensor &mean,
    at::Tensor &rstd,
    int64_t norm_axis,
    float eps)
{
    const std::vector<nntile::Index> input_graph =
        pytorch_shape_to_graph(input.sizes());
    const nntile::Index axis = static_cast<nntile::Index>(norm_axis);
    const nntile::Index norm_len =
        input_graph[static_cast<std::size_t>(axis)];
    const std::vector<nntile::Index> reduced_graph =
        reduced_shape_along_axis(input_graph, axis);
    const nntile::Index normalized_ndim =
        static_cast<nntile::Index>(input_graph.size()) - axis;

    auto *input_node = get_or_create_data_node(
        input,
        input_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(input));
    auto *out_node = get_or_create_data_node(
        output,
        input_graph,
        nntile::DataType::FP32,
        false);
    auto *mean_node = get_or_create_data_node(
        mean,
        reduced_graph,
        nntile::DataType::FP32,
        false);
    auto *rstd_node = get_or_create_data_node(
        rstd,
        reduced_graph,
        nntile::DataType::FP32,
        false);

    nntile::TensorGraph::TensorNode *weight_node = nullptr;
    nntile::TensorGraph::TensorNode *bias_node = nullptr;
    if (has_weight)
    {
        weight_node = get_or_create_data_node(
            *weight,
            {norm_len},
            nntile::DataType::FP32,
            mark_as_input_for_operand(*weight));
    }
    if (has_bias)
    {
        bias_node = get_or_create_data_node(
            *bias,
            {norm_len},
            nntile::DataType::FP32,
            mark_as_input_for_operand(*bias));
    }

    nntile::tensor::torch_layer_norm(
        input_node,
        weight_node,
        bias_node,
        out_node,
        mean_node,
        rstd_node,
        normalized_ndim,
        static_cast<nntile::Scalar>(eps));

    register_data_node(output, out_node);
    register_data_node(mean, mean_node);
    register_data_node(rstd, rstd_node);
}

void tensor_sdpa_forward_fp32(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor *mask,
    at::Tensor &out,
    int64_t /*batch_ndim*/)
{
    const std::vector<nntile::Index> q_graph =
        pytorch_shape_to_graph(q.sizes());

    auto *q_node = get_or_create_data_node(
        q,
        q_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(q));
    auto *k_node = get_or_create_data_node(
        k,
        pytorch_shape_to_graph(k.sizes()),
        nntile::DataType::FP32,
        mark_as_input_for_operand(k));
    auto *v_node = get_or_create_data_node(
        v,
        pytorch_shape_to_graph(v.sizes()),
        nntile::DataType::FP32,
        mark_as_input_for_operand(v));

    nntile::starpu::TorchDispatchArgs extra;
    extra.iargs[0] = mask != nullptr ? 1 : 0;
    extra.iargs[1] = 0;
    auto *out_node = nntile::tensor::torch_ternary(
        nntile::starpu::TorchKind::Sdpa,
        q_node,
        k_node,
        v_node,
        q_graph,
        extra);
    register_data_node(out, out_node);
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

void tensor_swap_two_axes_fp32(
    const at::Tensor &src,
    at::Tensor &dst,
    int64_t dim0,
    int64_t dim1)
{
    const auto in_shape = pytorch_shape_to_graph(src.sizes());
    const auto out_shape = pytorch_shape_to_graph(dst.sizes());
    auto *in_node = get_or_create_data_node(
        src,
        in_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(src));
    nntile::starpu::TorchDispatchArgs extra{};
    extra.iargs[0] = static_cast<nntile::Index>(dim0);
    extra.iargs[1] = static_cast<nntile::Index>(dim1);
    auto *out_node = nntile::tensor::torch_unary(
        nntile::starpu::TorchKind::TransposeCopy,
        in_node,
        out_shape,
        extra);
    register_data_node(dst, out_node);
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
    const at::Tensor &,
    at::Tensor &,
    int64_t,
    int64_t,
    float)
{
    throw_op_disabled("sum_fiber");
}

void tensor_sum_slice_fp32(
    const at::Tensor &,
    at::Tensor &,
    int64_t,
    float,
    float)
{
    throw_op_disabled("sum_slice");
}

void tensor_add_slice_fp32(
    float,
    const at::Tensor &,
    float,
    const at::Tensor &,
    at::Tensor &,
    int64_t)
{
    throw_op_disabled("add_slice");
}

void tensor_cross_entropy_forward_fp32(
    const at::Tensor &,
    const at::Tensor &,
    std::int64_t,
    bool,
    at::Tensor &,
    at::Tensor &)
{
    throw_op_disabled("cross_entropy_forward");
}

void tensor_cross_entropy_backward_fp32(
    const at::Tensor &,
    const at::Tensor &,
    const at::Tensor &,
    const at::Tensor &,
    at::Tensor &,
    at::Tensor &,
    std::int64_t,
    bool)
{
    throw_op_disabled("cross_entropy_backward");
}

void tensor_sgd_step_fp32(
    int64_t,
    float,
    float,
    float,
    float,
    bool,
    const at::Tensor &,
    at::Tensor &,
    at::Tensor &)
{
    throw_op_disabled("sgd_step");
}

void tensor_adam_step_fp32(
    int64_t,
    float,
    float,
    float,
    float,
    float,
    const at::Tensor &,
    at::Tensor &,
    at::Tensor &,
    at::Tensor &)
{
    throw_op_disabled("adam_step");
}

void tensor_adamw_step_fp32(
    int64_t,
    float,
    float,
    float,
    float,
    float,
    const at::Tensor &,
    at::Tensor &,
    at::Tensor &,
    at::Tensor &)
{
    throw_op_disabled("adamw_step");
}

void tensor_layer_norm_backward_fp32(
    const at::Tensor &grad_out,
    const at::Tensor &input,
    const at::Tensor &mean,
    const at::Tensor &rstd,
    const at::Tensor *weight,
    const at::Tensor *bias,
    bool has_weight,
    bool has_bias,
    at::Tensor *grad_input,
    at::Tensor *grad_weight,
    at::Tensor *grad_bias,
    bool grad_input_needed,
    bool grad_weight_needed,
    bool grad_bias_needed,
    int64_t norm_axis)
{
    const std::vector<nntile::Index> input_graph =
        pytorch_shape_to_graph(input.sizes());
    const nntile::Index axis = static_cast<nntile::Index>(norm_axis);
    const nntile::Index norm_len =
        input_graph[static_cast<std::size_t>(axis)];
    const std::vector<nntile::Index> reduced_graph =
        reduced_shape_along_axis(input_graph, axis);
    const nntile::Index normalized_ndim =
        static_cast<nntile::Index>(input_graph.size()) - axis;

    auto *grad_out_node = get_or_create_data_node(
        grad_out,
        input_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(grad_out));
    auto *input_node = get_or_create_data_node(
        input,
        input_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(input));
    auto *mean_node = get_or_create_data_node(
        mean,
        reduced_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(mean));
    auto *rstd_node = get_or_create_data_node(
        rstd,
        reduced_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(rstd));

    nntile::TensorGraph::TensorNode *weight_node = nullptr;
    if (has_weight && weight != nullptr)
    {
        weight_node = get_or_create_data_node(
            *weight,
            {norm_len},
            nntile::DataType::FP32,
            mark_as_input_for_operand(*weight));
    }
    nntile::TensorGraph::TensorNode *bias_node = nullptr;
    if (has_bias && bias != nullptr)
    {
        bias_node = get_or_create_data_node(
            *bias,
            {norm_len},
            nntile::DataType::FP32,
            mark_as_input_for_operand(*bias));
    }

    nntile::TensorGraph::TensorNode *gi_node = nullptr;
    nntile::TensorGraph::TensorNode *gw_node = nullptr;
    nntile::TensorGraph::TensorNode *gb_node = nullptr;
    if (grad_input_needed && grad_input != nullptr)
    {
        gi_node = get_or_create_data_node(
            *grad_input,
            input_graph,
            nntile::DataType::FP32,
            false);
    }
    if (grad_weight_needed && grad_weight != nullptr)
    {
        gw_node = get_or_create_data_node(
            *grad_weight,
            {norm_len},
            nntile::DataType::FP32,
            false);
    }
    if (grad_bias_needed && grad_bias != nullptr)
    {
        gb_node = get_or_create_data_node(
            *grad_bias,
            {norm_len},
            nntile::DataType::FP32,
            false);
    }

    nntile::tensor::torch_layer_norm_backward(
        grad_out_node,
        input_node,
        mean_node,
        rstd_node,
        weight_node,
        bias_node,
        gi_node,
        gw_node,
        gb_node,
        normalized_ndim,
        grad_input_needed,
        grad_weight_needed,
        grad_bias_needed);

    if (grad_input_needed && grad_input != nullptr)
    {
        register_data_node(*grad_input, gi_node);
    }
    if (grad_weight_needed && grad_weight != nullptr)
    {
        register_data_node(*grad_weight, gw_node);
    }
    if (grad_bias_needed && grad_bias != nullptr)
    {
        register_data_node(*grad_bias, gb_node);
    }
}

void tensor_rms_norm_forward_fp32(
    const at::Tensor &,
    const at::Tensor *,
    bool,
    at::Tensor &,
    at::Tensor &,
    int64_t,
    float)
{
    throw_op_disabled("rms_norm_forward");
}

void tensor_rms_norm_backward_fp32(
    const at::Tensor &,
    const at::Tensor &,
    const at::Tensor &,
    const at::Tensor *,
    bool,
    at::Tensor *,
    at::Tensor *,
    bool,
    bool,
    int64_t)
{
    throw_op_disabled("rms_norm_backward");
}

void tensor_rope_fp32(
    const at::Tensor &,
    const at::Tensor &,
    const at::Tensor &,
    at::Tensor &)
{
    throw_op_disabled("rope");
}

void tensor_rope_backward_fp32(
    const at::Tensor &,
    const at::Tensor &,
    const at::Tensor &,
    at::Tensor &)
{
    throw_op_disabled("rope_backward");
}

void tensor_mse_loss_fp32(const at::Tensor &, float, at::Tensor &)
{
    throw_op_disabled("mse_loss");
}

void tensor_mse_loss_backward_fp32(const at::Tensor &, float, at::Tensor &)
{
    throw_op_disabled("mse_loss_backward");
}

void tensor_embedding_backward_fp32(
    const at::Tensor &indices,
    const at::Tensor &grad_out,
    at::Tensor &grad_weight,
    nntile::Index /*axis*/,
    int /*redux*/)
{
    const std::vector<nntile::Index> index_graph =
        pytorch_shape_to_graph(indices.sizes());
    const std::vector<nntile::Index> grad_out_graph =
        pytorch_shape_to_graph(grad_out.sizes());
    const std::vector<nntile::Index> weight_graph =
        pytorch_shape_to_graph(grad_weight.sizes());

    auto *index_node = get_or_create_data_node(
        indices,
        index_graph,
        nntile::DataType::INT64,
        mark_as_input_for_operand(indices));
    auto *grad_out_node = get_or_create_data_node(
        grad_out,
        grad_out_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(grad_out));
    auto *grad_weight_node = get_or_create_data_node(
        grad_weight,
        weight_graph,
        nntile::DataType::FP32,
        false);

    nntile::tensor::torch_embedding_dense_backward(
        grad_out_node,
        index_node,
        grad_weight_node);
    register_data_node(grad_weight, grad_weight_node);
}

void tensor_sdpa_backward_fp32(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor *mask,
    const at::Tensor &grad_out,
    at::Tensor &grad_q,
    at::Tensor &grad_k,
    at::Tensor &grad_v,
    int64_t /*batch_ndim*/)
{
    const std::vector<nntile::Index> q_graph =
        pytorch_shape_to_graph(q.sizes());
    const std::vector<nntile::Index> k_graph =
        pytorch_shape_to_graph(k.sizes());
    const std::vector<nntile::Index> v_graph =
        pytorch_shape_to_graph(v.sizes());

    auto *q_node = get_or_create_data_node(
        q,
        q_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(q));
    auto *k_node = get_or_create_data_node(
        k,
        k_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(k));
    auto *v_node = get_or_create_data_node(
        v,
        v_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(v));
    auto *grad_out_node = get_or_create_data_node(
        grad_out,
        q_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(grad_out));

    nntile::TensorGraph::TensorNode *mask_node = nullptr;
    if (mask != nullptr)
    {
        const std::vector<nntile::Index> mask_graph =
            pytorch_shape_to_graph(mask->sizes());
        mask_node = get_or_create_data_node(
            *mask,
            mask_graph,
            nntile::DataType::BOOL,
            mark_as_input_for_operand(*mask));
    }

    auto *grad_q_node = get_or_create_data_node(
        grad_q,
        q_graph,
        nntile::DataType::FP32,
        false);
    auto *grad_k_node = get_or_create_data_node(
        grad_k,
        k_graph,
        nntile::DataType::FP32,
        false);
    auto *grad_v_node = get_or_create_data_node(
        grad_v,
        v_graph,
        nntile::DataType::FP32,
        false);

    nntile::tensor::torch_sdpa_backward(
        q_node,
        k_node,
        v_node,
        grad_out_node,
        mask_node,
        grad_q_node,
        grad_k_node,
        grad_v_node,
        /*is_causal=*/false);
    register_data_node(grad_q, grad_q_node);
    register_data_node(grad_k, grad_k_node);
    register_data_node(grad_v, grad_v_node);
}

void tensor_log_softmax_fp32(
    const at::Tensor &input,
    at::Tensor &out,
    int64_t dim)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(input.sizes());
    auto *in_node = get_or_create_data_node(
        input,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(input));
    nntile::starpu::TorchDispatchArgs extra;
    extra.iargs[0] = static_cast<nntile::Index>(dim);
    auto *out_node = nntile::tensor::torch_unary(
        nntile::starpu::TorchKind::LogSoftmax,
        in_node,
        graph_shape,
        extra);
    register_data_node(out, out_node);
}

void tensor_log_softmax_backward_fp32(
    const at::Tensor &grad_output,
    const at::Tensor &output,
    at::Tensor &grad_input,
    int64_t dim)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(output.sizes());
    auto *grad_out_node = get_or_create_data_node(
        grad_output,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(grad_output));
    auto *out_node = get_or_create_data_node(
        output,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(output));
    auto *grad_in_node = get_or_create_data_node(
        grad_input,
        graph_shape,
        nntile::DataType::FP32,
        false);
    nntile::starpu::TorchDispatchArgs extra;
    extra.iargs[0] = static_cast<nntile::Index>(dim);
    nntile::tensor::torch_binary(
        nntile::starpu::TorchKind::LogSoftmaxBackward,
        grad_out_node,
        out_node,
        grad_in_node,
        extra);
    register_data_node(grad_input, grad_in_node);
}

void tensor_nll_loss_forward_fp32(
    const at::Tensor &log_probs,
    const at::Tensor &target,
    at::Tensor &loss,
    at::Tensor &total_weight,
    int64_t reduction,
    int64_t ignore_index)
{
    const std::vector<nntile::Index> lp_graph =
        pytorch_shape_to_graph(log_probs.sizes());
    const std::vector<nntile::Index> tgt_graph =
        pytorch_shape_to_graph(target.sizes());
    const std::vector<nntile::Index> loss_graph =
        pytorch_shape_to_graph(loss.sizes());
    const std::vector<nntile::Index> tw_graph =
        pytorch_shape_to_graph(total_weight.sizes());

    auto *lp_node = get_or_create_data_node(
        log_probs,
        lp_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(log_probs));
    auto *tgt_node = get_or_create_data_node(
        target,
        tgt_graph,
        nntile::DataType::INT64,
        mark_as_input_for_operand(target));
    auto *loss_node = get_or_create_data_node(
        loss,
        loss_graph,
        nntile::DataType::FP32,
        false);
    auto *tw_node = get_or_create_data_node(
        total_weight,
        tw_graph,
        nntile::DataType::FP32,
        false);

    nntile::tensor::torch_nll_loss_forward(
        lp_node,
        tgt_node,
        loss_node,
        tw_node,
        static_cast<nntile::Index>(reduction),
        static_cast<nntile::Index>(ignore_index));
    register_data_node(loss, loss_node);
    register_data_node(total_weight, tw_node);
}

void tensor_nll_loss_backward_fp32(
    const at::Tensor &grad_output,
    const at::Tensor &log_probs,
    const at::Tensor &target,
    const at::Tensor &total_weight,
    at::Tensor &grad_input,
    int64_t reduction,
    int64_t ignore_index)
{
    const std::vector<nntile::Index> go_graph =
        pytorch_shape_to_graph(grad_output.sizes());
    const std::vector<nntile::Index> lp_graph =
        pytorch_shape_to_graph(log_probs.sizes());
    const std::vector<nntile::Index> tgt_graph =
        pytorch_shape_to_graph(target.sizes());
    const std::vector<nntile::Index> tw_graph =
        pytorch_shape_to_graph(total_weight.sizes());
    const std::vector<nntile::Index> gi_graph =
        pytorch_shape_to_graph(grad_input.sizes());

    auto *go_node = get_or_create_data_node(
        grad_output,
        go_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(grad_output));
    // TorchNllLossBackward reads grad_output with STARPU_R. If autograd
    // seeded a metadata scalar without a producer write, materialize it.
    if (is_metadata_only_tensor(grad_output) &&
        !go_node->has_producer())
    {
        const nntile::Scalar val = go_node->has_constant_value()
            ? go_node->constant_value()
            : static_cast<nntile::Scalar>(1.0);
        nntile::tensor::fill(val, go_node);
    }
    auto *lp_node = get_or_create_data_node(
        log_probs,
        lp_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(log_probs));
    auto *tgt_node = get_or_create_data_node(
        target,
        tgt_graph,
        nntile::DataType::INT64,
        mark_as_input_for_operand(target));
    auto *tw_node = get_or_create_data_node(
        total_weight,
        tw_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(total_weight));
    auto *gi_node = get_or_create_data_node(
        grad_input,
        gi_graph,
        nntile::DataType::FP32,
        false);

    nntile::tensor::torch_nll_loss_backward(
        go_node,
        lp_node,
        tgt_node,
        tw_node,
        gi_node,
        static_cast<nntile::Index>(reduction),
        static_cast<nntile::Index>(ignore_index));
    register_data_node(grad_input, gi_node);
}

} // namespace torch_nntile
