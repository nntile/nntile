/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_executor.cpp
 */

#include "nntile_executor.h"

#include "nntile_gemm_layout.h"
#include "nntile_graph_recorder.h"
#include "nntile_graph_recorder_impl.h"
#include "nntile_tensor_gc.h"

#include <ATen/Tensor.h>
#include <c10/util/Exception.h>

#ifdef TORCH_NNTILE_USE_LIBNNTILE

#include <nntile/base_types.hh>
#include <nntile/tensor/ops/add.hh>
#include <nntile/tensor/ops/add_fiber_inplace.hh>
#include <nntile/tensor/ops/add_inplace.hh>
#include <nntile/tensor/ops/add_slice.hh>
#include <nntile/tensor/ops/add_slice_inplace.hh>
#include <nntile/tensor/ops/concat.hh>
#include <nntile/tensor/ops/copy_intersection.hh>
#include <nntile/tensor/ops/embedding.hh>
#include <nntile/tensor/ops/embedding_backward.hh>
#include <nntile/tensor/ops/multiply.hh>
#include <nntile/tensor/ops/multiply_inplace.hh>
#include <nntile/tensor/ops/clear.hh>
#include <nntile/tensor/ops/copy.hh>
#include <nntile/tensor/ops/gemm.hh>
#include <nntile/tensor/ops/hypot.hh>
#include <nntile/tensor/ops/hypot_scalar_inverse.hh>
#include <nntile/tensor/ops/logsumexp.hh>
#include <nntile/tensor/ops/mask_scalar.hh>
#include <nntile/tensor/ops/maxsumexp.hh>
#include <nntile/tensor/ops/gelu.hh>
#include <nntile/tensor/ops/gelu_backward.hh>
#include <nntile/tensor/ops/gelu_inplace.hh>
#include <nntile/tensor/ops/gelutanh.hh>
#include <nntile/tensor/ops/gelutanh_backward.hh>
#include <nntile/tensor/ops/gelutanh_inplace.hh>
#include <nntile/tensor/ops/multiply_fiber.hh>
#include <nntile/tensor/ops/multiply_slice.hh>
#include <nntile/tensor/ops/norm.hh>
#include <nntile/tensor/ops/norm_slice.hh>
#include <nntile/tensor/ops/norm_slice_inplace.hh>
#include <nntile/tensor/ops/relu.hh>
#include <nntile/tensor/ops/relu_backward.hh>
#include <nntile/tensor/ops/adam_step.hh>
#include <nntile/tensor/ops/adamw_step.hh>
#include <nntile/tensor/ops/silu.hh>
#include <nntile/tensor/ops/silu_backward.hh>
#include <nntile/tensor/ops/silu_inplace.hh>
#include <nntile/tensor/ops/sgd_step.hh>
#include <nntile/tensor/ops/scale.hh>
#include <nntile/tensor/ops/scale_slice.hh>
#include <nntile/tensor/ops/softmax.hh>
#include <nntile/tensor/ops/softmax_inplace.hh>
#include <nntile/tensor/ops/subtract_indexed_outputs.hh>
#include <nntile/tensor/ops/sum_fiber.hh>
#include <nntile/tensor/ops/sum_slice.hh>
#include <nntile/tensor/ops/sumprod_fiber.hh>
#include <nntile/tensor/ops/sumprod_slice.hh>
#include <nntile/tensor/ops/total_sum_accum.hh>
#include <nntile/tensor/ops/transpose.hh>
#include <nntile/tensor/ops/swap_two_axes.hh>
#include <nntile/core/swap_two_axes_decompose.hh>

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
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
    if (is_staged_input_tensor(tensor))
    {
        return true;
    }
    if (tensor.device().is_cpu())
    {
        return true;
    }
    return false;
}

} // namespace

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

    auto *out_node = nntile::tensor::gemm(
        a_node,
        b_node,
        static_cast<nntile::Scalar>(params.alpha),
        params.trans_a,
        params.trans_b,
        static_cast<nntile::Index>(params.ndim),
        static_cast<nntile::Index>(params.batch_ndim))->set_name("out");
    register_data_node(out, out_node);
    maybe_execute_after_record();
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

    nntile::tensor::gemm(
        a_node,
        b_node,
        c_node,
        static_cast<nntile::Scalar>(params.alpha),
        static_cast<nntile::Scalar>(params.beta),
        params.trans_a,
        params.trans_b,
        static_cast<nntile::Index>(params.ndim),
        static_cast<nntile::Index>(params.batch_ndim));
    register_data_node(out, c_node);
    maybe_execute_after_record();
}

void tensor_add_fp32(
    float alpha,
    const at::Tensor &x,
    float beta,
    const at::Tensor &y,
    at::Tensor &out)
{
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

    auto *z_node = nntile::tensor::add(
        static_cast<nntile::Scalar>(alpha),
        x_node,
        static_cast<nntile::Scalar>(beta),
        y_node)->set_name("z");
    register_data_node(out, z_node);
    maybe_execute_after_record();
}

void tensor_model_transpose_forward_fp32(
    const at::Tensor &src,
    at::Tensor &dst,
    int64_t model_ndim)
{
    const std::vector<nntile::Index> src_graph =
        pytorch_shape_to_graph(src.sizes());
    const std::vector<nntile::Index> dst_graph =
        pytorch_shape_to_graph(dst.sizes());
    const nntile::Index n = static_cast<nntile::Index>(src_graph.size());
    TORCH_CHECK(
        model_ndim > 0 && model_ndim < static_cast<int64_t>(n),
        "nntile model_transpose: invalid model_ndim");
    const nntile::Index tensor_ndim =
        n - static_cast<nntile::Index>(model_ndim);

    auto *src_node = get_or_create_data_node(
        src,
        src_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(src));
    auto *dst_node = get_or_create_data_node(
        dst,
        dst_graph,
        nntile::DataType::FP32,
        false);

    nntile::tensor::transpose(
        static_cast<nntile::Scalar>(1.0),
        src_node,
        dst_node,
        tensor_ndim);
    register_data_node(dst, dst_node);
    maybe_execute_after_record();
}

void tensor_model_transpose_backward_fp32(
    const at::Tensor &grad_out,
    at::Tensor &grad_src,
    int64_t model_ndim)
{
    const std::vector<nntile::Index> grad_out_graph =
        pytorch_shape_to_graph(grad_out.sizes());
    const std::vector<nntile::Index> grad_src_graph =
        pytorch_shape_to_graph(grad_src.sizes());
    const nntile::Index n =
        static_cast<nntile::Index>(grad_out_graph.size());
    TORCH_CHECK(
        model_ndim > 0 && model_ndim < static_cast<int64_t>(n),
        "nntile model_transpose backward: invalid model_ndim");

    auto *grad_out_node = get_or_create_data_node(
        grad_out,
        grad_out_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(grad_out));
    auto *grad_src_node = get_or_create_data_node(
        grad_src,
        grad_src_graph,
        nntile::DataType::FP32,
        false);

    nntile::tensor::transpose(
        static_cast<nntile::Scalar>(1.0),
        grad_out_node,
        grad_src_node,
        static_cast<nntile::Index>(model_ndim));
    register_data_node(grad_src, grad_src_node);
    maybe_execute_after_record();
}

void tensor_swap_two_axes_fp32(
    const at::Tensor &src,
    at::Tensor &dst,
    int64_t dim0,
    int64_t dim1)
{
    const std::vector<nntile::Index> src_graph =
        pytorch_shape_to_graph(src.sizes());
    const nntile::Index n = static_cast<nntile::Index>(src_graph.size());
    nntile::Index d0 = static_cast<nntile::Index>(dim0);
    nntile::Index d1 = static_cast<nntile::Index>(dim1);
    if (d0 < 0)
    {
        d0 += n;
    }
    if (d1 < 0)
    {
        d1 += n;
    }
    TORCH_CHECK(
        d0 >= 0 && d0 < n && d1 >= 0 && d1 < n,
        "nntile swap_two_axes: axis out of range");
    TORCH_CHECK(d0 != d1, "nntile swap_two_axes: axes must differ");

    const nntile::core::SwapTwoAxesDecomposition decomp =
        nntile::core::decompose_swap_axes(src_graph, d0, d1);
    const std::vector<nntile::Index> &dst_graph = decomp.output_shape;

    auto *src_node = get_or_create_data_node(
        src,
        src_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(src));
    auto *dst_node = get_or_create_data_node(
        dst,
        dst_graph,
        nntile::DataType::FP32,
        false);

    nntile::tensor::swap_two_axes(src_node, dst_node, d0, d1);
    register_data_node(dst, dst_node);
    maybe_execute_after_record();
}

void tensor_add_inplace_fp32(
    float alpha,
    const at::Tensor &other,
    float beta,
    at::Tensor &self)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(self.sizes());

    auto *other_node = get_or_create_data_node(
        other,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(other));
    auto *self_node = get_or_create_data_node(
        self,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(self));

    nntile::tensor::add_inplace(
        static_cast<nntile::Scalar>(alpha),
        other_node,
        static_cast<nntile::Scalar>(beta),
        self_node);
    register_data_node(self, self_node);
    maybe_execute_after_record();
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

    auto *out_node = nntile::tensor::multiply(
        self_node,
        other_node,
        static_cast<nntile::Scalar>(1.0))->set_name("z");
    register_data_node(out, out_node);
    maybe_execute_after_record();
}

void tensor_mul_inplace_fp32(const at::Tensor &other, at::Tensor &self)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(self.sizes());

    auto *other_node = get_or_create_data_node(
        other,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(other));
    auto *self_node = get_or_create_data_node(
        self,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(self));

    nntile::tensor::multiply_inplace(
        static_cast<nntile::Scalar>(1.0),
        other_node,
        self_node);
    register_data_node(self, self_node);
    maybe_execute_after_record();
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

    auto *out_node = nntile::tensor::hypot(
        static_cast<nntile::Scalar>(1.0),
        self_node,
        static_cast<nntile::Scalar>(1.0),
        other_node)->set_name("hypot_out");
    register_data_node(out, out_node);
    maybe_execute_after_record();
}

void tensor_linear_fp32(
    const at::Tensor &input,
    const at::Tensor &weight,
    at::Tensor &out)
{
    const PreparedGemmOperands prepared = prepare_linear_operands(input, weight);
    tensor_gemm_fp32(
        prepared.params,
        prepared.a,
        prepared.a_gemm_shape,
        prepared.b,
        prepared.b_gemm_shape,
        out,
        prepared.out_shape);
}

void tensor_relu_fp32(const at::Tensor &input, at::Tensor &out)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(input.sizes());

    auto *src_node = get_or_create_data_node(
        input,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(input));

    auto *dst_node = nntile::tensor::relu(src_node)->set_name("dst");
    push_relu_preactivation_node(src_node);
    register_data_node(out, dst_node);
    maybe_execute_after_record();
}

void tensor_relu_backward_fp32(
    const at::Tensor &x,
    const at::Tensor &dy,
    at::Tensor &dx)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(x.sizes());

    nntile::TensorGraph::TensorNode *x_node =
        lookup_data_node(x, graph_shape);
    if (x_node == nullptr)
    {
        x_node = pop_relu_preactivation_node(graph_shape);
    }
    if (x_node == nullptr)
    {
        x_node = get_or_create_data_node(
            x,
            graph_shape,
            nntile::DataType::FP32,
            mark_as_input_for_operand(x));
    }
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

    nntile::tensor::clear(dx_node);
    nntile::tensor::relu_backward(x_node, dy_node, dx_node);
    register_data_node(dx, dx_node);
    maybe_execute_after_record();
}

void tensor_silu_fp32(const at::Tensor &input, at::Tensor &out)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(input.sizes());

    auto *src_node = get_or_create_data_node(
        input,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(input));

    auto *dst_node = nntile::tensor::silu(src_node)->set_name("dst");
    register_data_node(out, dst_node);
    maybe_execute_after_record();
}

void tensor_silu_inplace_fp32(at::Tensor &self)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(self.sizes());

    auto *node = get_or_create_data_node(
        self,
        graph_shape,
        nntile::DataType::FP32,
        true);

    nntile::tensor::silu_inplace(node);
    register_data_node(self, node);
    maybe_execute_after_record();
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

    nntile::tensor::clear(dx_node);
    nntile::tensor::silu_backward(x_node, dy_node, dx_node);
    register_data_node(dx, dx_node);
    maybe_execute_after_record();
}

void tensor_gelu_fp32(
    const at::Tensor &input,
    at::Tensor &out,
    bool approximate_tanh)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(input.sizes());

    auto *src_node = get_or_create_data_node(
        input,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(input));

    nntile::TensorGraph::TensorNode *dst_node = nullptr;
    if (approximate_tanh)
    {
        dst_node = nntile::tensor::gelutanh(src_node)->set_name("dst");
    }
    else
    {
        dst_node = nntile::tensor::gelu(src_node)->set_name("dst");
    }
    register_data_node(out, dst_node);
    maybe_execute_after_record();
}

void tensor_gelu_inplace_fp32(at::Tensor &self, bool approximate_tanh)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(self.sizes());

    auto *node = get_or_create_data_node(
        self,
        graph_shape,
        nntile::DataType::FP32,
        true);

    if (approximate_tanh)
    {
        nntile::tensor::gelutanh_inplace(node);
    }
    else
    {
        nntile::tensor::gelu_inplace(node);
    }
    register_data_node(self, node);
    maybe_execute_after_record();
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

    nntile::tensor::clear(dx_node);
    if (approximate_tanh)
    {
        nntile::tensor::gelutanh_backward(x_node, dy_node, dx_node);
    }
    else
    {
        nntile::tensor::gelu_backward(x_node, dy_node, dx_node);
    }
    register_data_node(dx, dx_node);
    maybe_execute_after_record();
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

void tensor_linear_backward_input_fp32(
    const at::Tensor &grad_out,
    const at::Tensor &weight,
    at::Tensor &grad_input)
{
    const PreparedGemmOperands forward = prepare_linear_operands(grad_input, weight);
    const GemmParams params = infer_linear_backward_grad_input_params(forward.params);
    const GemmMatrixLayout grad_out_layout = analyze_matrix_layout_for_nntile(grad_out);
    TORCH_CHECK(
        !grad_out_layout.needs_copy,
        "nntile linear_backward_input: grad_out must be contiguous or "
        "row/column-contiguous");
    const at::Tensor &grad_out_prepared = grad_out;
    tensor_gemm_fp32(
        params,
        grad_out_prepared,
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
    const PreparedGemmOperands forward = prepare_linear_operands(input, grad_weight);
    const GemmParams params = infer_linear_backward_grad_weight_params(forward.params);
    const GemmMatrixLayout grad_out_layout = analyze_matrix_layout_for_nntile(grad_out);
    TORCH_CHECK(
        !grad_out_layout.needs_copy,
        "nntile linear_backward_weight: grad_out must be contiguous or "
        "row/column-contiguous");
    TORCH_CHECK(
        forward.a.is_contiguous(),
        "nntile linear_backward_weight: input must be contiguous");
    const at::Tensor &grad_out_prepared = grad_out;
    const at::Tensor &input_prepared = forward.a;
    tensor_gemm_fp32(
        params,
        grad_out_prepared,
        grad_out_layout.gemm_shape,
        input_prepared,
        forward.a_gemm_shape,
        grad_weight,
        forward.b_gemm_shape);
}

namespace
{

constexpr int kRedux = 0;

std::vector<nntile::Index> maxsumexp_graph_shape(
    const std::vector<nntile::Index> &input_graph,
    nntile::Index axis)
{
    std::vector<nntile::Index> maxsumexp_shape;
    maxsumexp_shape.reserve(input_graph.size());
    for (nntile::Index i = 0; i < static_cast<nntile::Index>(input_graph.size());
         ++i)
    {
        if (i != axis)
        {
            maxsumexp_shape.push_back(input_graph[static_cast<std::size_t>(i)]);
        }
    }
    maxsumexp_shape.push_back(2);
    return maxsumexp_shape;
}

nntile::Index class_graph_axis(c10::IntArrayRef pytorch_logits_shape)
{
    return static_cast<nntile::Index>(pytorch_logits_shape.size()) - 1;
}

float cross_entropy_scale(
    const std::int64_t *labels_data,
    c10::IntArrayRef labels_shape,
    std::int64_t ignore_index,
    bool mean_reduction)
{
    if (!mean_reduction)
    {
        return 1.0f;
    }
    nntile::Index count = 0;
    nntile::Index total = 1;
    for (const auto dim : labels_shape)
    {
        total *= static_cast<nntile::Index>(dim);
    }
    for (nntile::Index i = 0; i < total; ++i)
    {
        if (labels_data[i] != ignore_index)
        {
            ++count;
        }
    }
    if (count <= 0)
    {
        count = 1;
    }
    return 1.0f / static_cast<float>(count);
}

} // namespace

void tensor_cross_entropy_forward_fp32(
    const at::Tensor &logits,
    const at::Tensor &labels,
    std::int64_t ignore_index,
    bool mean_reduction,
    at::Tensor &loss)
{
    const std::vector<nntile::Index> logits_graph =
        pytorch_shape_to_graph(logits.sizes());
    const std::vector<nntile::Index> labels_graph =
        pytorch_shape_to_graph(labels.sizes());
    const nntile::Index class_axis = class_graph_axis(logits.sizes());
    const std::vector<nntile::Index> maxsumexp_graph =
        maxsumexp_graph_shape(logits_graph, class_axis);
    const float scale = cross_entropy_scale(
        labels.data_ptr<std::int64_t>(),
        labels.sizes(),
        ignore_index,
        mean_reduction);

    auto *logits_node = get_or_create_data_node(
        logits,
        logits_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(logits));
    auto *labels_node = get_or_create_data_node(
        labels,
        labels_graph,
        nntile::DataType::INT64,
        mark_as_input_for_operand(labels));
    auto *loss_node = get_or_create_data_node(
        loss,
        {},
        nntile::DataType::FP32,
        false);

    auto &graph = *logits_node->graph();
    auto *maxsumexp_node =
        graph.data(maxsumexp_graph, nntile::DataType::FP32)
            ->set_name("maxsumexp");
    auto *logsumexp_node =
        graph.data(labels_graph, nntile::DataType::FP32)->set_name("logsumexp");

    nntile::tensor::clear(maxsumexp_node);
    nntile::tensor::maxsumexp(
        logits_node,
        maxsumexp_node,
        class_axis,
        kRedux);
    nntile::tensor::logsumexp(maxsumexp_node, logsumexp_node);
    nntile::tensor::clear(loss_node);
    nntile::tensor::total_sum_accum(
        static_cast<nntile::Scalar>(scale),
        logsumexp_node,
        logits_node,
        labels_node,
        loss_node,
        static_cast<nntile::Index>(ignore_index));

    register_data_node(loss, loss_node);
    maybe_execute_after_record();
}

void tensor_cross_entropy_backward_fp32(
    const at::Tensor &logits,
    const at::Tensor &labels,
    const at::Tensor &grad_output,
    at::Tensor &grad_row,
    at::Tensor &grad_logits,
    std::int64_t ignore_index,
    bool mean_reduction)
{
    const std::vector<nntile::Index> logits_graph =
        pytorch_shape_to_graph(logits.sizes());
    const std::vector<nntile::Index> labels_graph =
        pytorch_shape_to_graph(labels.sizes());
    const nntile::Index class_axis = class_graph_axis(logits.sizes());
    const std::vector<nntile::Index> maxsumexp_graph =
        maxsumexp_graph_shape(logits_graph, class_axis);
    const float ce_scale = cross_entropy_scale(
        labels.data_ptr<std::int64_t>(),
        labels.sizes(),
        ignore_index,
        mean_reduction);

    auto broadcast_grad_output_to_row = [&](
        nntile::TensorGraph::TensorNode *grad_output_node,
        nntile::TensorGraph::TensorNode *grad_row_node,
        nntile::TensorGraph &graph,
        const std::vector<nntile::Index> &labels_graph_shape)
    {
        nntile::TensorGraph::TensorNode *src_node = grad_output_node;
        for (std::size_t dim = 0; dim < labels_graph_shape.size(); ++dim)
        {
            nntile::TensorGraph::TensorNode *dst_node = grad_row_node;
            if (dim + 1 < labels_graph_shape.size())
            {
                std::vector<nntile::Index> dst_shape(
                    labels_graph_shape.begin(),
                    labels_graph_shape.begin() +
                        static_cast<std::ptrdiff_t>(dim) + 1);
                dst_node = graph.data(dst_shape, nntile::DataType::FP32)
                               ->set_name("grad_output_broadcast");
                track_graph_node(dst_node);
            }
            nntile::tensor::scale_slice(
                static_cast<nntile::Scalar>(1.0),
                src_node,
                dst_node,
                static_cast<nntile::Index>(dim));
            src_node = dst_node;
        }
    };

    auto *logits_node = get_or_create_data_node(
        logits,
        logits_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(logits));
    auto *labels_node = get_or_create_data_node(
        labels,
        labels_graph,
        nntile::DataType::INT64,
        mark_as_input_for_operand(labels));
    auto *grad_output_node = get_or_create_data_node(
        grad_output,
        {},
        nntile::DataType::FP32,
        mark_as_input_for_operand(grad_output));
    auto *grad_row_node = get_or_create_data_node(
        grad_row,
        labels_graph,
        nntile::DataType::FP32,
        false);
    auto *grad_logits_node = get_or_create_data_node(
        grad_logits,
        logits_graph,
        nntile::DataType::FP32,
        false);

    auto &graph = *logits_node->graph();
    auto *maxsumexp_node =
        graph.data(maxsumexp_graph, nntile::DataType::FP32)
            ->set_name("maxsumexp");

    // Broadcast scalar grad_output to labels shape via chained scale_slice.
    broadcast_grad_output_to_row(
        grad_output_node,
        grad_row_node,
        graph,
        labels_graph);

    nntile::tensor::clear(maxsumexp_node);
    nntile::tensor::maxsumexp(
        logits_node,
        maxsumexp_node,
        class_axis,
        kRedux);
    nntile::tensor::clear(grad_logits_node);
    nntile::tensor::softmax(
        maxsumexp_node,
        logits_node,
        grad_logits_node,
        static_cast<nntile::Scalar>(ce_scale),
        class_axis);
    nntile::tensor::subtract_indexed_outputs(
        static_cast<nntile::Scalar>(ce_scale),
        labels_node,
        grad_logits_node,
        static_cast<nntile::Index>(ignore_index));
    nntile::tensor::multiply_slice(
        static_cast<nntile::Scalar>(1.0),
        grad_row_node,
        grad_logits_node,
        class_axis);

    register_data_node(grad_logits, grad_logits_node);
    maybe_execute_after_record();
}

void tensor_softmax_fp32(
    const at::Tensor &input,
    at::Tensor &out,
    int64_t dim)
{
    const std::vector<nntile::Index> input_graph =
        pytorch_shape_to_graph(input.sizes());
    const nntile::Index axis = static_cast<nntile::Index>(dim);
    const std::vector<nntile::Index> maxsumexp_graph =
        maxsumexp_graph_shape(input_graph, axis);

    auto *src_node = get_or_create_data_node(
        input,
        input_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(input));
    auto *dst_node = get_or_create_data_node(
        out,
        input_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(out));

    auto &graph = *src_node->graph();
    auto *maxsumexp_node =
        graph.data(maxsumexp_graph, nntile::DataType::FP32)
            ->set_name("maxsumexp");

    nntile::tensor::clear(maxsumexp_node);
    nntile::tensor::maxsumexp(
        src_node,
        maxsumexp_node,
        axis,
        kRedux);
    nntile::tensor::softmax(
        maxsumexp_node,
        src_node,
        dst_node,
        static_cast<nntile::Scalar>(1.0),
        axis);

    register_data_node(out, dst_node);
    maybe_execute_after_record();
}

void tensor_sgd_step_fp32(
    int64_t num_iter,
    float momentum,
    float lr,
    float weight_decay,
    float dampening,
    bool nesterov,
    const at::Tensor &grad,
    at::Tensor &velocity,
    at::Tensor &param)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(grad.sizes());

    nntile::TensorGraph::TensorNode *grad_node =
        lookup_param_grad_node(param);
    if (grad_node == nullptr)
    {
        grad_node = lookup_data_node(grad, graph_shape);
    }
    if (grad_node == nullptr)
    {
        grad_node = get_or_create_data_node(
            grad,
            graph_shape,
            nntile::DataType::FP32,
            mark_as_input_for_operand(grad));
    }
    TORCH_CHECK(
        grad_node != nullptr,
        "nntile sgd_step: parameter grad is not registered in the graph; "
        "run backward before the optimizer step");
    at::Tensor mutable_grad = grad;
    register_grad_alias_for_host_copy(mutable_grad, grad_node);
    auto *velocity_node = get_or_create_data_node(
        velocity,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(velocity));
    auto *param_node = get_or_create_data_node(
        param,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(param));

    nntile::tensor::sgd_step(
        static_cast<nntile::Index>(num_iter),
        static_cast<nntile::Scalar>(momentum),
        static_cast<nntile::Scalar>(lr),
        static_cast<nntile::Scalar>(weight_decay),
        static_cast<nntile::Scalar>(dampening),
        nesterov,
        grad_node,
        velocity_node,
        param_node);

    register_data_node(velocity, velocity_node);
    register_data_node(param, param_node);
    maybe_execute_after_record();
}

namespace
{

constexpr int kNormRedux = 0;
constexpr nntile::Index kBatchNdim = 0;

std::vector<nntile::Index> reduced_shape_along_axis(
    const std::vector<nntile::Index> &input_graph,
    nntile::Index axis)
{
    std::vector<nntile::Index> reduced;
    reduced.reserve(input_graph.size() - 1);
    for (nntile::Index i = 0; i < static_cast<nntile::Index>(input_graph.size());
         ++i)
    {
        if (i != axis)
        {
            reduced.push_back(input_graph[static_cast<std::size_t>(i)]);
        }
    }
    return reduced;
}

std::vector<nntile::Index> keepdim_shape_along_axis(
    const std::vector<nntile::Index> &input_graph,
    nntile::Index axis)
{
    std::vector<nntile::Index> keepdim = input_graph;
    keepdim[static_cast<std::size_t>(axis)] = 1;
    return keepdim;
}

nntile::TensorGraph::TensorNode *make_graph_tensor(
    nntile::TensorGraph &graph,
    const std::vector<nntile::Index> &shape,
    const char *name)
{
    auto *node = graph.data(shape, nntile::DataType::FP32)->set_name(name);
    track_graph_node(node);
    return node;
}

void broadcast_slice_to_keepdim(
    nntile::TensorGraph::TensorNode *slice_node,
    nntile::TensorGraph::TensorNode *keepdim_node,
    nntile::Index axis)
{
    nntile::tensor::clear(keepdim_node);
    nntile::tensor::scale_slice(
        static_cast<nntile::Scalar>(1.0),
        slice_node,
        keepdim_node,
        axis);
}

} // namespace

void tensor_softmax_backward_fp32(
    const at::Tensor &grad_output,
    const at::Tensor &output,
    at::Tensor &grad_input,
    int64_t dim)
{
    const std::vector<nntile::Index> input_graph =
        pytorch_shape_to_graph(output.sizes());
    const nntile::Index axis = static_cast<nntile::Index>(dim);
    const std::vector<nntile::Index> reduced_graph =
        reduced_shape_along_axis(input_graph, axis);

    auto *grad_output_node = get_or_create_data_node(
        grad_output,
        input_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(grad_output));
    auto *output_node = get_or_create_data_node(
        output,
        input_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(output));
    auto *grad_input_node = get_or_create_data_node(
        grad_input,
        input_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(grad_input));

    nntile::TensorGraph &graph = *output_node->graph();
    auto *sumprod_buf = make_graph_tensor(graph, reduced_graph, "sumprod_buf");
    auto *grad_temp = make_graph_tensor(graph, input_graph, "grad_temp");

    nntile::tensor::sumprod_slice(
        output_node,
        grad_output_node,
        sumprod_buf,
        axis,
        kNormRedux,
        static_cast<nntile::Scalar>(1.0),
        static_cast<nntile::Scalar>(0.0));
    nntile::tensor::add_slice(
        static_cast<nntile::Scalar>(-1.0),
        sumprod_buf,
        static_cast<nntile::Scalar>(1.0),
        grad_output_node,
        grad_temp,
        axis);
    nntile::tensor::multiply_inplace(
        static_cast<nntile::Scalar>(1.0),
        output_node,
        grad_temp);
    nntile::tensor::copy(grad_temp, grad_input_node);

    register_data_node(grad_input, grad_input_node);
    maybe_execute_after_record();
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
    const nntile::Index norm_len = input_graph[static_cast<std::size_t>(axis)];
    const float inv_l =
        1.0f / static_cast<float>(static_cast<std::int64_t>(norm_len));
    const float inv_sqrt_l =
        1.0f / std::sqrt(static_cast<float>(static_cast<std::int64_t>(norm_len)));
    const float eps_sqrt = std::sqrt(eps);
    const std::vector<nntile::Index> reduced_graph =
        reduced_shape_along_axis(input_graph, axis);
    const std::vector<nntile::Index> keepdim_graph =
        keepdim_shape_along_axis(input_graph, axis);

    auto *input_node = get_or_create_data_node(
        input,
        input_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(input));
    nntile::TensorGraph &graph = *input_node->graph();

    auto *mean_reduced = make_graph_tensor(graph, reduced_graph, "mean_red");
    nntile::tensor::sum_slice(
        input_node,
        mean_reduced,
        axis,
        kNormRedux,
        static_cast<nntile::Scalar>(inv_l),
        static_cast<nntile::Scalar>(0.0));

    auto *mean_node = get_or_create_data_node(
        mean,
        keepdim_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(mean));
    broadcast_slice_to_keepdim(mean_reduced, mean_node, axis);

    auto *centered = nntile::tensor::add_slice(
        static_cast<nntile::Scalar>(-1.0),
        mean_reduced,
        static_cast<nntile::Scalar>(1.0),
        input_node,
        axis);

    auto *rstd_reduced = make_graph_tensor(graph, reduced_graph, "rstd_red");
    nntile::tensor::norm_slice_inplace(
        static_cast<nntile::Scalar>(inv_sqrt_l),
        centered,
        static_cast<nntile::Scalar>(0.0),
        rstd_reduced,
        axis,
        kNormRedux);
    nntile::tensor::hypot_scalar_inverse(
        static_cast<nntile::Scalar>(eps_sqrt),
        static_cast<nntile::Scalar>(1.0),
        rstd_reduced);

    auto *rstd_node = get_or_create_data_node(
        rstd,
        keepdim_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(rstd));
    broadcast_slice_to_keepdim(rstd_reduced, rstd_node, axis);

    nntile::tensor::multiply_slice(
        static_cast<nntile::Scalar>(1.0),
        rstd_reduced,
        centered,
        axis);

    nntile::TensorGraph::TensorNode *scaled = centered;
    if (has_weight)
    {
        auto *weight_node = get_or_create_data_node(
            *weight,
            {norm_len},
            nntile::DataType::FP32,
            mark_as_input_for_operand(*weight));
        scaled = nntile::tensor::multiply_fiber(
            static_cast<nntile::Scalar>(1.0),
            weight_node,
            centered,
            axis);
    }

    auto *output_node = get_or_create_data_node(
        output,
        input_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(output));
    if (has_bias)
    {
        auto *bias_node = get_or_create_data_node(
            *bias,
            {norm_len},
            nntile::DataType::FP32,
            mark_as_input_for_operand(*bias));
        nntile::tensor::copy(scaled, output_node);
        nntile::tensor::add_fiber_inplace(
            static_cast<nntile::Scalar>(1.0),
            bias_node,
            static_cast<nntile::Scalar>(1.0),
            output_node,
            axis,
            kBatchNdim);
    }
    else
    {
        nntile::tensor::copy(scaled, output_node);
    }

    register_data_node(output, output_node);
    register_data_node(mean, mean_node);
    register_data_node(rstd, rstd_node);
    maybe_execute_after_record();
}

void tensor_layer_norm_backward_fp32(
    const at::Tensor &grad_out,
    const at::Tensor &input,
    const at::Tensor &mean,
    const at::Tensor &rstd,
    const at::Tensor *weight,
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
    const nntile::Index norm_len = input_graph[static_cast<std::size_t>(axis)];
    const float inv_l =
        1.0f / static_cast<float>(static_cast<std::int64_t>(norm_len));
    const std::vector<nntile::Index> reduced_graph =
        reduced_shape_along_axis(input_graph, axis);

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
    nntile::TensorGraph &graph = *grad_out_node->graph();

    auto *x_hat = nntile::tensor::add_slice(
        static_cast<nntile::Scalar>(-1.0),
        mean_node,
        static_cast<nntile::Scalar>(1.0),
        input_node,
        axis);
    nntile::tensor::multiply_slice(
        static_cast<nntile::Scalar>(1.0),
        rstd_node,
        x_hat,
        axis);

    if (grad_bias_needed && grad_bias != nullptr)
    {
        auto *grad_bias_node = get_or_create_data_node(
            *grad_bias,
            {norm_len},
            nntile::DataType::FP32,
            mark_as_input_for_operand(*grad_bias));
        nntile::tensor::clear(grad_bias_node);
        nntile::tensor::sum_fiber(
            grad_out_node,
            grad_bias_node,
            axis,
            kBatchNdim,
            kNormRedux,
            static_cast<nntile::Scalar>(1.0),
            static_cast<nntile::Scalar>(0.0));
        register_data_node(*grad_bias, grad_bias_node);
    }

    nntile::TensorGraph::TensorNode *grad_temp = grad_out_node;
    if (has_weight)
    {
        auto *weight_node = get_or_create_data_node(
            *weight,
            {norm_len},
            nntile::DataType::FP32,
            mark_as_input_for_operand(*weight));
        grad_temp = nntile::tensor::multiply_fiber(
            static_cast<nntile::Scalar>(1.0),
            weight_node,
            grad_out_node,
            axis);
    }

    if (grad_weight_needed && grad_weight != nullptr)
    {
        auto *grad_weight_node = get_or_create_data_node(
            *grad_weight,
            {norm_len},
            nntile::DataType::FP32,
            mark_as_input_for_operand(*grad_weight));
        nntile::tensor::clear(grad_weight_node);
        nntile::tensor::sumprod_fiber(
            grad_out_node,
            x_hat,
            grad_weight_node,
            axis,
            kNormRedux,
            static_cast<nntile::Scalar>(1.0),
            static_cast<nntile::Scalar>(0.0));
        register_data_node(*grad_weight, grad_weight_node);
    }

    if (grad_input_needed && grad_input != nullptr)
    {
        auto *grad_input_node = get_or_create_data_node(
            *grad_input,
            input_graph,
            nntile::DataType::FP32,
            mark_as_input_for_operand(*grad_input));
        auto *mean_buf = make_graph_tensor(graph, reduced_graph, "mean_buf");
        auto *tmp_grad = make_graph_tensor(graph, input_graph, "tmp_grad");

        nntile::tensor::copy(x_hat, tmp_grad);
        nntile::tensor::sumprod_slice(
            grad_temp,
            tmp_grad,
            mean_buf,
            axis,
            kNormRedux,
            static_cast<nntile::Scalar>(-inv_l),
            static_cast<nntile::Scalar>(0.0));
        nntile::tensor::multiply_slice(
            static_cast<nntile::Scalar>(1.0),
            mean_buf,
            tmp_grad,
            axis);
        nntile::tensor::add_inplace(
            static_cast<nntile::Scalar>(1.0),
            grad_temp,
            static_cast<nntile::Scalar>(1.0),
            tmp_grad);
        nntile::tensor::sum_slice(
            grad_temp,
            mean_buf,
            axis,
            kNormRedux,
            static_cast<nntile::Scalar>(inv_l),
            static_cast<nntile::Scalar>(0.0));
        nntile::tensor::add_slice_inplace(
            static_cast<nntile::Scalar>(-1.0),
            mean_buf,
            static_cast<nntile::Scalar>(1.0),
            tmp_grad,
            axis);
        nntile::tensor::multiply_slice(
            static_cast<nntile::Scalar>(1.0),
            rstd_node,
            tmp_grad,
            axis);
        nntile::tensor::copy(tmp_grad, grad_input_node);
        register_data_node(*grad_input, grad_input_node);
    }

    maybe_execute_after_record();
}

void tensor_rms_norm_forward_fp32(
    const at::Tensor &input,
    const at::Tensor *weight,
    bool has_weight,
    at::Tensor &output,
    at::Tensor &rstd,
    int64_t norm_axis,
    float eps)
{
    const std::vector<nntile::Index> input_graph =
        pytorch_shape_to_graph(input.sizes());
    const nntile::Index axis = static_cast<nntile::Index>(norm_axis);
    const nntile::Index norm_len = input_graph[static_cast<std::size_t>(axis)];
    const float inv_sqrt_l =
        1.0f / std::sqrt(static_cast<float>(static_cast<std::int64_t>(norm_len)));
    const float eps_sqrt = std::sqrt(eps);
    const std::vector<nntile::Index> reduced_graph =
        reduced_shape_along_axis(input_graph, axis);
    const std::vector<nntile::Index> keepdim_graph =
        keepdim_shape_along_axis(input_graph, axis);

    auto *input_node = get_or_create_data_node(
        input,
        input_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(input));
    nntile::TensorGraph &graph = *input_node->graph();

    auto *rstd_reduced = make_graph_tensor(graph, reduced_graph, "rstd_red");
    nntile::tensor::norm_slice_inplace(
        static_cast<nntile::Scalar>(inv_sqrt_l),
        input_node,
        static_cast<nntile::Scalar>(0.0),
        rstd_reduced,
        axis,
        kNormRedux);
    nntile::tensor::hypot_scalar_inverse(
        static_cast<nntile::Scalar>(eps_sqrt),
        static_cast<nntile::Scalar>(1.0),
        rstd_reduced);

    auto *rstd_node = get_or_create_data_node(
        rstd,
        keepdim_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(rstd));
    broadcast_slice_to_keepdim(rstd_reduced, rstd_node, axis);

    auto *normalized = nntile::tensor::copy(input_node);
    nntile::tensor::multiply_slice(
        static_cast<nntile::Scalar>(1.0),
        rstd_reduced,
        normalized,
        axis);

    auto *output_node = get_or_create_data_node(
        output,
        input_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(output));
    if (has_weight)
    {
        auto *weight_node = get_or_create_data_node(
            *weight,
            {norm_len},
            nntile::DataType::FP32,
            mark_as_input_for_operand(*weight));
        auto *scaled = nntile::tensor::multiply_fiber(
            static_cast<nntile::Scalar>(1.0),
            weight_node,
            normalized,
            axis);
        nntile::tensor::copy(scaled, output_node);
    }
    else
    {
        nntile::tensor::copy(normalized, output_node);
    }

    register_data_node(output, output_node);
    register_data_node(rstd, rstd_node);
    maybe_execute_after_record();
}

void tensor_rms_norm_backward_fp32(
    const at::Tensor &grad_out,
    const at::Tensor &input,
    const at::Tensor &rstd,
    const at::Tensor *weight,
    bool has_weight,
    at::Tensor *grad_input,
    at::Tensor *grad_weight,
    bool grad_input_needed,
    bool grad_weight_needed,
    int64_t norm_axis)
{
    const std::vector<nntile::Index> input_graph =
        pytorch_shape_to_graph(input.sizes());
    const nntile::Index axis = static_cast<nntile::Index>(norm_axis);
    const nntile::Index norm_len = input_graph[static_cast<std::size_t>(axis)];
    const float inv_l =
        -1.0f / static_cast<float>(static_cast<std::int64_t>(norm_len));
    const std::vector<nntile::Index> reduced_graph =
        reduced_shape_along_axis(input_graph, axis);

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
    auto *rstd_node = get_or_create_data_node(
        rstd,
        reduced_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(rstd));
    nntile::TensorGraph &graph = *grad_out_node->graph();

    auto *normalized = nntile::tensor::copy(input_node);
    nntile::tensor::multiply_slice(
        static_cast<nntile::Scalar>(1.0),
        rstd_node,
        normalized,
        axis);

    if (grad_weight_needed && has_weight && grad_weight != nullptr)
    {
        auto *grad_weight_node = get_or_create_data_node(
            *grad_weight,
            {norm_len},
            nntile::DataType::FP32,
            mark_as_input_for_operand(*grad_weight));
        nntile::tensor::clear(grad_weight_node);
        nntile::tensor::sumprod_fiber(
            grad_out_node,
            normalized,
            grad_weight_node,
            axis,
            kNormRedux,
            static_cast<nntile::Scalar>(1.0),
            static_cast<nntile::Scalar>(0.0));
        register_data_node(*grad_weight, grad_weight_node);
    }

    if (grad_input_needed && grad_input != nullptr)
    {
        auto *grad_input_node = get_or_create_data_node(
            *grad_input,
            input_graph,
            nntile::DataType::FP32,
            mark_as_input_for_operand(*grad_input));
        auto *mean_buf = make_graph_tensor(graph, reduced_graph, "mean_buf");
        auto *grad_temp = make_graph_tensor(graph, input_graph, "grad_temp");
        auto *tmp_grad = make_graph_tensor(graph, input_graph, "tmp_grad");

        if (has_weight)
        {
            auto *weight_node = get_or_create_data_node(
                *weight,
                {norm_len},
                nntile::DataType::FP32,
                mark_as_input_for_operand(*weight));
            nntile::tensor::multiply_fiber(
                static_cast<nntile::Scalar>(1.0),
                weight_node,
                grad_out_node,
                grad_temp,
                axis);
        }
        else
        {
            nntile::tensor::copy(grad_out_node, grad_temp);
        }

        nntile::tensor::copy(normalized, tmp_grad);
        nntile::tensor::sumprod_slice(
            grad_temp,
            tmp_grad,
            mean_buf,
            axis,
            kNormRedux,
            static_cast<nntile::Scalar>(inv_l),
            static_cast<nntile::Scalar>(0.0));
        nntile::tensor::multiply_slice(
            static_cast<nntile::Scalar>(1.0),
            mean_buf,
            tmp_grad,
            axis);
        nntile::tensor::add_inplace(
            static_cast<nntile::Scalar>(1.0),
            grad_temp,
            static_cast<nntile::Scalar>(1.0),
            tmp_grad);
        nntile::tensor::multiply_slice(
            static_cast<nntile::Scalar>(1.0),
            rstd_node,
            tmp_grad,
            axis);
        nntile::tensor::copy(tmp_grad, grad_input_node);
        register_data_node(*grad_input, grad_input_node);
    }

    maybe_execute_after_record();
}

void tensor_adam_step_fp32(
    int64_t num_iter,
    float beta_1,
    float beta_2,
    float eps,
    float lr,
    float weight_decay,
    const at::Tensor &grad,
    at::Tensor &first_moment,
    at::Tensor &second_moment,
    at::Tensor &param)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(grad.sizes());

    nntile::TensorGraph::TensorNode *grad_node =
        lookup_param_grad_node(param);
    if (grad_node == nullptr)
    {
        grad_node = lookup_data_node(grad, graph_shape);
    }
    if (grad_node == nullptr)
    {
        grad_node = get_or_create_data_node(
            grad,
            graph_shape,
            nntile::DataType::FP32,
            mark_as_input_for_operand(grad));
    }
    TORCH_CHECK(
        grad_node != nullptr,
        "nntile adam_step: parameter grad is not registered in the graph; "
        "run backward before the optimizer step");
    at::Tensor mutable_grad = grad;
    register_grad_alias_for_host_copy(mutable_grad, grad_node);
    auto *first_moment_node = get_or_create_data_node(
        first_moment,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(first_moment));
    auto *second_moment_node = get_or_create_data_node(
        second_moment,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(second_moment));
    auto *param_node = get_or_create_data_node(
        param,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(param));

    nntile::tensor::adam_step(
        static_cast<nntile::Index>(num_iter),
        static_cast<nntile::Scalar>(beta_1),
        static_cast<nntile::Scalar>(beta_2),
        static_cast<nntile::Scalar>(eps),
        static_cast<nntile::Scalar>(lr),
        static_cast<nntile::Scalar>(weight_decay),
        grad_node,
        first_moment_node,
        second_moment_node,
        param_node);

    register_data_node(first_moment, first_moment_node);
    register_data_node(second_moment, second_moment_node);
    register_data_node(param, param_node);
    maybe_execute_after_record();
}

void tensor_adamw_step_fp32(
    int64_t num_iter,
    float beta_1,
    float beta_2,
    float eps,
    float lr,
    float weight_decay,
    const at::Tensor &grad,
    at::Tensor &first_moment,
    at::Tensor &second_moment,
    at::Tensor &param)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(grad.sizes());

    nntile::TensorGraph::TensorNode *grad_node =
        lookup_param_grad_node(param);
    if (grad_node == nullptr)
    {
        grad_node = lookup_data_node(grad, graph_shape);
    }
    if (grad_node == nullptr)
    {
        grad_node = get_or_create_data_node(
            grad,
            graph_shape,
            nntile::DataType::FP32,
            mark_as_input_for_operand(grad));
    }
    TORCH_CHECK(
        grad_node != nullptr,
        "nntile adam_step: parameter grad is not registered in the graph; "
        "run backward before the optimizer step");
    at::Tensor mutable_grad = grad;
    register_grad_alias_for_host_copy(mutable_grad, grad_node);
    auto *first_moment_node = get_or_create_data_node(
        first_moment,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(first_moment));
    auto *second_moment_node = get_or_create_data_node(
        second_moment,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(second_moment));
    auto *param_node = get_or_create_data_node(
        param,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(param));

    nntile::tensor::adamw_step(
        static_cast<nntile::Index>(num_iter),
        static_cast<nntile::Scalar>(beta_1),
        static_cast<nntile::Scalar>(beta_2),
        static_cast<nntile::Scalar>(eps),
        static_cast<nntile::Scalar>(lr),
        static_cast<nntile::Scalar>(weight_decay),
        grad_node,
        first_moment_node,
        second_moment_node,
        param_node);

    register_data_node(first_moment, first_moment_node);
    register_data_node(second_moment, second_moment_node);
    register_data_node(param, param_node);
    maybe_execute_after_record();
}

void tensor_norm_fp32(
    const at::Tensor &x,
    at::Tensor &out)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(x.sizes());

    auto *x_node = get_or_create_data_node(
        x,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(x));
    auto *out_node = get_or_create_data_node(
        out,
        std::vector<nntile::Index>{},
        nntile::DataType::FP32,
        mark_as_input_for_operand(out));

    nntile::tensor::clear(out_node);
    nntile::tensor::norm(
        x_node,
        out_node,
        static_cast<nntile::Scalar>(1.0),
        static_cast<nntile::Scalar>(0.0));
    register_data_node(out, out_node);
    maybe_execute_after_record();
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
    const std::vector<nntile::Index> reduced_graph =
        reduced_shape_along_axis(input_graph, ax);

    auto *x_node = get_or_create_data_node(
        x,
        input_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(x));

    if (keepdim)
    {
        const std::vector<nntile::Index> keepdim_graph =
            keepdim_shape_along_axis(input_graph, ax);
        auto *out_node = get_or_create_data_node(
            out,
            keepdim_graph,
            nntile::DataType::FP32,
            mark_as_input_for_operand(out));
        nntile::TensorGraph &graph = *x_node->graph();
        auto *reduced = make_graph_tensor(graph, reduced_graph, "norm_red");
        auto *base = make_graph_tensor(graph, reduced_graph, "norm_base");
        nntile::tensor::clear(base);
        nntile::tensor::norm_slice(
            static_cast<nntile::Scalar>(1.0),
            x_node,
            static_cast<nntile::Scalar>(0.0),
            base,
            reduced,
            ax,
            kNormRedux);
        broadcast_slice_to_keepdim(reduced, out_node, ax);
        register_data_node(out, out_node);
    }
    else
    {
        auto *out_node = get_or_create_data_node(
            out,
            reduced_graph,
            nntile::DataType::FP32,
            mark_as_input_for_operand(out));
        nntile::TensorGraph &graph = *x_node->graph();
        auto *base = make_graph_tensor(graph, reduced_graph, "norm_base");
        nntile::tensor::clear(base);
        nntile::tensor::norm_slice(
            static_cast<nntile::Scalar>(1.0),
            x_node,
            static_cast<nntile::Scalar>(0.0),
            base,
            out_node,
            ax,
            kNormRedux);
        register_data_node(out, out_node);
    }
    maybe_execute_after_record();
}

void tensor_norm_backward_fp32(
    const at::Tensor &grad_out,
    const at::Tensor &x,
    const at::Tensor &norm_values,
    at::Tensor &grad_input,
    bool is_global,
    int64_t axis)
{
    const std::vector<nntile::Index> input_graph =
        pytorch_shape_to_graph(x.sizes());
    constexpr float kNormEps = 1e-12f;

    auto *x_node = get_or_create_data_node(
        x,
        input_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(x));
    auto *grad_input_node = get_or_create_data_node(
        grad_input,
        input_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(grad_input));
    nntile::TensorGraph &graph = *x_node->graph();

    if (is_global)
    {
        auto *grad_out_node = get_or_create_data_node(
            grad_out,
            std::vector<nntile::Index>{},
            nntile::DataType::FP32,
            mark_as_input_for_operand(grad_out));
        auto *norm_node = get_or_create_data_node(
            norm_values,
            std::vector<nntile::Index>{},
            nntile::DataType::FP32,
            mark_as_input_for_operand(norm_values));

        auto *inv_norm = make_graph_tensor(graph, std::vector<nntile::Index>{}, "inv_norm");
        nntile::tensor::copy(norm_node, inv_norm);
        nntile::tensor::hypot_scalar_inverse(
            static_cast<nntile::Scalar>(kNormEps),
            static_cast<nntile::Scalar>(1.0),
            inv_norm);

        nntile::tensor::copy(x_node, grad_input_node);
        nntile::tensor::multiply_slice(
            static_cast<nntile::Scalar>(1.0),
            grad_out_node,
            grad_input_node,
            static_cast<nntile::Index>(0));
        nntile::tensor::multiply_slice(
            static_cast<nntile::Scalar>(1.0),
            inv_norm,
            grad_input_node,
            static_cast<nntile::Index>(0));
    }
    else
    {
        const nntile::Index ax = static_cast<nntile::Index>(axis);
        const std::vector<nntile::Index> reduced_graph =
            reduced_shape_along_axis(input_graph, ax);

        auto *grad_out_node = get_or_create_data_node(
            grad_out,
            reduced_graph,
            nntile::DataType::FP32,
            mark_as_input_for_operand(grad_out));
        auto *norm_node = get_or_create_data_node(
            norm_values,
            reduced_graph,
            nntile::DataType::FP32,
            mark_as_input_for_operand(norm_values));

        auto *inv_norm = make_graph_tensor(graph, reduced_graph, "inv_norm");
        nntile::tensor::copy(norm_node, inv_norm);
        nntile::tensor::hypot_scalar_inverse(
            static_cast<nntile::Scalar>(kNormEps),
            static_cast<nntile::Scalar>(1.0),
            inv_norm);

        nntile::tensor::copy(x_node, grad_input_node);
        nntile::tensor::multiply_slice(
            static_cast<nntile::Scalar>(1.0),
            grad_out_node,
            grad_input_node,
            ax);
        nntile::tensor::multiply_slice(
            static_cast<nntile::Scalar>(1.0),
            inv_norm,
            grad_input_node,
            ax);
    }

    register_data_node(grad_input, grad_input_node);
    maybe_execute_after_record();
}

void tensor_sum_to_scalar_fp32(
    const at::Tensor &input,
    at::Tensor &out)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(input.sizes());
    if (graph_shape.empty())
    {
        if (input.data_ptr<float>() != out.data_ptr<float>())
        {
            std::memcpy(
                out.data_ptr<float>(),
                input.data_ptr<float>(),
                sizeof(float));
        }
        return;
    }

    auto *input_node = get_or_create_data_node(
        input,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(input));
    nntile::TensorGraph &graph = *input_node->graph();

    nntile::TensorGraph::TensorNode *cur = input_node;
    std::vector<nntile::Index> cur_shape = graph_shape;
    while (cur_shape.size() > 1)
    {
        const std::vector<nntile::Index> next_shape =
            reduced_shape_along_axis(cur_shape, 0);
        auto *next = make_graph_tensor(graph, next_shape, "sum_to_scalar");
        nntile::tensor::clear(next);
        nntile::tensor::sum_slice(
            cur,
            next,
            0,
            kNormRedux,
            static_cast<nntile::Scalar>(1.0),
            static_cast<nntile::Scalar>(0.0));
        cur = next;
        cur_shape = next_shape;
    }

    auto *out_node = get_or_create_data_node(
        out,
        std::vector<nntile::Index>{},
        nntile::DataType::FP32,
        mark_as_input_for_operand(out));
    nntile::tensor::clear(out_node);
    nntile::tensor::sum_slice(
        cur,
        out_node,
        0,
        kNormRedux,
        static_cast<nntile::Scalar>(1.0),
        static_cast<nntile::Scalar>(0.0));
    register_data_node(out, out_node);
    maybe_execute_after_record();
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

    std::vector<nntile::Index> cur_shape =
        pytorch_shape_to_graph(input_shape);
    auto *cur_node = get_or_create_data_node(
        input,
        cur_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(input));
    nntile::TensorGraph &graph = *cur_node->graph();

    for (std::size_t idx = 0; idx < dims.size(); ++idx)
    {
        const nntile::Index axis =
            static_cast<nntile::Index>(dims[idx]);
        const bool is_last = idx + 1 == dims.size();
        const std::vector<nntile::Index> reduced =
            reduced_shape_along_axis(cur_shape, axis);

        if (is_last)
        {
            if (keepdim)
            {
                const std::vector<nntile::Index> keepdim_shape =
                    keepdim_shape_along_axis(cur_shape, axis);
                auto *out_node = get_or_create_data_node(
                    out,
                    keepdim_shape,
                    nntile::DataType::FP32,
                    mark_as_input_for_operand(out));
                auto *reduced_node =
                    make_graph_tensor(graph, reduced, "sum_red");
                nntile::tensor::clear(reduced_node);
                nntile::tensor::sum_slice(
                    cur_node,
                    reduced_node,
                    axis,
                    kNormRedux,
                    static_cast<nntile::Scalar>(1.0),
                    static_cast<nntile::Scalar>(0.0));
                broadcast_slice_to_keepdim(
                    reduced_node,
                    out_node,
                    axis);
                register_data_node(out, out_node);
            }
            else
            {
                auto *out_node = get_or_create_data_node(
                    out,
                    reduced,
                    nntile::DataType::FP32,
                    mark_as_input_for_operand(out));
                nntile::tensor::clear(out_node);
                nntile::tensor::sum_slice(
                    cur_node,
                    out_node,
                    axis,
                    kNormRedux,
                    static_cast<nntile::Scalar>(1.0),
                    static_cast<nntile::Scalar>(0.0));
                register_data_node(out, out_node);
            }
            maybe_execute_after_record();
            return;
        }

        if (keepdim)
        {
            const std::vector<nntile::Index> keepdim_shape =
                keepdim_shape_along_axis(cur_shape, axis);
            auto *keepdim_node =
                make_graph_tensor(graph, keepdim_shape, "sum_tmp");
            auto *reduced_node =
                make_graph_tensor(graph, reduced, "sum_red");
            nntile::tensor::clear(reduced_node);
            nntile::tensor::sum_slice(
                cur_node,
                reduced_node,
                axis,
                kNormRedux,
                static_cast<nntile::Scalar>(1.0),
                static_cast<nntile::Scalar>(0.0));
            broadcast_slice_to_keepdim(
                reduced_node,
                keepdim_node,
                axis);
            cur_node = keepdim_node;
            cur_shape = keepdim_shape;
            continue;
        }

        auto *next = make_graph_tensor(graph, reduced, "sum_tmp");
        nntile::tensor::clear(next);
        nntile::tensor::sum_slice(
            cur_node,
            next,
            axis,
            kNormRedux,
            static_cast<nntile::Scalar>(1.0),
            static_cast<nntile::Scalar>(0.0));
        cur_node = next;
        cur_shape = reduced;
    }
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
    auto *out_node = get_or_create_data_node(
        out,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(out));
    nntile::tensor::scale(
        static_cast<nntile::Scalar>(scalar),
        input_node,
        out_node);
    register_data_node(out, out_node);
    maybe_execute_after_record();
}

void tensor_cat_fp32(
    const std::vector<at::Tensor> &inputs,
    at::Tensor &out,
    int64_t dim)
{
    TORCH_CHECK(!inputs.empty(), "tensor_cat_fp32: expected non-empty inputs");
    const nntile::Index axis = static_cast<nntile::Index>(dim);

    const std::vector<nntile::Index> first_graph =
        pytorch_shape_to_graph(inputs[0].sizes());
    auto *acc_node = get_or_create_data_node(
        inputs[0],
        first_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(inputs[0]));

    for (std::size_t i = 1; i < inputs.size(); ++i)
    {
        const std::vector<nntile::Index> shape_graph =
            pytorch_shape_to_graph(inputs[i].sizes());
        auto *next_node = get_or_create_data_node(
            inputs[i],
            shape_graph,
            nntile::DataType::FP32,
            mark_as_input_for_operand(inputs[i]));
        acc_node = nntile::tensor::concat(
            acc_node,
            next_node,
            axis)->set_name("cat");
    }

    register_data_node(out, acc_node);
    maybe_execute_after_record();
}

void tensor_narrow_fp32(
    const at::Tensor &input,
    int64_t dim,
    int64_t start,
    int64_t length,
    at::Tensor &out)
{
    (void) length;
    const nntile::Index axis = static_cast<nntile::Index>(dim);
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(input.sizes());

    auto *input_node = get_or_create_data_node(
        input,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(input));

    const std::vector<nntile::Index> out_graph =
        pytorch_shape_to_graph(out.sizes());
    auto *out_node = get_or_create_data_node(
        out,
        out_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(out));

    nntile::tensor::clear(out_node);

    const nntile::Index ndim = static_cast<nntile::Index>(graph_shape.size());
    std::vector<nntile::Index> zero(static_cast<size_t>(ndim), 0);
    std::vector<nntile::Index> dst_off = zero;
    dst_off[static_cast<size_t>(axis)] = static_cast<nntile::Index>(start);

    nntile::tensor::copy_intersection(
        input_node,
        zero,
        out_node,
        dst_off);
    register_data_node(out, out_node);
    maybe_execute_after_record();
}

void tensor_split_with_sizes_fp32(
    const at::Tensor &input,
    int64_t dim,
    const std::vector<int64_t> &split_sizes,
    const std::vector<at::Tensor> &outputs)
{
    const nntile::Index axis = static_cast<nntile::Index>(dim);
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(input.sizes());

    auto *input_node = get_or_create_data_node(
        input,
        graph_shape,
        nntile::DataType::FP32,
        mark_as_input_for_operand(input));

    const nntile::Index ndim = static_cast<nntile::Index>(graph_shape.size());
    std::vector<nntile::Index> zero(static_cast<size_t>(ndim), 0);
    nntile::Index accumulate = 0;

    for (std::size_t i = 0; i < split_sizes.size(); ++i)
    {
        const std::vector<nntile::Index> out_graph =
            pytorch_shape_to_graph(outputs[i].sizes());
        auto *out_node = get_or_create_data_node(
            outputs[i],
            out_graph,
            nntile::DataType::FP32,
            mark_as_input_for_operand(outputs[i]));

        nntile::tensor::clear(out_node);

        std::vector<nntile::Index> dst_off = zero;
        dst_off[static_cast<size_t>(axis)] = accumulate;
        nntile::tensor::copy_intersection(
            input_node,
            zero,
            out_node,
            dst_off);
        register_data_node(outputs[i], out_node);
        accumulate += static_cast<nntile::Index>(split_sizes[i]);
    }

    maybe_execute_after_record();
}

void tensor_embedding_forward_fp32(
    const at::Tensor &indices,
    const at::Tensor &weight,
    at::Tensor &out,
    nntile::Index axis)
{
    const std::vector<nntile::Index> index_graph =
        pytorch_shape_to_graph(indices.sizes());
    const std::vector<nntile::Index> weight_graph =
        pytorch_shape_to_graph(weight.sizes());
    const std::vector<nntile::Index> out_graph =
        pytorch_shape_to_graph(out.sizes());

    auto *index_node = get_or_create_data_node(
        indices,
        index_graph,
        nntile::DataType::INT64,
        mark_as_input_for_operand(indices));
    auto *weight_node = get_or_create_data_node(
        weight,
        weight_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(weight));
    auto *out_node = get_or_create_data_node(
        out,
        out_graph,
        nntile::DataType::FP32,
        false);

    nntile::tensor::embedding(index_node, weight_node, out_node, axis);
    register_data_node(out, out_node);
    maybe_execute_after_record();
}

void tensor_embedding_backward_fp32(
    const at::Tensor &indices,
    const at::Tensor &grad_out,
    at::Tensor &grad_weight,
    nntile::Index axis,
    int redux)
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
        mark_as_input_for_operand(grad_weight));

    nntile::tensor::clear(grad_weight_node);
    nntile::tensor::embedding_backward(
        index_node,
        grad_out_node,
        grad_weight_node,
        axis,
        redux);
    register_data_node(grad_weight, grad_weight_node);
    maybe_execute_after_record();
}

namespace
{

constexpr float kSdpaMaskVal =
    -std::numeric_limits<float>::infinity();
constexpr int kSdpaRedux = 0;

nntile::TensorGraph::TensorNode *make_sdpa_temp_tensor(
    nntile::TensorGraph &graph,
    const std::vector<nntile::Index> &shape,
    const char *name)
{
    auto *node = graph.data(shape, nntile::DataType::FP32)->set_name(name);
    track_graph_node(node);
    return node;
}

nntile::TensorGraph::TensorNode *compute_sdpa_attn(
    nntile::TensorGraph::TensorNode *q_node,
    nntile::TensorGraph::TensorNode *k_node,
    nntile::TensorGraph::TensorNode *mask_node,
    nntile::Index batch_ndim,
    float scale)
{
    const auto &q_shape = q_node->shape();
    const auto &k_shape = k_node->shape();
    const nntile::Index q_ndim = static_cast<nntile::Index>(q_shape.size());
    const nntile::Index q_seq = q_shape[static_cast<std::size_t>(q_ndim - 2)];
    const nntile::Index k_seq = k_shape[static_cast<std::size_t>(q_ndim - 2)];

    std::vector<nntile::Index> batch_shape(
        q_shape.begin(),
        q_shape.begin() + static_cast<ptrdiff_t>(batch_ndim));

    std::vector<nntile::Index> attn_shape = batch_shape;
    attn_shape.push_back(q_seq);
    attn_shape.push_back(k_seq);

    nntile::TensorGraph &graph = *q_node->graph();
    auto *attn_node = make_sdpa_temp_tensor(graph, attn_shape, "sdpa_attn");
    nntile::tensor::gemm(
        q_node,
        k_node,
        attn_node,
        static_cast<nntile::Scalar>(scale),
        static_cast<nntile::Scalar>(0.0),
        false,
        true,
        static_cast<nntile::Index>(1),
        batch_ndim);

    if (mask_node != nullptr)
    {
        nntile::tensor::mask_scalar(
            mask_node,
            static_cast<nntile::Scalar>(kSdpaMaskVal),
            attn_node,
            batch_ndim);
    }

    std::vector<nntile::Index> maxsumexp_shape = batch_shape;
    maxsumexp_shape.push_back(q_seq);
    maxsumexp_shape.push_back(static_cast<nntile::Index>(2));
    auto *maxsumexp_node =
        make_sdpa_temp_tensor(graph, maxsumexp_shape, "sdpa_maxsumexp");
    nntile::tensor::clear(maxsumexp_node);

    const nntile::Index attn_axis = q_ndim - 1;
    nntile::tensor::maxsumexp(
        attn_node,
        maxsumexp_node,
        attn_axis,
        kSdpaRedux);
    nntile::tensor::softmax_inplace(
        maxsumexp_node,
        attn_node,
        static_cast<nntile::Scalar>(1.0),
        attn_axis);

    return attn_node;
}

} // namespace

void tensor_sdpa_forward_fp32(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor *mask,
    at::Tensor &out,
    int64_t batch_ndim)
{
    const std::vector<nntile::Index> q_graph = pytorch_shape_to_graph(q.sizes());
    const std::vector<nntile::Index> k_graph = pytorch_shape_to_graph(k.sizes());
    const std::vector<nntile::Index> v_graph = pytorch_shape_to_graph(v.sizes());
    const nntile::Index batch_ndim_graph =
        static_cast<nntile::Index>(batch_ndim);
    const nntile::Index q_ndim = static_cast<nntile::Index>(q_graph.size());
    const nntile::Index head_size = q_graph[static_cast<std::size_t>(q_ndim - 1)];
    const float scale =
        1.0f / std::sqrt(static_cast<float>(static_cast<std::int64_t>(head_size)));

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
    auto *out_node = get_or_create_data_node(
        out,
        q_graph,
        nntile::DataType::FP32,
        false);

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

    auto *attn_node = compute_sdpa_attn(
        q_node,
        k_node,
        mask_node,
        batch_ndim_graph,
        scale);
    nntile::tensor::gemm(
        attn_node,
        v_node,
        out_node,
        static_cast<nntile::Scalar>(1.0),
        static_cast<nntile::Scalar>(0.0),
        false,
        false,
        static_cast<nntile::Index>(1),
        batch_ndim_graph);
    register_data_node(out, out_node);
    maybe_execute_after_record();
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
    int64_t batch_ndim)
{
    const std::vector<nntile::Index> q_graph = pytorch_shape_to_graph(q.sizes());
    const std::vector<nntile::Index> k_graph = pytorch_shape_to_graph(k.sizes());
    const std::vector<nntile::Index> v_graph = pytorch_shape_to_graph(v.sizes());
    const nntile::Index batch_ndim_graph =
        static_cast<nntile::Index>(batch_ndim);
    const nntile::Index q_ndim = static_cast<nntile::Index>(q_graph.size());
    const nntile::Index head_size = q_graph[static_cast<std::size_t>(q_ndim - 1)];
    const nntile::Index q_seq = q_graph[static_cast<std::size_t>(q_ndim - 2)];
    const float scale =
        1.0f / std::sqrt(static_cast<float>(static_cast<std::int64_t>(head_size)));

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

    nntile::TensorGraph &graph = *q_node->graph();
    auto *attn_node = compute_sdpa_attn(
        q_node,
        k_node,
        mask_node,
        batch_ndim_graph,
        scale);

    std::vector<nntile::Index> batch_shape(
        q_graph.begin(),
        q_graph.begin() + static_cast<ptrdiff_t>(batch_ndim_graph));
    std::vector<nntile::Index> attn_shape = batch_shape;
    attn_shape.push_back(q_seq);
    attn_shape.push_back(k_graph[static_cast<std::size_t>(q_ndim - 2)]);
    std::vector<nntile::Index> sumprod_shape = batch_shape;
    sumprod_shape.push_back(q_seq);

    auto *grad_temp = make_sdpa_temp_tensor(graph, attn_shape, "sdpa_grad_temp");
    auto *sumprod_buf = make_sdpa_temp_tensor(graph, sumprod_shape, "sdpa_sumprod");

    auto *grad_v_node = get_or_create_data_node(
        grad_v,
        v_graph,
        nntile::DataType::FP32,
        false);
    nntile::tensor::gemm(
        attn_node,
        grad_out_node,
        grad_v_node,
        static_cast<nntile::Scalar>(1.0),
        static_cast<nntile::Scalar>(0.0),
        true,
        false,
        static_cast<nntile::Index>(1),
        batch_ndim_graph);

    nntile::tensor::gemm(
        grad_out_node,
        v_node,
        grad_temp,
        static_cast<nntile::Scalar>(1.0),
        static_cast<nntile::Scalar>(0.0),
        false,
        true,
        static_cast<nntile::Index>(1),
        batch_ndim_graph);

    const nntile::Index attn_axis = q_ndim - 1;
    nntile::tensor::sumprod_slice(
        attn_node,
        grad_temp,
        sumprod_buf,
        attn_axis,
        kSdpaRedux,
        static_cast<nntile::Scalar>(1.0),
        static_cast<nntile::Scalar>(0.0));
    nntile::tensor::add_slice_inplace(
        static_cast<nntile::Scalar>(-1.0),
        sumprod_buf,
        static_cast<nntile::Scalar>(1.0),
        grad_temp,
        attn_axis);
    nntile::tensor::multiply_inplace(
        static_cast<nntile::Scalar>(1.0),
        attn_node,
        grad_temp);

    auto *grad_q_node = get_or_create_data_node(
        grad_q,
        q_graph,
        nntile::DataType::FP32,
        false);
    nntile::tensor::gemm(
        grad_temp,
        k_node,
        grad_q_node,
        static_cast<nntile::Scalar>(scale),
        static_cast<nntile::Scalar>(0.0),
        false,
        false,
        static_cast<nntile::Index>(1),
        batch_ndim_graph);

    auto *grad_k_node = get_or_create_data_node(
        grad_k,
        k_graph,
        nntile::DataType::FP32,
        false);
    nntile::tensor::gemm(
        grad_temp,
        q_node,
        grad_k_node,
        static_cast<nntile::Scalar>(scale),
        static_cast<nntile::Scalar>(0.0),
        true,
        false,
        static_cast<nntile::Index>(1),
        batch_ndim_graph);

    register_data_node(grad_v, grad_v_node);
    register_data_node(grad_q, grad_q_node);
    register_data_node(grad_k, grad_k_node);
    at::Tensor grad_q_alias = grad_q;
    at::Tensor grad_k_alias = grad_k;
    at::Tensor grad_v_alias = grad_v;
    register_grad_alias_for_host_copy(grad_q_alias, grad_q_node);
    register_grad_alias_for_host_copy(grad_k_alias, grad_k_node);
    register_grad_alias_for_host_copy(grad_v_alias, grad_v_node);
    maybe_execute_after_record();
}

} // namespace torch_nntile

#else

#include <stdexcept>
#include <string>
#include <utility>

namespace torch_nntile
{

namespace
{

[[noreturn]] void require_libnntile(const char *op)
{
    throw std::runtime_error(
        std::string("torch_nntile ") + op +
        " requires libnntile (rebuild with NNTILE_BUILD_DIR set)");
}

int64_t normalize_swap_dim(int64_t dim, int64_t ndim)
{
    if (dim < 0)
    {
        dim += ndim;
    }
    return dim;
}

void swap_two_axes_reference_fp32(
    const float *src,
    c10::IntArrayRef shape,
    float *dst,
    int64_t dim0,
    int64_t dim1)
{
    const int64_t n = static_cast<int64_t>(shape.size());
    dim0 = normalize_swap_dim(dim0, n);
    dim1 = normalize_swap_dim(dim1, n);
    if (dim0 > dim1)
    {
        std::swap(dim0, dim1);
    }

    int64_t d0 = 1;
    for (int64_t i = 0; i < dim0; ++i)
    {
        d0 *= shape[static_cast<size_t>(i)];
    }
    const int64_t d1 = shape[static_cast<size_t>(dim0)];
    int64_t d2 = 1;
    for (int64_t i = dim0 + 1; i < dim1; ++i)
    {
        d2 *= shape[static_cast<size_t>(i)];
    }
    const int64_t d3 = shape[static_cast<size_t>(dim1)];
    int64_t d4 = 1;
    for (int64_t i = dim1 + 1; i < n; ++i)
    {
        d4 *= shape[static_cast<size_t>(i)];
    }

    for (int64_t i0 = 0; i0 < d0; ++i0)
    {
        for (int64_t i1 = 0; i1 < d1; ++i1)
        {
            for (int64_t i2 = 0; i2 < d2; ++i2)
            {
                for (int64_t i3 = 0; i3 < d3; ++i3)
                {
                    for (int64_t i4 = 0; i4 < d4; ++i4)
                    {
                        const int64_t src_idx =
                            ((((i0 * d1 + i1) * d2 + i2) * d3 + i3) * d4 +
                                i4);
                        const int64_t dst_idx =
                            ((((i0 * d3 + i3) * d2 + i2) * d1 + i1) * d4 +
                                i4);
                        dst[static_cast<size_t>(dst_idx)] =
                            src[static_cast<size_t>(src_idx)];
                    }
                }
            }
        }
    }
}

} // namespace

void tensor_add_fp32(
    float /*alpha*/,
    const at::Tensor & /*x*/,
    float /*beta*/,
    const at::Tensor & /*y*/,
    at::Tensor & /*out*/)
{
    require_libnntile("add");
}

void tensor_model_transpose_forward_fp32(
    const at::Tensor & /*src*/,
    at::Tensor & /*dst*/,
    int64_t /*model_ndim*/)
{
    require_libnntile("model_transpose_forward");
}

void tensor_model_transpose_backward_fp32(
    const at::Tensor & /*grad_out*/,
    at::Tensor & /*grad_src*/,
    int64_t /*model_ndim*/)
{
    require_libnntile("model_transpose_backward");
}

void tensor_swap_two_axes_fp32(
    const at::Tensor &src,
    at::Tensor &dst,
    int64_t dim0,
    int64_t dim1)
{
    swap_two_axes_reference_fp32(
        src.data_ptr<float>(),
        src.sizes(),
        dst.data_ptr<float>(),
        dim0,
        dim1);
}

void tensor_add_inplace_fp32(
    float /*alpha*/,
    const at::Tensor & /*other*/,
    float /*beta*/,
    at::Tensor & /*self*/)
{
    require_libnntile("add_");
}

void tensor_mul_fp32(
    const at::Tensor & /*self*/,
    const at::Tensor & /*other*/,
    at::Tensor & /*out*/)
{
    require_libnntile("mul");
}

void tensor_mul_inplace_fp32(
    const at::Tensor & /*other*/,
    at::Tensor & /*self*/)
{
    require_libnntile("mul_");
}

void tensor_hypot_fp32(
    const at::Tensor & /*self*/,
    const at::Tensor & /*other*/,
    at::Tensor & /*out*/)
{
    require_libnntile("hypot");
}

void tensor_linear_fp32(
    const at::Tensor & /*input*/,
    const at::Tensor & /*weight*/,
    at::Tensor & /*out*/)
{
    require_libnntile("linear");
}

void tensor_relu_fp32(const at::Tensor & /*input*/, at::Tensor & /*out*/)
{
    require_libnntile("relu");
}

void tensor_relu_backward_fp32(
    const at::Tensor & /*x*/,
    const at::Tensor & /*dy*/,
    at::Tensor & /*dx*/)
{
    require_libnntile("relu_backward");
}

void tensor_silu_fp32(
    const at::Tensor & /*input*/,
    at::Tensor & /*out*/)
{
    require_libnntile("silu");
}

void tensor_silu_inplace_fp32(at::Tensor & /*self*/)
{
    require_libnntile("silu_inplace");
}

void tensor_silu_backward_fp32(
    const at::Tensor & /*x*/,
    const at::Tensor & /*dy*/,
    at::Tensor & /*dx*/)
{
    require_libnntile("silu_backward");
}

void tensor_gelu_fp32(
    const at::Tensor & /*input*/,
    at::Tensor & /*out*/,
    bool /*approximate_tanh*/)
{
    require_libnntile("gelu");
}

void tensor_gelu_inplace_fp32(at::Tensor & /*self*/, bool /*approximate_tanh*/)
{
    require_libnntile("gelu_inplace");
}

void tensor_gelu_backward_fp32(
    const at::Tensor & /*x*/,
    const at::Tensor & /*dy*/,
    at::Tensor & /*dx*/,
    bool /*approximate_tanh*/)
{
    require_libnntile("gelu_backward");
}

void tensor_gemm_fp32(
    const GemmParams & /*params*/,
    const at::Tensor & /*a*/,
    c10::IntArrayRef /*a_gemm_shape*/,
    const at::Tensor & /*b*/,
    c10::IntArrayRef /*b_gemm_shape*/,
    at::Tensor & /*out*/,
    c10::IntArrayRef /*out_shape*/)
{
    require_libnntile("gemm");
}

void tensor_gemm_accumulate_fp32(
    const GemmParams & /*params*/,
    const at::Tensor & /*a*/,
    c10::IntArrayRef /*a_gemm_shape*/,
    const at::Tensor & /*b*/,
    c10::IntArrayRef /*b_gemm_shape*/,
    const at::Tensor & /*c*/,
    c10::IntArrayRef /*c_shape*/,
    at::Tensor & /*out*/,
    c10::IntArrayRef /*out_shape*/)
{
    require_libnntile("gemm_accumulate");
}

void tensor_mm_fp32(
    const at::Tensor & /*a*/,
    const at::Tensor & /*b*/,
    at::Tensor & /*out*/)
{
    require_libnntile("mm");
}

void tensor_linear_backward_input_fp32(
    const at::Tensor & /*grad_out*/,
    const at::Tensor & /*weight*/,
    at::Tensor & /*grad_input*/)
{
    require_libnntile("linear_backward_input");
}

void tensor_linear_backward_weight_fp32(
    const at::Tensor & /*grad_out*/,
    const at::Tensor & /*input*/,
    at::Tensor & /*grad_weight*/)
{
    require_libnntile("linear_backward_weight");
}

void tensor_cross_entropy_forward_fp32(
    const at::Tensor & /*logits*/,
    const at::Tensor & /*labels*/,
    std::int64_t /*ignore_index*/,
    bool /*mean_reduction*/,
    at::Tensor & /*loss*/)
{
    require_libnntile("cross_entropy_forward");
}

void tensor_cross_entropy_backward_fp32(
    const at::Tensor & /*logits*/,
    const at::Tensor & /*labels*/,
    const at::Tensor & /*grad_output*/,
    at::Tensor & /*grad_row*/,
    at::Tensor & /*grad_logits*/,
    std::int64_t /*ignore_index*/,
    bool /*mean_reduction*/)
{
    require_libnntile("cross_entropy_backward");
}

void tensor_softmax_fp32(
    const at::Tensor & /*input*/,
    at::Tensor & /*out*/,
    int64_t /*dim*/)
{
    require_libnntile("softmax");
}

void tensor_softmax_backward_fp32(
    const at::Tensor & /*grad_output*/,
    const at::Tensor & /*output*/,
    at::Tensor & /*grad_input*/,
    int64_t /*dim*/)
{
    require_libnntile("softmax_backward");
}

void tensor_sgd_step_fp32(
    int64_t /*num_iter*/,
    float /*momentum*/,
    float /*lr*/,
    float /*weight_decay*/,
    float /*dampening*/,
    bool /*nesterov*/,
    const at::Tensor & /*grad*/,
    at::Tensor & /*velocity*/,
    at::Tensor & /*param*/)
{
    require_libnntile("sgd_step");
}

void tensor_layer_norm_forward_fp32(
    const at::Tensor & /*input*/,
    const at::Tensor * /*weight*/,
    const at::Tensor * /*bias*/,
    bool /*has_weight*/,
    bool /*has_bias*/,
    at::Tensor & /*output*/,
    at::Tensor & /*mean*/,
    at::Tensor & /*rstd*/,
    int64_t /*norm_axis*/,
    float /*eps*/)
{
    require_libnntile("layer_norm_forward");
}

void tensor_layer_norm_backward_fp32(
    const at::Tensor & /*grad_out*/,
    const at::Tensor & /*input*/,
    const at::Tensor & /*mean*/,
    const at::Tensor & /*rstd*/,
    const at::Tensor * /*weight*/,
    bool /*has_weight*/,
    bool /*has_bias*/,
    at::Tensor * /*grad_input*/,
    at::Tensor * /*grad_weight*/,
    at::Tensor * /*grad_bias*/,
    bool /*grad_input_needed*/,
    bool /*grad_weight_needed*/,
    bool /*grad_bias_needed*/,
    int64_t /*norm_axis*/)
{
    require_libnntile("layer_norm_backward");
}

void tensor_rms_norm_forward_fp32(
    const at::Tensor & /*input*/,
    const at::Tensor * /*weight*/,
    bool /*has_weight*/,
    at::Tensor & /*output*/,
    at::Tensor & /*rstd*/,
    int64_t /*norm_axis*/,
    float /*eps*/)
{
    require_libnntile("rms_norm_forward");
}

void tensor_rms_norm_backward_fp32(
    const at::Tensor & /*grad_out*/,
    const at::Tensor & /*input*/,
    const at::Tensor & /*rstd*/,
    const at::Tensor * /*weight*/,
    bool /*has_weight*/,
    at::Tensor * /*grad_input*/,
    at::Tensor * /*grad_weight*/,
    bool /*grad_input_needed*/,
    bool /*grad_weight_needed*/,
    int64_t /*norm_axis*/)
{
    require_libnntile("rms_norm_backward");
}

void tensor_adam_step_fp32(
    int64_t /*num_iter*/,
    float /*beta_1*/,
    float /*beta_2*/,
    float /*eps*/,
    float /*lr*/,
    float /*weight_decay*/,
    const at::Tensor & /*grad*/,
    at::Tensor & /*first_moment*/,
    at::Tensor & /*second_moment*/,
    at::Tensor & /*param*/)
{
    require_libnntile("adam_step");
}

void tensor_adamw_step_fp32(
    int64_t /*num_iter*/,
    float /*beta_1*/,
    float /*beta_2*/,
    float /*eps*/,
    float /*lr*/,
    float /*weight_decay*/,
    const at::Tensor & /*grad*/,
    at::Tensor & /*first_moment*/,
    at::Tensor & /*second_moment*/,
    at::Tensor & /*param*/)
{
    require_libnntile("adamw_step");
}

void tensor_norm_fp32(
    const at::Tensor & /*x*/,
    at::Tensor & /*out*/)
{
    require_libnntile("norm");
}

void tensor_norm_slice_fp32(
    const at::Tensor & /*x*/,
    at::Tensor & /*out*/,
    int64_t /*axis*/,
    bool /*keepdim*/)
{
    require_libnntile("norm_slice");
}

void tensor_norm_backward_fp32(
    const at::Tensor & /*grad_out*/,
    const at::Tensor & /*x*/,
    const at::Tensor & /*norm_values*/,
    at::Tensor & /*grad_input*/,
    bool /*is_global*/,
    int64_t /*axis*/)
{
    require_libnntile("norm_backward");
}

void tensor_sum_to_scalar_fp32(
    const at::Tensor & /*input*/,
    at::Tensor & /*out*/)
{
    require_libnntile("sum_to_scalar");
}

void tensor_sum_dimlist_fp32(
    const at::Tensor & /*input*/,
    at::Tensor & /*out*/,
    at::OptionalIntArrayRef /*dim*/,
    bool /*keepdim*/)
{
    require_libnntile("sum");
}

void tensor_mul_scalar_fp32(
    const at::Tensor & /*input*/,
    at::Tensor & /*out*/,
    float /*scalar*/)
{
    require_libnntile("mul_scalar");
}

void tensor_cat_fp32(
    const std::vector<at::Tensor> & /*inputs*/,
    at::Tensor & /*out*/,
    int64_t /*dim*/)
{
    require_libnntile("cat");
}

void tensor_narrow_fp32(
    const at::Tensor & /*input*/,
    int64_t /*dim*/,
    int64_t /*start*/,
    int64_t /*length*/,
    at::Tensor & /*out*/)
{
    require_libnntile("narrow");
}

void tensor_split_with_sizes_fp32(
    const at::Tensor & /*input*/,
    int64_t /*dim*/,
    const std::vector<int64_t> & /*split_sizes*/,
    const std::vector<at::Tensor> & /*outputs*/)
{
    require_libnntile("split_with_sizes");
}

void tensor_embedding_forward_fp32(
    const at::Tensor & /*indices*/,
    const at::Tensor & /*weight*/,
    at::Tensor & /*out*/,
    nntile::Index /*axis*/)
{
    require_libnntile("embedding");
}

void tensor_embedding_backward_fp32(
    const at::Tensor & /*indices*/,
    const at::Tensor & /*grad_out*/,
    at::Tensor & /*grad_weight*/,
    nntile::Index /*axis*/,
    int /*redux*/)
{
    require_libnntile("embedding_backward");
}

void tensor_sdpa_forward_fp32(
    const at::Tensor & /*q*/,
    const at::Tensor & /*k*/,
    const at::Tensor & /*v*/,
    const at::Tensor * /*mask*/,
    at::Tensor & /*out*/,
    int64_t /*batch_ndim*/)
{
    require_libnntile("sdpa_forward");
}

void tensor_sdpa_backward_fp32(
    const at::Tensor & /*q*/,
    const at::Tensor & /*k*/,
    const at::Tensor & /*v*/,
    const at::Tensor * /*mask*/,
    const at::Tensor & /*grad_out*/,
    at::Tensor & /*grad_q*/,
    at::Tensor & /*grad_k*/,
    at::Tensor & /*grad_v*/,
    int64_t /*batch_ndim*/)
{
    require_libnntile("sdpa_backward");
}

} // namespace torch_nntile

#endif
