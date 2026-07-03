/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_executor.cpp
 */

#include "nntile_executor.h"

#include "nntile_context.h"
#include "nntile_graph_recorder.h"
#include "nntile_graph_recorder_impl.h"
#include "nntile_tensor_gc.h"

#include <ATen/Tensor.h>

#ifdef TORCH_NNTILE_USE_LIBNNTILE

#include <nntile/base_types.hh>
#include <nntile/tensor/ops/add.hh>
#include <nntile/tensor/ops/add_fiber_inplace.hh>
#include <nntile/tensor/ops/add_inplace.hh>
#include <nntile/tensor/ops/add_slice.hh>
#include <nntile/tensor/ops/add_slice_inplace.hh>
#include <nntile/tensor/ops/multiply.hh>
#include <nntile/tensor/ops/multiply_inplace.hh>
#include <nntile/tensor/ops/clear.hh>
#include <nntile/tensor/ops/copy.hh>
#include <nntile/tensor/ops/gemm.hh>
#include <nntile/tensor/ops/hypot_scalar_inverse.hh>
#include <nntile/tensor/ops/logsumexp.hh>
#include <nntile/tensor/ops/maxsumexp.hh>
#include <nntile/tensor/ops/gelu.hh>
#include <nntile/tensor/ops/gelu_backward.hh>
#include <nntile/tensor/ops/gelu_inplace.hh>
#include <nntile/tensor/ops/gelutanh.hh>
#include <nntile/tensor/ops/gelutanh_backward.hh>
#include <nntile/tensor/ops/gelutanh_inplace.hh>
#include <nntile/tensor/ops/multiply_fiber.hh>
#include <nntile/tensor/ops/multiply_slice.hh>
#include <nntile/tensor/ops/norm_slice_inplace.hh>
#include <nntile/tensor/ops/relu.hh>
#include <nntile/tensor/ops/relu_backward.hh>
#include <nntile/tensor/ops/adam_step.hh>
#include <nntile/tensor/ops/adamw_step.hh>
#include <nntile/tensor/ops/silu.hh>
#include <nntile/tensor/ops/silu_backward.hh>
#include <nntile/tensor/ops/silu_inplace.hh>
#include <nntile/tensor/ops/sgd_step.hh>
#include <nntile/tensor/ops/scale_slice.hh>
#include <nntile/tensor/ops/softmax.hh>
#include <nntile/tensor/ops/subtract_indexed_outputs.hh>
#include <nntile/tensor/ops/sum_fiber.hh>
#include <nntile/tensor/ops/sum_slice.hh>
#include <nntile/tensor/ops/sumprod_fiber.hh>
#include <nntile/tensor/ops/sumprod_slice.hh>
#include <nntile/tensor/ops/total_sum_accum.hh>

#include <cmath>
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
    if (!has_host_staging(tensor))
    {
        return false;
    }
    // Graph-mode intermediates are metadata-only (nbytes==0). Eager-mode
    // chained ops reuse host-backed activations across execute() cycles.
    return !is_metadata_only_tensor(tensor);
}

} // namespace

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
        true);
    auto *self_node = get_or_create_data_node(
        self,
        graph_shape,
        nntile::DataType::FP32,
        true);

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
        true);
    auto *self_node = get_or_create_data_node(
        self,
        graph_shape,
        nntile::DataType::FP32,
        true);

    nntile::tensor::multiply_inplace(
        static_cast<nntile::Scalar>(1.0),
        other_node,
        self_node);
    register_data_node(self, self_node);
    maybe_execute_after_record();
}

void tensor_linear_fp32(
    const at::Tensor &input,
    const at::Tensor &weight,
    at::Tensor &out)
{
    const std::vector<nntile::Index> input_graph =
        pytorch_shape_to_graph(input.sizes());
    const std::vector<nntile::Index> weight_graph =
        pytorch_shape_to_graph(weight.sizes());

    auto *input_node = get_or_create_data_node(
        input,
        input_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(input));
    auto *weight_node = get_or_create_data_node(
        weight,
        weight_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(weight));

    auto *out_node = nntile::tensor::gemm(
        input_node,
        weight_node,
        static_cast<nntile::Scalar>(1.0),
        false,
        true,
        static_cast<nntile::Index>(1),
        static_cast<nntile::Index>(0))->set_name("output");
    register_data_node(out, out_node);
    maybe_execute_after_record();
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
    const std::vector<nntile::Index> a_graph =
        pytorch_shape_to_graph(a.sizes());
    const std::vector<nntile::Index> b_graph =
        pytorch_shape_to_graph(b.sizes());

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
        static_cast<nntile::Scalar>(1.0),
        false,
        false,
        static_cast<nntile::Index>(1),
        static_cast<nntile::Index>(0))->set_name("out");
    register_data_node(out, out_node);
    maybe_execute_after_record();
}

void tensor_linear_backward_input_fp32(
    const at::Tensor &grad_out,
    const at::Tensor &weight,
    at::Tensor &grad_input)
{
    const std::vector<nntile::Index> grad_out_graph =
        pytorch_shape_to_graph(grad_out.sizes());
    const std::vector<nntile::Index> weight_graph =
        pytorch_shape_to_graph(weight.sizes());

    auto *grad_out_node = get_or_create_data_node(
        grad_out,
        grad_out_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(grad_out));
    auto *weight_node = get_or_create_data_node(
        weight,
        weight_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(weight));

    auto *grad_input_node = nntile::tensor::gemm(
        grad_out_node,
        weight_node,
        static_cast<nntile::Scalar>(1.0),
        false,
        false,
        static_cast<nntile::Index>(1),
        static_cast<nntile::Index>(0))->set_name("grad_input");
    register_data_node(grad_input, grad_input_node);
    maybe_execute_after_record();
}

void tensor_linear_backward_weight_fp32(
    const at::Tensor &grad_out,
    const at::Tensor &input,
    at::Tensor &grad_weight)
{
    const std::vector<nntile::Index> grad_out_graph =
        pytorch_shape_to_graph(grad_out.sizes());
    const std::vector<nntile::Index> input_graph =
        pytorch_shape_to_graph(input.sizes());

    auto *grad_out_node = get_or_create_data_node(
        grad_out,
        grad_out_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(grad_out));
    auto *input_node = get_or_create_data_node(
        input,
        input_graph,
        nntile::DataType::FP32,
        mark_as_input_for_operand(input));

    auto *grad_weight_node = nntile::tensor::gemm(
        grad_out_node,
        input_node,
        static_cast<nntile::Scalar>(1.0),
        true,
        false,
        static_cast<nntile::Index>(1),
        static_cast<nntile::Index>(0))->set_name("grad_weight");
    register_data_node(grad_weight, grad_weight_node);
    maybe_execute_after_record();
}

namespace
{

constexpr int kRedux = 0;

std::vector<nntile::Index> maxsumexp_graph_shape(
    const std::vector<nntile::Index> &logits_graph)
{
    const nntile::Index class_axis =
        static_cast<nntile::Index>(logits_graph.size()) - 1;
    std::vector<nntile::Index> maxsumexp_shape;
    maxsumexp_shape.reserve(logits_graph.size());
    for (nntile::Index i = 0; i < class_axis; ++i)
    {
        maxsumexp_shape.push_back(logits_graph[static_cast<std::size_t>(i)]);
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
    const std::vector<nntile::Index> maxsumexp_graph =
        maxsumexp_graph_shape(logits_graph);
    const nntile::Index class_axis = class_graph_axis(logits.sizes());
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
        mark_as_input_for_operand(loss));

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
    const std::vector<nntile::Index> maxsumexp_graph =
        maxsumexp_graph_shape(logits_graph);
    const nntile::Index class_axis = class_graph_axis(logits.sizes());
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
        mark_as_input_for_operand(grad_logits));

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

    auto *grad_node = get_or_create_data_node(
        grad,
        graph_shape,
        nntile::DataType::FP32,
        true);
    auto *velocity_node = get_or_create_data_node(
        velocity,
        graph_shape,
        nntile::DataType::FP32,
        true);
    auto *param_node = get_or_create_data_node(
        param,
        graph_shape,
        nntile::DataType::FP32,
        true);

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

    auto *grad_node = get_or_create_data_node(
        grad,
        graph_shape,
        nntile::DataType::FP32,
        true);
    auto *first_moment_node = get_or_create_data_node(
        first_moment,
        graph_shape,
        nntile::DataType::FP32,
        true);
    auto *second_moment_node = get_or_create_data_node(
        second_moment,
        graph_shape,
        nntile::DataType::FP32,
        true);
    auto *param_node = get_or_create_data_node(
        param,
        graph_shape,
        nntile::DataType::FP32,
        true);

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

    auto *grad_node = get_or_create_data_node(
        grad,
        graph_shape,
        nntile::DataType::FP32,
        true);
    auto *first_moment_node = get_or_create_data_node(
        first_moment,
        graph_shape,
        nntile::DataType::FP32,
        true);
    auto *second_moment_node = get_or_create_data_node(
        second_moment,
        graph_shape,
        nntile::DataType::FP32,
        true);
    auto *param_node = get_or_create_data_node(
        param,
        graph_shape,
        nntile::DataType::FP32,
        true);

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

} // namespace torch_nntile

#else

#include <stdexcept>
#include <string>

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

} // namespace torch_nntile

#endif
