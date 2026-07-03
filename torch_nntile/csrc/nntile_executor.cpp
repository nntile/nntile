/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_executor.cpp
 */

#include "nntile_executor.h"

#include "nntile_context.h"
#include "nntile_graph_recorder.h"
#include "nntile_graph_recorder_impl.h"

#ifdef TORCH_NNTILE_USE_LIBNNTILE

#include <nntile/base_types.hh>
#include <nntile/tensor/ops/add.hh>
#include <nntile/tensor/ops/add_fiber_inplace.hh>
#include <nntile/tensor/ops/add_inplace.hh>
#include <nntile/tensor/ops/add_slice.hh>
#include <nntile/tensor/ops/add_slice_inplace.hh>
#include <nntile/tensor/ops/concat.hh>
#include <nntile/tensor/ops/multiply.hh>
#include <nntile/tensor/ops/multiply_inplace.hh>
#include <nntile/tensor/ops/clear.hh>
#include <nntile/tensor/ops/copy.hh>
#include <nntile/tensor/ops/gemm.hh>
#include <nntile/tensor/ops/hypot.hh>
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

} // namespace

void tensor_add_fp32(
    float alpha,
    const float *x_data,
    float beta,
    const float *y_data,
    float *out_data,
    c10::IntArrayRef pytorch_shape)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(pytorch_shape);

    auto *x_node = get_or_create_data_node(
        const_cast<float *>(x_data),
        graph_shape,
        nntile::DataType::FP32,
        true);
    auto *y_node = get_or_create_data_node(
        const_cast<float *>(y_data),
        graph_shape,
        nntile::DataType::FP32,
        true);

    auto *z_node = nntile::tensor::add(
        static_cast<nntile::Scalar>(alpha),
        x_node,
        static_cast<nntile::Scalar>(beta),
        y_node)->set_name("z");
    register_data_node(out_data, z_node);
    maybe_execute_after_record();
}

void tensor_add_inplace_fp32(
    float alpha,
    const float *other_data,
    float beta,
    float *self_data,
    c10::IntArrayRef pytorch_shape)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(pytorch_shape);

    auto *other_node = get_or_create_data_node(
        const_cast<float *>(other_data),
        graph_shape,
        nntile::DataType::FP32,
        true);
    auto *self_node = get_or_create_data_node(
        self_data,
        graph_shape,
        nntile::DataType::FP32,
        true);

    nntile::tensor::add_inplace(
        static_cast<nntile::Scalar>(alpha),
        other_node,
        static_cast<nntile::Scalar>(beta),
        self_node);
    register_data_node(self_data, self_node);
    maybe_execute_after_record();
}

void tensor_mul_fp32(
    const float *self_data,
    const float *other_data,
    float *out_data,
    c10::IntArrayRef pytorch_shape)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(pytorch_shape);

    auto *self_node = get_or_create_data_node(
        const_cast<float *>(self_data),
        graph_shape,
        nntile::DataType::FP32,
        true);
    auto *other_node = get_or_create_data_node(
        const_cast<float *>(other_data),
        graph_shape,
        nntile::DataType::FP32,
        true);

    auto *out_node = nntile::tensor::multiply(
        self_node,
        other_node,
        static_cast<nntile::Scalar>(1.0))->set_name("z");
    register_data_node(out_data, out_node);
    maybe_execute_after_record();
}

void tensor_mul_inplace_fp32(
    const float *other_data,
    float *self_data,
    c10::IntArrayRef pytorch_shape)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(pytorch_shape);

    auto *other_node = get_or_create_data_node(
        const_cast<float *>(other_data),
        graph_shape,
        nntile::DataType::FP32,
        true);
    auto *self_node = get_or_create_data_node(
        self_data,
        graph_shape,
        nntile::DataType::FP32,
        true);

    nntile::tensor::multiply_inplace(
        static_cast<nntile::Scalar>(1.0),
        other_node,
        self_node);
    register_data_node(self_data, self_node);
    maybe_execute_after_record();
}

void tensor_hypot_fp32(
    const float *self_data,
    const float *other_data,
    float *out_data,
    c10::IntArrayRef pytorch_shape)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(pytorch_shape);

    auto *self_node = get_or_create_data_node(
        const_cast<float *>(self_data),
        graph_shape,
        nntile::DataType::FP32,
        true);
    auto *other_node = get_or_create_data_node(
        const_cast<float *>(other_data),
        graph_shape,
        nntile::DataType::FP32,
        true);

    auto *out_node = nntile::tensor::hypot(
        static_cast<nntile::Scalar>(1.0),
        self_node,
        static_cast<nntile::Scalar>(1.0),
        other_node)->set_name("hypot_out");
    register_data_node(out_data, out_node);
    maybe_execute_after_record();
}

void tensor_linear_fp32(
    const float *input_data,
    c10::IntArrayRef input_shape,
    const float *weight_data,
    c10::IntArrayRef weight_shape,
    float *out_data,
    c10::IntArrayRef out_shape)
{
    const std::vector<nntile::Index> input_graph =
        pytorch_shape_to_graph(input_shape);
    const std::vector<nntile::Index> weight_graph =
        pytorch_shape_to_graph(weight_shape);

    auto *input_node = get_or_create_data_node(
        const_cast<float *>(input_data),
        input_graph,
        nntile::DataType::FP32,
        true);
    auto *weight_node = get_or_create_data_node(
        const_cast<float *>(weight_data),
        weight_graph,
        nntile::DataType::FP32,
        true);

    auto *out_node = nntile::tensor::gemm(
        input_node,
        weight_node,
        static_cast<nntile::Scalar>(1.0),
        false,
        true,
        static_cast<nntile::Index>(1),
        static_cast<nntile::Index>(0))->set_name("output");
    register_data_node(out_data, out_node);
    maybe_execute_after_record();
}

void tensor_relu_fp32(
    const float *input_data,
    float *out_data,
    c10::IntArrayRef pytorch_shape)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(pytorch_shape);

    auto *src_node = get_or_create_data_node(
        const_cast<float *>(input_data),
        graph_shape,
        nntile::DataType::FP32,
        true);

    auto *dst_node = nntile::tensor::relu(src_node)->set_name("dst");
    register_data_node(out_data, dst_node);
    maybe_execute_after_record();
}

void tensor_relu_backward_fp32(
    const float *x_data,
    const float *dy_data,
    float *dx_data,
    c10::IntArrayRef pytorch_shape)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(pytorch_shape);

    auto *x_node = get_or_create_data_node(
        const_cast<float *>(x_data),
        graph_shape,
        nntile::DataType::FP32,
        true);
    auto *dy_node = get_or_create_data_node(
        const_cast<float *>(dy_data),
        graph_shape,
        nntile::DataType::FP32,
        true);
    auto *dx_node = get_or_create_data_node(
        dx_data,
        graph_shape,
        nntile::DataType::FP32,
        true);

    nntile::tensor::clear(dx_node);
    nntile::tensor::relu_backward(x_node, dy_node, dx_node);
    register_data_node(dx_data, dx_node);
    maybe_execute_after_record();
}

void tensor_silu_fp32(
    const float *input_data,
    float *out_data,
    c10::IntArrayRef pytorch_shape)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(pytorch_shape);

    auto *src_node = get_or_create_data_node(
        const_cast<float *>(input_data),
        graph_shape,
        nntile::DataType::FP32,
        true);

    auto *dst_node = nntile::tensor::silu(src_node)->set_name("dst");
    register_data_node(out_data, dst_node);
    maybe_execute_after_record();
}

void tensor_silu_inplace_fp32(
    float *data,
    c10::IntArrayRef pytorch_shape)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(pytorch_shape);

    auto *node = get_or_create_data_node(
        data,
        graph_shape,
        nntile::DataType::FP32,
        true);

    nntile::tensor::silu_inplace(node);
    register_data_node(data, node);
    maybe_execute_after_record();
}

void tensor_silu_backward_fp32(
    const float *x_data,
    const float *dy_data,
    float *dx_data,
    c10::IntArrayRef pytorch_shape)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(pytorch_shape);

    auto *x_node = get_or_create_data_node(
        const_cast<float *>(x_data),
        graph_shape,
        nntile::DataType::FP32,
        true);
    auto *dy_node = get_or_create_data_node(
        const_cast<float *>(dy_data),
        graph_shape,
        nntile::DataType::FP32,
        true);
    auto *dx_node = get_or_create_data_node(
        dx_data,
        graph_shape,
        nntile::DataType::FP32,
        true);

    nntile::tensor::clear(dx_node);
    nntile::tensor::silu_backward(x_node, dy_node, dx_node);
    register_data_node(dx_data, dx_node);
    maybe_execute_after_record();
}

void tensor_gelu_fp32(
    const float *input_data,
    float *out_data,
    c10::IntArrayRef pytorch_shape,
    bool approximate_tanh)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(pytorch_shape);

    auto *src_node = get_or_create_data_node(
        const_cast<float *>(input_data),
        graph_shape,
        nntile::DataType::FP32,
        true);

    nntile::TensorGraph::TensorNode *dst_node = nullptr;
    if (approximate_tanh)
    {
        dst_node = nntile::tensor::gelutanh(src_node)->set_name("dst");
    }
    else
    {
        dst_node = nntile::tensor::gelu(src_node)->set_name("dst");
    }
    register_data_node(out_data, dst_node);
    maybe_execute_after_record();
}

void tensor_gelu_inplace_fp32(
    float *data,
    c10::IntArrayRef pytorch_shape,
    bool approximate_tanh)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(pytorch_shape);

    auto *node = get_or_create_data_node(
        data,
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
    register_data_node(data, node);
    maybe_execute_after_record();
}

void tensor_gelu_backward_fp32(
    const float *x_data,
    const float *dy_data,
    float *dx_data,
    c10::IntArrayRef pytorch_shape,
    bool approximate_tanh)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(pytorch_shape);

    auto *x_node = get_or_create_data_node(
        const_cast<float *>(x_data),
        graph_shape,
        nntile::DataType::FP32,
        true);
    auto *dy_node = get_or_create_data_node(
        const_cast<float *>(dy_data),
        graph_shape,
        nntile::DataType::FP32,
        true);
    auto *dx_node = get_or_create_data_node(
        dx_data,
        graph_shape,
        nntile::DataType::FP32,
        true);

    nntile::tensor::clear(dx_node);
    if (approximate_tanh)
    {
        nntile::tensor::gelutanh_backward(x_node, dy_node, dx_node);
    }
    else
    {
        nntile::tensor::gelu_backward(x_node, dy_node, dx_node);
    }
    register_data_node(dx_data, dx_node);
    maybe_execute_after_record();
}

void tensor_mm_fp32(
    const float *a_data,
    c10::IntArrayRef a_shape,
    const float *b_data,
    c10::IntArrayRef b_shape,
    float *out_data,
    c10::IntArrayRef out_shape)
{
    const std::vector<nntile::Index> a_graph =
        pytorch_shape_to_graph(a_shape);
    const std::vector<nntile::Index> b_graph =
        pytorch_shape_to_graph(b_shape);

    auto *a_node = get_or_create_data_node(
        const_cast<float *>(a_data),
        a_graph,
        nntile::DataType::FP32,
        true);
    auto *b_node = get_or_create_data_node(
        const_cast<float *>(b_data),
        b_graph,
        nntile::DataType::FP32,
        true);

    auto *out_node = nntile::tensor::gemm(
        a_node,
        b_node,
        static_cast<nntile::Scalar>(1.0),
        false,
        false,
        static_cast<nntile::Index>(1),
        static_cast<nntile::Index>(0))->set_name("out");
    register_data_node(out_data, out_node);
    maybe_execute_after_record();
}

void tensor_linear_backward_input_fp32(
    const float *grad_out_data,
    c10::IntArrayRef grad_out_shape,
    const float *weight_data,
    c10::IntArrayRef weight_shape,
    float *grad_input_data,
    c10::IntArrayRef grad_input_shape)
{
    const std::vector<nntile::Index> grad_out_graph =
        pytorch_shape_to_graph(grad_out_shape);
    const std::vector<nntile::Index> weight_graph =
        pytorch_shape_to_graph(weight_shape);

    auto *grad_out_node = get_or_create_data_node(
        const_cast<float *>(grad_out_data),
        grad_out_graph,
        nntile::DataType::FP32,
        true);
    auto *weight_node = get_or_create_data_node(
        const_cast<float *>(weight_data),
        weight_graph,
        nntile::DataType::FP32,
        true);

    auto *grad_input_node = nntile::tensor::gemm(
        grad_out_node,
        weight_node,
        static_cast<nntile::Scalar>(1.0),
        false,
        false,
        static_cast<nntile::Index>(1),
        static_cast<nntile::Index>(0))->set_name("grad_input");
    register_data_node(grad_input_data, grad_input_node);
    maybe_execute_after_record();
}

void tensor_linear_backward_weight_fp32(
    const float *grad_out_data,
    c10::IntArrayRef grad_out_shape,
    const float *input_data,
    c10::IntArrayRef input_shape,
    float *grad_weight_data,
    c10::IntArrayRef grad_weight_shape)
{
    const std::vector<nntile::Index> grad_out_graph =
        pytorch_shape_to_graph(grad_out_shape);
    const std::vector<nntile::Index> input_graph =
        pytorch_shape_to_graph(input_shape);

    auto *grad_out_node = get_or_create_data_node(
        const_cast<float *>(grad_out_data),
        grad_out_graph,
        nntile::DataType::FP32,
        true);
    auto *input_node = get_or_create_data_node(
        const_cast<float *>(input_data),
        input_graph,
        nntile::DataType::FP32,
        true);

    auto *grad_weight_node = nntile::tensor::gemm(
        grad_out_node,
        input_node,
        static_cast<nntile::Scalar>(1.0),
        true,
        false,
        static_cast<nntile::Index>(1),
        static_cast<nntile::Index>(0))->set_name("grad_weight");
    register_data_node(grad_weight_data, grad_weight_node);
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
    const float *logits_data,
    c10::IntArrayRef logits_shape,
    const std::int64_t *labels_data,
    c10::IntArrayRef labels_shape,
    std::int64_t ignore_index,
    bool mean_reduction,
    float *loss_data)
{
    const std::vector<nntile::Index> logits_graph =
        pytorch_shape_to_graph(logits_shape);
    const std::vector<nntile::Index> labels_graph =
        pytorch_shape_to_graph(labels_shape);
    const std::vector<nntile::Index> maxsumexp_graph =
        maxsumexp_graph_shape(logits_graph);
    const nntile::Index class_axis = class_graph_axis(logits_shape);
    const float scale = cross_entropy_scale(
        labels_data,
        labels_shape,
        ignore_index,
        mean_reduction);

    auto *logits_node = get_or_create_data_node(
        const_cast<float *>(logits_data),
        logits_graph,
        nntile::DataType::FP32,
        true);
    auto *labels_node = get_or_create_data_node(
        const_cast<std::int64_t *>(labels_data),
        labels_graph,
        nntile::DataType::INT64,
        true);
    auto *loss_node = get_or_create_data_node(
        loss_data,
        {},
        nntile::DataType::FP32,
        true);

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

    register_data_node(loss_data, loss_node);
    maybe_execute_after_record();
}

void tensor_cross_entropy_backward_fp32(
    const float *logits_data,
    c10::IntArrayRef logits_shape,
    const std::int64_t *labels_data,
    c10::IntArrayRef labels_shape,
    const float *grad_output_data,
    float *grad_row_data,
    float *grad_logits_data,
    std::int64_t ignore_index,
    bool mean_reduction)
{
    const std::vector<nntile::Index> logits_graph =
        pytorch_shape_to_graph(logits_shape);
    const std::vector<nntile::Index> labels_graph =
        pytorch_shape_to_graph(labels_shape);
    const std::vector<nntile::Index> maxsumexp_graph =
        maxsumexp_graph_shape(logits_graph);
    const nntile::Index class_axis = class_graph_axis(logits_shape);
    const float ce_scale = cross_entropy_scale(
        labels_data,
        labels_shape,
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
        const_cast<float *>(logits_data),
        logits_graph,
        nntile::DataType::FP32,
        true);
    auto *labels_node = get_or_create_data_node(
        const_cast<std::int64_t *>(labels_data),
        labels_graph,
        nntile::DataType::INT64,
        true);
    auto *grad_output_node = get_or_create_data_node(
        const_cast<float *>(grad_output_data),
        {},
        nntile::DataType::FP32,
        true);
    auto *grad_row_node = get_or_create_data_node(
        grad_row_data,
        labels_graph,
        nntile::DataType::FP32,
        false);
    auto *grad_logits_node = get_or_create_data_node(
        grad_logits_data,
        logits_graph,
        nntile::DataType::FP32,
        true);

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

    register_data_node(grad_logits_data, grad_logits_node);
    maybe_execute_after_record();
}

void tensor_sgd_step_fp32(
    int64_t num_iter,
    float momentum,
    float lr,
    float weight_decay,
    float dampening,
    bool nesterov,
    const float *grad_data,
    float *velocity_data,
    float *param_data,
    c10::IntArrayRef pytorch_shape)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(pytorch_shape);

    auto *grad_node = get_or_create_data_node(
        const_cast<float *>(grad_data),
        graph_shape,
        nntile::DataType::FP32,
        true);
    auto *velocity_node = get_or_create_data_node(
        velocity_data,
        graph_shape,
        nntile::DataType::FP32,
        true);
    auto *param_node = get_or_create_data_node(
        param_data,
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

    register_data_node(velocity_data, velocity_node);
    register_data_node(param_data, param_node);
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
    const float *input_data,
    c10::IntArrayRef input_shape,
    const float *weight_data,
    const float *bias_data,
    bool has_weight,
    bool has_bias,
    float *output_data,
    float *mean_data,
    float *rstd_data,
    int64_t norm_axis,
    float eps)
{
    const std::vector<nntile::Index> input_graph =
        pytorch_shape_to_graph(input_shape);
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
        const_cast<float *>(input_data),
        input_graph,
        nntile::DataType::FP32,
        true);
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
        mean_data,
        keepdim_graph,
        nntile::DataType::FP32,
        true);
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
        rstd_data,
        keepdim_graph,
        nntile::DataType::FP32,
        true);
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
            const_cast<float *>(weight_data),
            {norm_len},
            nntile::DataType::FP32,
            true);
        scaled = nntile::tensor::multiply_fiber(
            static_cast<nntile::Scalar>(1.0),
            weight_node,
            centered,
            axis);
    }

    auto *output_node = get_or_create_data_node(
        output_data,
        input_graph,
        nntile::DataType::FP32,
        true);
    if (has_bias)
    {
        auto *bias_node = get_or_create_data_node(
            const_cast<float *>(bias_data),
            {norm_len},
            nntile::DataType::FP32,
            true);
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

    register_data_node(output_data, output_node);
    register_data_node(mean_data, mean_node);
    register_data_node(rstd_data, rstd_node);
    maybe_execute_after_record();
}

void tensor_layer_norm_backward_fp32(
    const float *grad_out_data,
    const float *input_data,
    const float *mean_data,
    const float *rstd_data,
    const float *weight_data,
    bool has_weight,
    bool has_bias,
    float *grad_input_data,
    float *grad_weight_data,
    float *grad_bias_data,
    bool grad_input_needed,
    bool grad_weight_needed,
    bool grad_bias_needed,
    c10::IntArrayRef input_shape,
    int64_t norm_axis)
{
    const std::vector<nntile::Index> input_graph =
        pytorch_shape_to_graph(input_shape);
    const nntile::Index axis = static_cast<nntile::Index>(norm_axis);
    const nntile::Index norm_len = input_graph[static_cast<std::size_t>(axis)];
    const float inv_l =
        1.0f / static_cast<float>(static_cast<std::int64_t>(norm_len));
    const std::vector<nntile::Index> reduced_graph =
        reduced_shape_along_axis(input_graph, axis);

    auto *grad_out_node = get_or_create_data_node(
        const_cast<float *>(grad_out_data),
        input_graph,
        nntile::DataType::FP32,
        true);
    auto *input_node = get_or_create_data_node(
        const_cast<float *>(input_data),
        input_graph,
        nntile::DataType::FP32,
        true);
    auto *mean_node = get_or_create_data_node(
        const_cast<float *>(mean_data),
        reduced_graph,
        nntile::DataType::FP32,
        true);
    auto *rstd_node = get_or_create_data_node(
        const_cast<float *>(rstd_data),
        reduced_graph,
        nntile::DataType::FP32,
        true);
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

    if (grad_bias_needed)
    {
        auto *grad_bias_node = get_or_create_data_node(
            grad_bias_data,
            {norm_len},
            nntile::DataType::FP32,
            true);
        nntile::tensor::clear(grad_bias_node);
        nntile::tensor::sum_fiber(
            grad_out_node,
            grad_bias_node,
            axis,
            kBatchNdim,
            kNormRedux,
            static_cast<nntile::Scalar>(1.0),
            static_cast<nntile::Scalar>(0.0));
        register_data_node(grad_bias_data, grad_bias_node);
    }

    nntile::TensorGraph::TensorNode *grad_temp = grad_out_node;
    if (has_weight)
    {
        auto *weight_node = get_or_create_data_node(
            const_cast<float *>(weight_data),
            {norm_len},
            nntile::DataType::FP32,
            true);
        grad_temp = nntile::tensor::multiply_fiber(
            static_cast<nntile::Scalar>(1.0),
            weight_node,
            grad_out_node,
            axis);
    }

    if (grad_weight_needed)
    {
        auto *grad_weight_node = get_or_create_data_node(
            grad_weight_data,
            {norm_len},
            nntile::DataType::FP32,
            true);
        nntile::tensor::clear(grad_weight_node);
        nntile::tensor::sumprod_fiber(
            grad_out_node,
            x_hat,
            grad_weight_node,
            axis,
            kNormRedux,
            static_cast<nntile::Scalar>(1.0),
            static_cast<nntile::Scalar>(0.0));
        register_data_node(grad_weight_data, grad_weight_node);
    }

    if (grad_input_needed)
    {
        auto *grad_input_node = get_or_create_data_node(
            grad_input_data,
            input_graph,
            nntile::DataType::FP32,
            true);
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
        register_data_node(grad_input_data, grad_input_node);
    }

    maybe_execute_after_record();
}

void tensor_rms_norm_forward_fp32(
    const float *input_data,
    c10::IntArrayRef input_shape,
    const float *weight_data,
    bool has_weight,
    float *output_data,
    float *rstd_data,
    int64_t norm_axis,
    float eps)
{
    const std::vector<nntile::Index> input_graph =
        pytorch_shape_to_graph(input_shape);
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
        const_cast<float *>(input_data),
        input_graph,
        nntile::DataType::FP32,
        true);
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
        rstd_data,
        keepdim_graph,
        nntile::DataType::FP32,
        true);
    broadcast_slice_to_keepdim(rstd_reduced, rstd_node, axis);

    auto *normalized = nntile::tensor::copy(input_node);
    nntile::tensor::multiply_slice(
        static_cast<nntile::Scalar>(1.0),
        rstd_reduced,
        normalized,
        axis);

    auto *output_node = get_or_create_data_node(
        output_data,
        input_graph,
        nntile::DataType::FP32,
        true);
    if (has_weight)
    {
        auto *weight_node = get_or_create_data_node(
            const_cast<float *>(weight_data),
            {norm_len},
            nntile::DataType::FP32,
            true);
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

    register_data_node(output_data, output_node);
    register_data_node(rstd_data, rstd_node);
    maybe_execute_after_record();
}

void tensor_rms_norm_backward_fp32(
    const float *grad_out_data,
    const float *input_data,
    const float *rstd_data,
    const float *weight_data,
    bool has_weight,
    float *grad_input_data,
    float *grad_weight_data,
    bool grad_input_needed,
    bool grad_weight_needed,
    c10::IntArrayRef input_shape,
    int64_t norm_axis)
{
    const std::vector<nntile::Index> input_graph =
        pytorch_shape_to_graph(input_shape);
    const nntile::Index axis = static_cast<nntile::Index>(norm_axis);
    const nntile::Index norm_len = input_graph[static_cast<std::size_t>(axis)];
    const float inv_l =
        -1.0f / static_cast<float>(static_cast<std::int64_t>(norm_len));
    const std::vector<nntile::Index> reduced_graph =
        reduced_shape_along_axis(input_graph, axis);

    auto *grad_out_node = get_or_create_data_node(
        const_cast<float *>(grad_out_data),
        input_graph,
        nntile::DataType::FP32,
        true);
    auto *input_node = get_or_create_data_node(
        const_cast<float *>(input_data),
        input_graph,
        nntile::DataType::FP32,
        true);
    auto *rstd_node = get_or_create_data_node(
        const_cast<float *>(rstd_data),
        reduced_graph,
        nntile::DataType::FP32,
        true);
    nntile::TensorGraph &graph = *grad_out_node->graph();

    auto *normalized = nntile::tensor::copy(input_node);
    nntile::tensor::multiply_slice(
        static_cast<nntile::Scalar>(1.0),
        rstd_node,
        normalized,
        axis);

    if (grad_weight_needed && has_weight)
    {
        auto *grad_weight_node = get_or_create_data_node(
            grad_weight_data,
            {norm_len},
            nntile::DataType::FP32,
            true);
        nntile::tensor::clear(grad_weight_node);
        nntile::tensor::sumprod_fiber(
            grad_out_node,
            normalized,
            grad_weight_node,
            axis,
            kNormRedux,
            static_cast<nntile::Scalar>(1.0),
            static_cast<nntile::Scalar>(0.0));
        register_data_node(grad_weight_data, grad_weight_node);
    }

    if (grad_input_needed)
    {
        auto *grad_input_node = get_or_create_data_node(
            grad_input_data,
            input_graph,
            nntile::DataType::FP32,
            true);
        auto *mean_buf = make_graph_tensor(graph, reduced_graph, "mean_buf");
        auto *grad_temp = make_graph_tensor(graph, input_graph, "grad_temp");
        auto *tmp_grad = make_graph_tensor(graph, input_graph, "tmp_grad");

        if (has_weight)
        {
            auto *weight_node = get_or_create_data_node(
                const_cast<float *>(weight_data),
                {norm_len},
                nntile::DataType::FP32,
                true);
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
        register_data_node(grad_input_data, grad_input_node);
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
    const float *grad_data,
    float *first_moment_data,
    float *second_moment_data,
    float *param_data,
    c10::IntArrayRef pytorch_shape)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(pytorch_shape);

    auto *grad_node = get_or_create_data_node(
        const_cast<float *>(grad_data),
        graph_shape,
        nntile::DataType::FP32,
        true);
    auto *first_moment_node = get_or_create_data_node(
        first_moment_data,
        graph_shape,
        nntile::DataType::FP32,
        true);
    auto *second_moment_node = get_or_create_data_node(
        second_moment_data,
        graph_shape,
        nntile::DataType::FP32,
        true);
    auto *param_node = get_or_create_data_node(
        param_data,
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

    register_data_node(first_moment_data, first_moment_node);
    register_data_node(second_moment_data, second_moment_node);
    register_data_node(param_data, param_node);
    maybe_execute_after_record();
}

void tensor_adamw_step_fp32(
    int64_t num_iter,
    float beta_1,
    float beta_2,
    float eps,
    float lr,
    float weight_decay,
    const float *grad_data,
    float *first_moment_data,
    float *second_moment_data,
    float *param_data,
    c10::IntArrayRef pytorch_shape)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(pytorch_shape);

    auto *grad_node = get_or_create_data_node(
        const_cast<float *>(grad_data),
        graph_shape,
        nntile::DataType::FP32,
        true);
    auto *first_moment_node = get_or_create_data_node(
        first_moment_data,
        graph_shape,
        nntile::DataType::FP32,
        true);
    auto *second_moment_node = get_or_create_data_node(
        second_moment_data,
        graph_shape,
        nntile::DataType::FP32,
        true);
    auto *param_node = get_or_create_data_node(
        param_data,
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

    register_data_node(first_moment_data, first_moment_node);
    register_data_node(second_moment_data, second_moment_node);
    register_data_node(param_data, param_node);
    maybe_execute_after_record();
}

void tensor_cat_fp32(
    const std::vector<const float *> &input_data,
    const std::vector<c10::IntArrayRef> &input_shapes,
    float *out_data,
    c10::IntArrayRef out_shape,
    int64_t dim)
{
    (void) out_shape;
    const nntile::Index axis = static_cast<nntile::Index>(dim);

    const std::vector<nntile::Index> first_graph =
        pytorch_shape_to_graph(input_shapes[0]);
    auto *acc_node = get_or_create_data_node(
        const_cast<float *>(input_data[0]),
        first_graph,
        nntile::DataType::FP32,
        true);

    for (std::size_t i = 1; i < input_data.size(); ++i)
    {
        const std::vector<nntile::Index> shape_graph =
            pytorch_shape_to_graph(input_shapes[i]);
        auto *next_node = get_or_create_data_node(
            const_cast<float *>(input_data[i]),
            shape_graph,
            nntile::DataType::FP32,
            true);
        acc_node = nntile::tensor::concat(
            acc_node,
            next_node,
            axis)->set_name("cat");
    }

    register_data_node(out_data, acc_node);
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
    const float * /*x_data*/,
    float /*beta*/,
    const float * /*y_data*/,
    float * /*out_data*/,
    c10::IntArrayRef /*pytorch_shape*/)
{
    require_libnntile("add");
}

void tensor_add_inplace_fp32(
    float /*alpha*/,
    const float * /*other_data*/,
    float /*beta*/,
    float * /*self_data*/,
    c10::IntArrayRef /*pytorch_shape*/)
{
    require_libnntile("add_");
}

void tensor_mul_fp32(
    const float * /*self_data*/,
    const float * /*other_data*/,
    float * /*out_data*/,
    c10::IntArrayRef /*pytorch_shape*/)
{
    require_libnntile("mul");
}

void tensor_mul_inplace_fp32(
    const float * /*other_data*/,
    float * /*self_data*/,
    c10::IntArrayRef /*pytorch_shape*/)
{
    require_libnntile("mul_");
}

void tensor_hypot_fp32(
    const float * /*self_data*/,
    const float * /*other_data*/,
    float * /*out_data*/,
    c10::IntArrayRef /*pytorch_shape*/)
{
    require_libnntile("hypot");
}

void tensor_linear_fp32(
    const float * /*input_data*/,
    c10::IntArrayRef /*input_shape*/,
    const float * /*weight_data*/,
    c10::IntArrayRef /*weight_shape*/,
    float * /*out_data*/,
    c10::IntArrayRef /*out_shape*/)
{
    require_libnntile("linear");
}

void tensor_relu_fp32(
    const float * /*input_data*/,
    float * /*out_data*/,
    c10::IntArrayRef /*pytorch_shape*/)
{
    require_libnntile("relu");
}

void tensor_relu_backward_fp32(
    const float * /*x_data*/,
    const float * /*dy_data*/,
    float * /*dx_data*/,
    c10::IntArrayRef /*pytorch_shape*/)
{
    require_libnntile("relu_backward");
}

void tensor_silu_fp32(
    const float * /*input_data*/,
    float * /*out_data*/,
    c10::IntArrayRef /*pytorch_shape*/)
{
    require_libnntile("silu");
}

void tensor_silu_inplace_fp32(
    float * /*data*/,
    c10::IntArrayRef /*pytorch_shape*/)
{
    require_libnntile("silu_inplace");
}

void tensor_silu_backward_fp32(
    const float * /*x_data*/,
    const float * /*dy_data*/,
    float * /*dx_data*/,
    c10::IntArrayRef /*pytorch_shape*/)
{
    require_libnntile("silu_backward");
}

void tensor_gelu_fp32(
    const float * /*input_data*/,
    float * /*out_data*/,
    c10::IntArrayRef /*pytorch_shape*/,
    bool /*approximate_tanh*/)
{
    require_libnntile("gelu");
}

void tensor_gelu_inplace_fp32(
    float * /*data*/,
    c10::IntArrayRef /*pytorch_shape*/,
    bool /*approximate_tanh*/)
{
    require_libnntile("gelu_inplace");
}

void tensor_gelu_backward_fp32(
    const float * /*x_data*/,
    const float * /*dy_data*/,
    float * /*dx_data*/,
    c10::IntArrayRef /*pytorch_shape*/,
    bool /*approximate_tanh*/)
{
    require_libnntile("gelu_backward");
}

void tensor_mm_fp32(
    const float * /*a_data*/,
    c10::IntArrayRef /*a_shape*/,
    const float * /*b_data*/,
    c10::IntArrayRef /*b_shape*/,
    float * /*out_data*/,
    c10::IntArrayRef /*out_shape*/)
{
    require_libnntile("mm");
}

void tensor_linear_backward_input_fp32(
    const float * /*grad_out_data*/,
    c10::IntArrayRef /*grad_out_shape*/,
    const float * /*weight_data*/,
    c10::IntArrayRef /*weight_shape*/,
    float * /*grad_input_data*/,
    c10::IntArrayRef /*grad_input_shape*/)
{
    require_libnntile("linear_backward_input");
}

void tensor_linear_backward_weight_fp32(
    const float * /*grad_out_data*/,
    c10::IntArrayRef /*grad_out_shape*/,
    const float * /*input_data*/,
    c10::IntArrayRef /*input_shape*/,
    float * /*grad_weight_data*/,
    c10::IntArrayRef /*grad_weight_shape*/)
{
    require_libnntile("linear_backward_weight");
}

void tensor_cross_entropy_forward_fp32(
    const float * /*logits_data*/,
    c10::IntArrayRef /*logits_shape*/,
    const std::int64_t * /*labels_data*/,
    c10::IntArrayRef /*labels_shape*/,
    std::int64_t /*ignore_index*/,
    bool /*mean_reduction*/,
    float * /*loss_data*/)
{
    require_libnntile("cross_entropy_forward");
}

void tensor_cross_entropy_backward_fp32(
    const float * /*logits_data*/,
    c10::IntArrayRef /*logits_shape*/,
    const std::int64_t * /*labels_data*/,
    c10::IntArrayRef /*labels_shape*/,
    const float * /*grad_output_data*/,
    float * /*grad_row_data*/,
    float * /*grad_logits_data*/,
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
    const float * /*grad_data*/,
    float * /*velocity_data*/,
    float * /*param_data*/,
    c10::IntArrayRef /*pytorch_shape*/)
{
    require_libnntile("sgd_step");
}

void tensor_layer_norm_forward_fp32(
    const float * /*input_data*/,
    c10::IntArrayRef /*input_shape*/,
    const float * /*weight_data*/,
    const float * /*bias_data*/,
    bool /*has_weight*/,
    bool /*has_bias*/,
    float * /*output_data*/,
    float * /*mean_data*/,
    float * /*rstd_data*/,
    int64_t /*norm_axis*/,
    float /*eps*/)
{
    require_libnntile("layer_norm_forward");
}

void tensor_layer_norm_backward_fp32(
    const float * /*grad_out_data*/,
    const float * /*input_data*/,
    const float * /*mean_data*/,
    const float * /*rstd_data*/,
    const float * /*weight_data*/,
    bool /*has_weight*/,
    bool /*has_bias*/,
    float * /*grad_input_data*/,
    float * /*grad_weight_data*/,
    float * /*grad_bias_data*/,
    bool /*grad_input_needed*/,
    bool /*grad_weight_needed*/,
    bool /*grad_bias_needed*/,
    c10::IntArrayRef /*input_shape*/,
    int64_t /*norm_axis*/)
{
    require_libnntile("layer_norm_backward");
}

void tensor_rms_norm_forward_fp32(
    const float * /*input_data*/,
    c10::IntArrayRef /*input_shape*/,
    const float * /*weight_data*/,
    bool /*has_weight*/,
    float * /*output_data*/,
    float * /*rstd_data*/,
    int64_t /*norm_axis*/,
    float /*eps*/)
{
    require_libnntile("rms_norm_forward");
}

void tensor_rms_norm_backward_fp32(
    const float * /*grad_out_data*/,
    const float * /*input_data*/,
    const float * /*rstd_data*/,
    const float * /*weight_data*/,
    bool /*has_weight*/,
    float * /*grad_input_data*/,
    float * /*grad_weight_data*/,
    bool /*grad_input_needed*/,
    bool /*grad_weight_needed*/,
    c10::IntArrayRef /*input_shape*/,
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
    const float * /*grad_data*/,
    float * /*first_moment_data*/,
    float * /*second_moment_data*/,
    float * /*param_data*/,
    c10::IntArrayRef /*pytorch_shape*/)
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
    const float * /*grad_data*/,
    float * /*first_moment_data*/,
    float * /*second_moment_data*/,
    float * /*param_data*/,
    c10::IntArrayRef /*pytorch_shape*/)
{
    require_libnntile("adamw_step");
}

void tensor_cat_fp32(
    const std::vector<const float *> & /*input_data*/,
    const std::vector<c10::IntArrayRef> & /*input_shapes*/,
    float * /*out_data*/,
    c10::IntArrayRef /*out_shape*/,
    int64_t /*dim*/)
{
    require_libnntile("cat");
}

} // namespace torch_nntile

#endif
