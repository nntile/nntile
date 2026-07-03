/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_executor.cpp
 */

#include "nntile_executor.h"

#include "nntile_gemm_layout.h"
#include "nntile_context.h"
#include "nntile_graph_recorder.h"
#include "nntile_graph_recorder_impl.h"

#ifdef TORCH_NNTILE_USE_LIBNNTILE

#include <nntile/base_types.hh>
#include <nntile/tensor/ops/add.hh>
#include <nntile/tensor/ops/add_inplace.hh>
#include <nntile/tensor/ops/multiply.hh>
#include <nntile/tensor/ops/multiply_inplace.hh>
#include <nntile/tensor/ops/clear.hh>
#include <nntile/tensor/ops/gemm.hh>
#include <nntile/tensor/ops/logsumexp.hh>
#include <nntile/tensor/ops/maxsumexp.hh>
#include <nntile/tensor/ops/gelu.hh>
#include <nntile/tensor/ops/gelu_backward.hh>
#include <nntile/tensor/ops/gelu_inplace.hh>
#include <nntile/tensor/ops/gelutanh.hh>
#include <nntile/tensor/ops/gelutanh_backward.hh>
#include <nntile/tensor/ops/gelutanh_inplace.hh>
#include <nntile/tensor/ops/relu.hh>
#include <nntile/tensor/ops/relu_backward.hh>
#include <nntile/tensor/ops/adam_step.hh>
#include <nntile/tensor/ops/adamw_step.hh>
#include <nntile/tensor/ops/silu.hh>
#include <nntile/tensor/ops/silu_backward.hh>
#include <nntile/tensor/ops/silu_inplace.hh>
#include <nntile/tensor/ops/sgd_step.hh>
#include <nntile/tensor/ops/multiply_slice.hh>
#include <nntile/tensor/ops/scale_slice.hh>
#include <nntile/tensor/ops/softmax.hh>
#include <nntile/tensor/ops/subtract_indexed_outputs.hh>
#include <nntile/tensor/ops/total_sum_accum.hh>

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

void tensor_gemm_fp32(
    const GemmParams &params,
    const float *a_data,
    c10::IntArrayRef a_gemm_shape,
    const float *b_data,
    c10::IntArrayRef b_gemm_shape,
    float *out_data,
    c10::IntArrayRef out_shape)
{
    const std::vector<nntile::Index> a_graph =
        pytorch_shape_to_graph(a_gemm_shape);
    const std::vector<nntile::Index> b_graph =
        pytorch_shape_to_graph(b_gemm_shape);

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

    nntile::TensorGraph::TensorNode *out_node = nullptr;
    out_node = nntile::tensor::gemm(
        a_node,
        b_node,
        static_cast<nntile::Scalar>(params.alpha),
        params.trans_a,
        params.trans_b,
        static_cast<nntile::Index>(params.ndim),
        static_cast<nntile::Index>(params.batch_ndim))->set_name("out");
    register_data_node(out_data, out_node);
    maybe_execute_after_record();
}

void tensor_gemm_accumulate_fp32(
    const GemmParams &params,
    const float *a_data,
    c10::IntArrayRef a_gemm_shape,
    const float *b_data,
    c10::IntArrayRef b_gemm_shape,
    const float *c_data,
    c10::IntArrayRef c_shape,
    float *out_data,
    c10::IntArrayRef out_shape)
{
    const std::vector<nntile::Index> a_graph =
        pytorch_shape_to_graph(a_gemm_shape);
    const std::vector<nntile::Index> b_graph =
        pytorch_shape_to_graph(b_gemm_shape);
    const std::vector<nntile::Index> c_graph =
        pytorch_shape_to_graph(c_shape);

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
    auto *c_node = get_or_create_data_node(
        const_cast<float *>(c_data),
        c_graph,
        nntile::DataType::FP32,
        true);

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
    register_data_node(out_data, c_node);
    maybe_execute_after_record();
}

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

void tensor_linear_fp32(
    const float *input_data,
    c10::IntArrayRef input_shape,
    const float *weight_data,
    c10::IntArrayRef weight_shape,
    float *out_data,
    c10::IntArrayRef out_shape)
{
    GemmParams params;
    params.trans_a = false;
    params.trans_b = true;
    params.ndim = 1;
    params.batch_ndim = 0;
    tensor_gemm_fp32(
        params,
        input_data,
        input_shape,
        weight_data,
        weight_shape,
        out_data,
        out_shape);
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
    GemmParams params;
    tensor_gemm_fp32(
        params,
        a_data,
        a_shape,
        b_data,
        b_shape,
        out_data,
        out_shape);
}

void tensor_linear_backward_input_fp32(
    const float *grad_out_data,
    c10::IntArrayRef grad_out_shape,
    const float *weight_data,
    c10::IntArrayRef weight_shape,
    float *grad_input_data,
    c10::IntArrayRef grad_input_shape)
{
    GemmParams params;
    params.trans_a = false;
    params.trans_b = false;
    params.ndim = 1;
    params.batch_ndim = grad_out_shape.size() > 1
        ? static_cast<int64_t>(grad_out_shape.size()) - 1
        : 0;
    tensor_gemm_fp32(
        params,
        grad_out_data,
        grad_out_shape,
        weight_data,
        weight_shape,
        grad_input_data,
        grad_input_shape);
}

void tensor_linear_backward_weight_fp32(
    const float *grad_out_data,
    c10::IntArrayRef grad_out_shape,
    const float *input_data,
    c10::IntArrayRef input_shape,
    float *grad_weight_data,
    c10::IntArrayRef grad_weight_shape)
{
    GemmParams params;
    params.trans_a = true;
    params.trans_b = false;
    params.ndim = 1;
    params.batch_ndim = grad_out_shape.size() > 1
        ? static_cast<int64_t>(grad_out_shape.size()) - 1
        : 0;
    tensor_gemm_fp32(
        params,
        grad_out_data,
        grad_out_shape,
        input_data,
        input_shape,
        grad_weight_data,
        grad_weight_shape);
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

void tensor_gemm_fp32(
    const GemmParams & /*params*/,
    const float * /*a_data*/,
    c10::IntArrayRef /*a_gemm_shape*/,
    const float * /*b_data*/,
    c10::IntArrayRef /*b_gemm_shape*/,
    float * /*out_data*/,
    c10::IntArrayRef /*out_shape*/)
{
    require_libnntile("gemm");
}

void tensor_gemm_accumulate_fp32(
    const GemmParams & /*params*/,
    const float * /*a_data*/,
    c10::IntArrayRef /*a_gemm_shape*/,
    const float * /*b_data*/,
    c10::IntArrayRef /*b_gemm_shape*/,
    const float * /*c_data*/,
    c10::IntArrayRef /*c_shape*/,
    float * /*out_data*/,
    c10::IntArrayRef /*out_shape*/)
{
    require_libnntile("gemm_accumulate");
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

} // namespace torch_nntile

#endif
