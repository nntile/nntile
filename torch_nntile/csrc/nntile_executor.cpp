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
#include <nntile/tensor/ops/clear.hh>
#include <nntile/tensor/ops/gemm.hh>
#include <nntile/tensor/ops/logsumexp.hh>
#include <nntile/tensor/ops/maxsumexp.hh>
#include <nntile/tensor/ops/relu.hh>
#include <nntile/tensor/ops/relu_backward.hh>
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

} // namespace torch_nntile

#endif
