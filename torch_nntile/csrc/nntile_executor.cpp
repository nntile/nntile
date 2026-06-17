/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_executor.cpp
 */

#include "nntile_executor.h"

#include "nntile_context.h"

#ifdef TORCH_NNTILE_USE_LIBNNTILE

#include <nntile/base_types.hh>
#include <nntile/runtime.hh>
#include <nntile/tensor/ops/add.hh>
#include <nntile/tensor/ops/clear.hh>
#include <nntile/tensor/ops/gemm.hh>
#include <nntile/tensor/ops/logsumexp.hh>
#include <nntile/tensor/ops/maxsumexp.hh>
#include <nntile/tensor/ops/relu.hh>
#include <nntile/tensor/ops/relu_backward.hh>
#include <nntile/tensor/ops/sgd_step.hh>
#include <nntile/tensor/ops/softmax.hh>
#include <nntile/tensor/ops/subtract_indexed_outputs.hh>
#include <nntile/tensor/ops/total_sum_accum.hh>
#include <nntile/tile/graph.hh>

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
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

nntile::Index graph_numel(const std::vector<nntile::Index> &graph_shape)
{
    nntile::Index nelems = 1;
    for (const nntile::Index dim : graph_shape)
    {
        nelems *= dim;
    }
    return nelems;
}

void run_graph_output(
    const char *graph_name,
    nntile::TensorGraph::TensorNode *out_node,
    const std::vector<std::pair<nntile::TensorGraph::TensorNode *, const float *>>
        &inputs,
    float *out_data,
    const std::vector<nntile::Index> &out_graph_shape)
{
    ensure_nntile_context();

    nntile::TileGraph tile_graph =
        nntile::TileGraph::from_tensor_graph(*out_node->graph());
    nntile::Runtime runtime(tile_graph);
    runtime.compile();

    const std::size_t expected =
        static_cast<std::size_t>(graph_numel(out_graph_shape));
    out_node->mark_input(true);

    for (const auto &[node, data] : inputs)
    {
        const nntile::Index count = graph_numel(node->shape());
        runtime.bind_data(node, data, static_cast<std::size_t>(count));
    }
    runtime.bind_data(out_node, out_data, expected);
    runtime.execute();
    runtime.wait();

    const std::vector<float> result = runtime.get_output<float>(out_node);
    if (result.size() != expected)
    {
        throw std::runtime_error(
            std::string(graph_name) + ": output size mismatch");
    }
    if (result.data() != out_data)
    {
        std::memcpy(out_data, result.data(), expected * sizeof(float));
    }
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

    nntile::TensorGraph graph("torch_add");
    auto *x_node =
        graph.data(graph_shape, nntile::DataType::FP32)->set_name("x");
    auto *y_node =
        graph.data(graph_shape, nntile::DataType::FP32)->set_name("y");
    x_node->mark_input(true);
    y_node->mark_input(true);

    auto *z_node = nntile::tensor::add(
        static_cast<nntile::Scalar>(alpha),
        x_node,
        static_cast<nntile::Scalar>(beta),
        y_node)->set_name("z");
    z_node->mark_output(true);

    run_graph_output(
        "torch_add",
        z_node,
        {{x_node, x_data}, {y_node, y_data}},
        out_data,
        graph_shape);
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
    const std::vector<nntile::Index> out_graph =
        pytorch_shape_to_graph(out_shape);

    nntile::TensorGraph graph("torch_linear");
    auto *input_node =
        graph.data(input_graph, nntile::DataType::FP32)->set_name("input");
    auto *weight_node =
        graph.data(weight_graph, nntile::DataType::FP32)->set_name("weight");
    input_node->mark_input(true);
    weight_node->mark_input(true);

    auto *out_node = nntile::tensor::gemm(
        input_node,
        weight_node,
        static_cast<nntile::Scalar>(1.0),
        false,
        true,
        static_cast<nntile::Index>(1),
        static_cast<nntile::Index>(0))->set_name("output");
    out_node->mark_output(true);

    run_graph_output(
        "torch_linear",
        out_node,
        {{input_node, input_data}, {weight_node, weight_data}},
        out_data,
        out_graph);
}

void tensor_relu_fp32(
    const float *input_data,
    float *out_data,
    c10::IntArrayRef pytorch_shape)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(pytorch_shape);

    nntile::TensorGraph graph("torch_relu");
    auto *src_node =
        graph.data(graph_shape, nntile::DataType::FP32)->set_name("src");
    src_node->mark_input(true);

    auto *dst_node = nntile::tensor::relu(src_node)->set_name("dst");
    dst_node->mark_output(true);

    run_graph_output(
        "torch_relu",
        dst_node,
        {{src_node, input_data}},
        out_data,
        graph_shape);
}

void tensor_relu_backward_fp32(
    const float *x_data,
    const float *dy_data,
    float *dx_data,
    c10::IntArrayRef pytorch_shape)
{
    const std::vector<nntile::Index> graph_shape =
        pytorch_shape_to_graph(pytorch_shape);

    nntile::TensorGraph graph("torch_relu_backward");
    auto *x_node =
        graph.data(graph_shape, nntile::DataType::FP32)->set_name("x");
    auto *dy_node =
        graph.data(graph_shape, nntile::DataType::FP32)->set_name("dy");
    auto *dx_node =
        graph.data(graph_shape, nntile::DataType::FP32)->set_name("dx");
    x_node->mark_input(true);
    dy_node->mark_input(true);
    dx_node->mark_input(true);
    dx_node->mark_output(true);

    nntile::tensor::clear(dx_node);
    nntile::tensor::relu_backward(x_node, dy_node, dx_node);

    ensure_nntile_context();

    nntile::TileGraph tile_graph =
        nntile::TileGraph::from_tensor_graph(graph);
    nntile::Runtime runtime(tile_graph);
    runtime.compile();

    const nntile::Index count = graph_numel(graph_shape);
    std::vector<float> dx_init(static_cast<std::size_t>(count), 0.0f);
    runtime.bind_data(x_node, x_data, static_cast<std::size_t>(count));
    runtime.bind_data(dy_node, dy_data, static_cast<std::size_t>(count));
    runtime.bind_data(dx_node, dx_init.data(), static_cast<std::size_t>(count));
    runtime.execute();
    runtime.wait();

    const std::vector<float> result = runtime.get_output<float>(dx_node);
    const std::size_t expected = static_cast<std::size_t>(count);
    if (result.size() != expected)
    {
        throw std::runtime_error("torch_relu_backward: output size mismatch");
    }
    std::memcpy(dx_data, result.data(), expected * sizeof(float));
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
    const std::vector<nntile::Index> out_graph =
        pytorch_shape_to_graph(out_shape);

    nntile::TensorGraph graph("torch_mm");
    auto *a_node =
        graph.data(a_graph, nntile::DataType::FP32)->set_name("a");
    auto *b_node =
        graph.data(b_graph, nntile::DataType::FP32)->set_name("b");
    a_node->mark_input(true);
    b_node->mark_input(true);

    auto *out_node = nntile::tensor::gemm(
        a_node,
        b_node,
        static_cast<nntile::Scalar>(1.0),
        false,
        false,
        static_cast<nntile::Index>(1),
        static_cast<nntile::Index>(0))->set_name("out");
    out_node->mark_output(true);

    run_graph_output(
        "torch_mm",
        out_node,
        {{a_node, a_data}, {b_node, b_data}},
        out_data,
        out_graph);
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
    const std::vector<nntile::Index> grad_input_graph =
        pytorch_shape_to_graph(grad_input_shape);

    nntile::TensorGraph graph("torch_linear_backward_input");
    auto *grad_out_node = graph.data(grad_out_graph, nntile::DataType::FP32)
                              ->set_name("grad_out");
    auto *weight_node =
        graph.data(weight_graph, nntile::DataType::FP32)->set_name("weight");
    grad_out_node->mark_input(true);
    weight_node->mark_input(true);

    auto *grad_input_node = nntile::tensor::gemm(
        grad_out_node,
        weight_node,
        static_cast<nntile::Scalar>(1.0),
        false,
        false,
        static_cast<nntile::Index>(1),
        static_cast<nntile::Index>(0))->set_name("grad_input");
    grad_input_node->mark_output(true);

    run_graph_output(
        "torch_linear_backward_input",
        grad_input_node,
        {{grad_out_node, grad_out_data}, {weight_node, weight_data}},
        grad_input_data,
        grad_input_graph);
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
    const std::vector<nntile::Index> grad_weight_graph =
        pytorch_shape_to_graph(grad_weight_shape);

    nntile::TensorGraph graph("torch_linear_backward_weight");
    auto *grad_out_node = graph.data(grad_out_graph, nntile::DataType::FP32)
                              ->set_name("grad_out");
    auto *input_node =
        graph.data(input_graph, nntile::DataType::FP32)->set_name("input");
    grad_out_node->mark_input(true);
    input_node->mark_input(true);

    auto *grad_weight_node = nntile::tensor::gemm(
        grad_out_node,
        input_node,
        static_cast<nntile::Scalar>(1.0),
        true,
        false,
        static_cast<nntile::Index>(1),
        static_cast<nntile::Index>(0))->set_name("grad_weight");
    grad_weight_node->mark_output(true);

    run_graph_output(
        "torch_linear_backward_weight",
        grad_weight_node,
        {{grad_out_node, grad_out_data}, {input_node, input_data}},
        grad_weight_data,
        grad_weight_graph);
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

float tensor_cross_entropy_forward_fp32(
    const float *logits_data,
    c10::IntArrayRef logits_shape,
    const std::int64_t *labels_data,
    c10::IntArrayRef labels_shape,
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
    const float scale = cross_entropy_scale(
        labels_data,
        labels_shape,
        ignore_index,
        mean_reduction);

    nntile::TensorGraph graph("torch_cross_entropy_forward");
    auto *logits_node =
        graph.data(logits_graph, nntile::DataType::FP32)->set_name("logits");
    auto *labels_node = graph.data(labels_graph, nntile::DataType::INT64)
                              ->set_name("labels");
    auto *maxsumexp_node =
        graph.data(maxsumexp_graph, nntile::DataType::FP32)
            ->set_name("maxsumexp");
    auto *logsumexp_node =
        graph.data(labels_graph, nntile::DataType::FP32)->set_name("logsumexp");
    auto *loss_node = graph.data({}, nntile::DataType::FP32)->set_name("loss");

    logits_node->mark_input(true);
    labels_node->mark_input(true);
    loss_node->mark_input(true);
    loss_node->mark_output(true);

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

    ensure_nntile_context();

    nntile::TileGraph tile_graph =
        nntile::TileGraph::from_tensor_graph(graph);
    nntile::Runtime runtime(tile_graph);
    runtime.compile();

    const nntile::Index logits_count = graph_numel(logits_graph);
    const nntile::Index labels_count = graph_numel(labels_graph);
    float loss_init = 0.0f;
    runtime.bind_data(
        logits_node,
        logits_data,
        static_cast<std::size_t>(logits_count));
    runtime.bind_data(
        labels_node,
        labels_data,
        static_cast<std::size_t>(labels_count));
    runtime.bind_data(loss_node, &loss_init, 1);
    runtime.execute();
    runtime.wait();

    const std::vector<float> result = runtime.get_output<float>(loss_node);
    if (result.size() != 1)
    {
        throw std::runtime_error(
            "torch_cross_entropy_forward: expected scalar loss");
    }
    return result[0];
}

void tensor_cross_entropy_backward_fp32(
    const float *logits_data,
    c10::IntArrayRef logits_shape,
    const std::int64_t *labels_data,
    c10::IntArrayRef labels_shape,
    float grad_output,
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
    const float scale =
        cross_entropy_scale(
            labels_data,
            labels_shape,
            ignore_index,
            mean_reduction)
        * grad_output;

    nntile::TensorGraph graph("torch_cross_entropy_backward");
    auto *logits_node =
        graph.data(logits_graph, nntile::DataType::FP32)->set_name("logits");
    auto *labels_node = graph.data(labels_graph, nntile::DataType::INT64)
                              ->set_name("labels");
    auto *maxsumexp_node =
        graph.data(maxsumexp_graph, nntile::DataType::FP32)
            ->set_name("maxsumexp");
    auto *grad_logits_node =
        graph.data(logits_graph, nntile::DataType::FP32)
            ->set_name("grad_logits");

    logits_node->mark_input(true);
    labels_node->mark_input(true);
    grad_logits_node->mark_input(true);
    grad_logits_node->mark_output(true);

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
        static_cast<nntile::Scalar>(scale),
        class_axis);
    nntile::tensor::subtract_indexed_outputs(
        static_cast<nntile::Scalar>(scale),
        labels_node,
        grad_logits_node,
        static_cast<nntile::Index>(ignore_index));

    ensure_nntile_context();

    nntile::TileGraph tile_graph =
        nntile::TileGraph::from_tensor_graph(graph);
    nntile::Runtime runtime(tile_graph);
    runtime.compile();

    const nntile::Index logits_count = graph_numel(logits_graph);
    const nntile::Index labels_count = graph_numel(labels_graph);
    std::vector<float> grad_init(static_cast<std::size_t>(logits_count), 0.0f);
    runtime.bind_data(
        logits_node,
        logits_data,
        static_cast<std::size_t>(logits_count));
    runtime.bind_data(
        labels_node,
        labels_data,
        static_cast<std::size_t>(labels_count));
    runtime.bind_data(
        grad_logits_node,
        grad_init.data(),
        static_cast<std::size_t>(logits_count));
    runtime.execute();
    runtime.wait();

    const std::vector<float> result =
        runtime.get_output<float>(grad_logits_node);
    const std::size_t expected = static_cast<std::size_t>(logits_count);
    if (result.size() != expected)
    {
        throw std::runtime_error(
            "torch_cross_entropy_backward: output size mismatch");
    }
    std::memcpy(grad_logits_data, result.data(), expected * sizeof(float));
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
    const nntile::Index nelems = graph_numel(graph_shape);

    nntile::TensorGraph graph("torch_sgd_step");
    auto *grad_node =
        graph.data(graph_shape, nntile::DataType::FP32)->set_name("grad");
    auto *velocity_node =
        graph.data(graph_shape, nntile::DataType::FP32)->set_name("velocity");
    auto *param_node =
        graph.data(graph_shape, nntile::DataType::FP32)->set_name("param");

    grad_node->mark_input(true);
    velocity_node->mark_input(true);
    param_node->mark_input(true);
    velocity_node->mark_output(true);
    param_node->mark_output(true);

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

    ensure_nntile_context();

    nntile::TileGraph tile_graph =
        nntile::TileGraph::from_tensor_graph(graph);
    nntile::Runtime runtime(tile_graph);
    runtime.compile();

    runtime.bind_data(
        grad_node,
        grad_data,
        static_cast<std::size_t>(nelems));
    runtime.bind_data(
        velocity_node,
        velocity_data,
        static_cast<std::size_t>(nelems));
    runtime.bind_data(
        param_node,
        param_data,
        static_cast<std::size_t>(nelems));
    runtime.execute();
    runtime.wait();

    const std::vector<float> velocity_out =
        runtime.get_output<float>(velocity_node);
    const std::vector<float> param_out =
        runtime.get_output<float>(param_node);
    const std::size_t expected = static_cast<std::size_t>(nelems);
    if (velocity_out.size() != expected || param_out.size() != expected)
    {
        throw std::runtime_error("torch_sgd_step: output size mismatch");
    }
    std::memcpy(velocity_data, velocity_out.data(), expected * sizeof(float));
    std::memcpy(param_data, param_out.data(), expected * sizeof(float));
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

float tensor_cross_entropy_forward_fp32(
    const float * /*logits_data*/,
    c10::IntArrayRef /*logits_shape*/,
    const std::int64_t * /*labels_data*/,
    c10::IntArrayRef /*labels_shape*/,
    std::int64_t /*ignore_index*/,
    bool /*mean_reduction*/)
{
    require_libnntile("cross_entropy_forward");
    return 0.0f;
}

void tensor_cross_entropy_backward_fp32(
    const float * /*logits_data*/,
    c10::IntArrayRef /*logits_shape*/,
    const std::int64_t * /*labels_data*/,
    c10::IntArrayRef /*labels_shape*/,
    float /*grad_output*/,
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

} // namespace torch_nntile

#endif
