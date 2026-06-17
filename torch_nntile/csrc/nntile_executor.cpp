/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_executor.cpp
 */

#include "nntile_executor.h"

#include "nntile_context.h"

#ifdef TORCH_NNTILE_USE_LIBNNTILE

#include <nntile/base_types.hh>
#include <nntile/nn/shape_layout.hh>
#include <nntile/runtime.hh>
#include <nntile/tensor/ops/add.hh>
#include <nntile/tensor/ops/clear.hh>
#include <nntile/tensor/ops/gemm.hh>
#include <nntile/tensor/ops/relu.hh>
#include <nntile/tensor/ops/relu_backward.hh>
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

std::vector<nntile::Index> pytorch_shape_to_storage(c10::IntArrayRef shape)
{
    std::vector<nntile::Index> graph_shape;
    graph_shape.reserve(shape.size());
    for (const auto dim : shape)
    {
        graph_shape.push_back(static_cast<nntile::Index>(dim));
    }
    return nntile::nn::graph_shape_to_storage(graph_shape);
}

nntile::Index storage_numel(const std::vector<nntile::Index> &storage_shape)
{
    nntile::Index nelems = 1;
    for (const nntile::Index dim : storage_shape)
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
    const std::vector<nntile::Index> &out_storage_shape)
{
    ensure_nntile_context();

    nntile::TileGraph tile_graph =
        nntile::TileGraph::from_tensor_graph(*out_node->graph());
    nntile::Runtime runtime(tile_graph);
    runtime.compile();

    for (const auto &[node, data] : inputs)
    {
        const nntile::Index count = storage_numel(node->shape());
        runtime.bind_data(node, data, static_cast<std::size_t>(count));
    }
    runtime.execute();
    runtime.wait();

    const std::vector<float> result = runtime.get_output<float>(out_node);
    const std::size_t expected =
        static_cast<std::size_t>(storage_numel(out_storage_shape));
    if (result.size() != expected)
    {
        throw std::runtime_error(
            std::string(graph_name) + ": output size mismatch");
    }
    std::memcpy(out_data, result.data(), expected * sizeof(float));
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
    const std::vector<nntile::Index> storage_shape =
        pytorch_shape_to_storage(pytorch_shape);

    nntile::TensorGraph graph("torch_add");
    auto *x_node =
        graph.data(storage_shape, nntile::DataType::FP32)->set_name("x");
    auto *y_node =
        graph.data(storage_shape, nntile::DataType::FP32)->set_name("y");
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
        storage_shape);
}

void tensor_linear_fp32(
    const float *input_data,
    c10::IntArrayRef input_shape,
    const float *weight_data,
    c10::IntArrayRef weight_shape,
    float *out_data,
    c10::IntArrayRef out_shape)
{
    const std::vector<nntile::Index> input_storage =
        pytorch_shape_to_storage(input_shape);
    const std::vector<nntile::Index> weight_storage =
        pytorch_shape_to_storage(weight_shape);
    const std::vector<nntile::Index> out_storage =
        pytorch_shape_to_storage(out_shape);

    nntile::TensorGraph graph("torch_linear");
    auto *input_node =
        graph.data(input_storage, nntile::DataType::FP32)->set_name("input");
    auto *weight_node =
        graph.data(weight_storage, nntile::DataType::FP32)->set_name("weight");
    input_node->mark_input(true);
    weight_node->mark_input(true);

    auto *out_node = nntile::tensor::gemm(
        weight_node,
        input_node,
        static_cast<nntile::Scalar>(1.0),
        true,
        false,
        static_cast<nntile::Index>(1),
        static_cast<nntile::Index>(0))->set_name("output");
    out_node->mark_output(true);

    run_graph_output(
        "torch_linear",
        out_node,
        {{input_node, input_data}, {weight_node, weight_data}},
        out_data,
        out_storage);
}

void tensor_relu_fp32(
    const float *input_data,
    float *out_data,
    c10::IntArrayRef pytorch_shape)
{
    const std::vector<nntile::Index> storage_shape =
        pytorch_shape_to_storage(pytorch_shape);

    nntile::TensorGraph graph("torch_relu");
    auto *src_node =
        graph.data(storage_shape, nntile::DataType::FP32)->set_name("src");
    src_node->mark_input(true);

    auto *dst_node = nntile::tensor::relu(src_node)->set_name("dst");
    dst_node->mark_output(true);

    run_graph_output(
        "torch_relu",
        dst_node,
        {{src_node, input_data}},
        out_data,
        storage_shape);
}

void tensor_relu_backward_fp32(
    const float *x_data,
    const float *dy_data,
    float *dx_data,
    c10::IntArrayRef pytorch_shape)
{
    const std::vector<nntile::Index> storage_shape =
        pytorch_shape_to_storage(pytorch_shape);

    nntile::TensorGraph graph("torch_relu_backward");
    auto *x_node =
        graph.data(storage_shape, nntile::DataType::FP32)->set_name("x");
    auto *dy_node =
        graph.data(storage_shape, nntile::DataType::FP32)->set_name("dy");
    auto *dx_node =
        graph.data(storage_shape, nntile::DataType::FP32)->set_name("dx");
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

    const nntile::Index count = storage_numel(storage_shape);
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
    const std::vector<nntile::Index> a_storage =
        pytorch_shape_to_storage(a_shape);
    const std::vector<nntile::Index> b_storage =
        pytorch_shape_to_storage(b_shape);
    const std::vector<nntile::Index> out_storage =
        pytorch_shape_to_storage(out_shape);

    nntile::TensorGraph graph("torch_mm");
    auto *a_node =
        graph.data(a_storage, nntile::DataType::FP32)->set_name("a");
    auto *b_node =
        graph.data(b_storage, nntile::DataType::FP32)->set_name("b");
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
        out_storage);
}

void tensor_linear_backward_input_fp32(
    const float *grad_out_data,
    c10::IntArrayRef grad_out_shape,
    const float *weight_data,
    c10::IntArrayRef weight_shape,
    float *grad_input_data,
    c10::IntArrayRef grad_input_shape)
{
    const std::vector<nntile::Index> grad_out_storage =
        pytorch_shape_to_storage(grad_out_shape);
    const std::vector<nntile::Index> weight_storage =
        pytorch_shape_to_storage(weight_shape);
    const std::vector<nntile::Index> grad_input_storage =
        pytorch_shape_to_storage(grad_input_shape);

    nntile::TensorGraph graph("torch_linear_backward_input");
    auto *grad_out_node = graph.data(grad_out_storage, nntile::DataType::FP32)
                              ->set_name("grad_out");
    auto *weight_node =
        graph.data(weight_storage, nntile::DataType::FP32)->set_name("weight");
    grad_out_node->mark_input(true);
    weight_node->mark_input(true);

    auto *grad_input_node = nntile::tensor::gemm(
        weight_node,
        grad_out_node,
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
        grad_input_storage);
}

void tensor_linear_backward_weight_fp32(
    const float *grad_out_data,
    c10::IntArrayRef grad_out_shape,
    const float *input_data,
    c10::IntArrayRef input_shape,
    float *grad_weight_data,
    c10::IntArrayRef grad_weight_shape)
{
    const std::vector<nntile::Index> grad_out_storage =
        pytorch_shape_to_storage(grad_out_shape);
    const std::vector<nntile::Index> input_storage =
        pytorch_shape_to_storage(input_shape);
    const std::vector<nntile::Index> grad_weight_storage =
        pytorch_shape_to_storage(grad_weight_shape);

    nntile::TensorGraph graph("torch_linear_backward_weight");
    auto *grad_out_node = graph.data(grad_out_storage, nntile::DataType::FP32)
                              ->set_name("grad_out");
    auto *input_node =
        graph.data(input_storage, nntile::DataType::FP32)->set_name("input");
    grad_out_node->mark_input(true);
    input_node->mark_input(true);

    auto *grad_weight_node = nntile::tensor::gemm(
        input_node,
        grad_out_node,
        static_cast<nntile::Scalar>(1.0),
        false,
        true,
        static_cast<nntile::Index>(1),
        static_cast<nntile::Index>(0))->set_name("grad_weight");
    grad_weight_node->mark_output(true);

    run_graph_output(
        "torch_linear_backward_weight",
        grad_weight_node,
        {{grad_out_node, grad_out_data}, {input_node, input_data}},
        grad_weight_data,
        grad_weight_storage);
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

} // namespace torch_nntile

#endif
