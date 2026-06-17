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
#include <nntile/tensor/ops/gemm.hh>
#include <nntile/tensor/ops/relu.hh>
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

} // namespace torch_nntile

#endif
