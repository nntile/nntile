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
#include <nntile/tile/graph.hh>

#include <cstring>
#include <stdexcept>
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

} // namespace

void tensor_add_fp32(
    float alpha,
    const float *x_data,
    float beta,
    const float *y_data,
    float *out_data,
    c10::IntArrayRef pytorch_shape)
{
    ensure_nntile_context();

    const std::vector<nntile::Index> storage_shape =
        pytorch_shape_to_storage(pytorch_shape);
    nntile::Index nelems = 1;
    for (const nntile::Index dim : storage_shape)
    {
        nelems *= dim;
    }

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

    nntile::TileGraph tile_graph =
        nntile::TileGraph::from_tensor_graph(graph);
    nntile::Runtime runtime(tile_graph);
    runtime.compile();

    const std::size_t count = static_cast<std::size_t>(nelems);
    runtime.bind_data(x_node, x_data, count);
    runtime.bind_data(y_node, y_data, count);
    runtime.execute();
    runtime.wait();

    const std::vector<float> result = runtime.get_output<float>(z_node);
    if (result.size() != count)
    {
        throw std::runtime_error("torch_nntile add: output size mismatch");
    }
    std::memcpy(out_data, result.data(), count * sizeof(float));
}

} // namespace torch_nntile

#else

#include <stdexcept>

namespace torch_nntile
{

void tensor_add_fp32(
    float /*alpha*/,
    const float * /*x_data*/,
    float /*beta*/,
    const float * /*y_data*/,
    float * /*out_data*/,
    c10::IntArrayRef /*pytorch_shape*/)
{
    throw std::runtime_error(
        "torch_nntile add requires libnntile "
        "(rebuild with NNTILE_BUILD_DIR set)");
}

} // namespace torch_nntile

#endif
