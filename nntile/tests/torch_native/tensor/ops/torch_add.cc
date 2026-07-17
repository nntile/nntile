/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file nntile/tests/torch_native/tensor/ops/torch_add.cc
 * TensorGraph torch_add structure + untiled execute vs aten.
 */

#include "aten_ref.hh"
#include "context_fixture.hh"

#include <nntile/tensor.hh>
#include <nntile/tensor/ops/torch_add.hh>
#include <nntile/tile.hh>

#include <ATen/ops/add.h>

#include <catch2/catch_test_macros.hpp>

#include <numeric>
#include <vector>

using namespace nntile;
namespace gt = nntile::tensor;
namespace tn = nntile::test::torch_native;

TEST_CASE("TensorGraph torch_add structure", "[torch_native][tensor]")
{
    TensorGraph graph("test");
    nntile::TensorRef x = graph.data({4, 5}, DataType::FP32);
    nntile::TensorRef y = graph.data({4, 5}, DataType::FP32);
    nntile::TensorRef z = nntile::TensorRef::adopt(
        gt::torch_add(x, y, /*alpha=*/1.5));

    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(graph.ops()[0]->op_name() == "TORCH_ADD");
    REQUIRE(graph.ops()[0]->outputs()[0] == z);
}

TEST_CASE_METHOD(
    nntile::test::ContextFixture,
    "TensorGraph torch_add untiled matches aten",
    "[torch_native][tensor]")
{
    const std::vector<Index> shape = {4, 6};
    const Index nelems = std::accumulate(
        shape.begin(),
        shape.end(),
        Index{1},
        std::multiplies<>());
    const Scalar alpha = 2.0f;

    std::vector<float> x_data(nelems), y_data(nelems), ref(nelems, 0.f);
    for (Index i = 0; i < nelems; ++i)
    {
        x_data[static_cast<size_t>(i)] = static_cast<float>(i + 1);
        y_data[static_cast<size_t>(i)] = static_cast<float>(i - 3);
    }

    tn::with_cpu_aten(
        [&]
        {
            at::Tensor tx = tn::blob_cpu(x_data.data(), shape);
            at::Tensor ty = tn::blob_cpu(y_data.data(), shape);
            at::Tensor tr = tn::blob_cpu(ref.data(), shape);
            at::add_out(tr, tx, ty, alpha);
        });

    TensorGraph graph("torch_add_untiled");
    nntile::TensorRef x = graph.data(shape, DataType::FP32);
    nntile::TensorRef y = graph.data(shape, DataType::FP32);
    nntile::TensorRef z = nntile::TensorRef::adopt(
        gt::torch_add(x, y, alpha));

    TileGraph tile_graph = TileGraph::from_tensor_graph(graph);
    Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data(x, x_data);
    runtime.bind_data(y, y_data);
    runtime.execute();
    runtime.wait();

    tn::require_close(runtime.get_output<float>(z), ref);
}
