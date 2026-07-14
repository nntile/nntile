/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/sqrt.cc
 * Test TensorGraph sqrt operation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/sqrt.hh"

#include "context_fixture.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tile.hh"
#include "nntile/tensor/ops/sqrt.hh"
#include "nntile/tensor.hh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <numeric>

using namespace nntile;
using namespace nntile;
namespace gt = nntile::tensor;

TEST_CASE("TensorGraph sqrt structure", "[graph][tensor]")
{
    constexpr Index dim0 = 4;
    constexpr Index dim1 = 5;

    TensorGraph graph("test");

    nntile::TensorRef src = graph.data({dim0, dim1});
    src->set_name("src");

    nntile::TensorRef dst = nntile::TensorRef::adopt(gt::sqrt(src));
    dst->set_name("dst");

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(dst->shape()[0] == dim0);
    REQUIRE(dst->shape()[1] == dim1);

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "SQRT");
    REQUIRE(ops[0]->inputs().size() == 1);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE("TensorGraph sqrt rejects duplicate tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    nntile::TensorRef src = graph.data({5, 4});
    src->set_name("src");

    REQUIRE_THROWS_AS(gt::sqrt(src, src), std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph sqrt tiled matches untiled",
    "[graph][tensor]")
{
    const auto shape = GENERATE(std::vector<Index>{4, 6},
        std::vector<Index>{6},
        std::vector<Index>{2, 4});

    using T = nntile::fp32_t;
    using Y = T::repr_t;
    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    std::vector<float> src_data(nelems);
    for (Index i = 0; i < nelems; ++i)
    {
        src_data[i] = static_cast<float>(Y(i + 1));
    }

    std::vector<float> untiled_result;
    {
        TensorGraph graph("sqrt_untiled");
        nntile::TensorRef src_node = graph.data(shape, DataType::FP32);
    src_node->set_name("src");
        nntile::TensorRef dst_node = nntile::TensorRef::adopt(gt::sqrt(src_node));
    dst_node->set_name("dst");
        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data(src_node, src_data);
        runtime.execute();
        runtime.wait();
        untiled_result = runtime.get_output<float>(dst_node);
    }

    std::vector<float> tiled_result;
    {
        TensorGraph graph("sqrt_tiled");
        nntile::TensorRef src_node = graph.data(shape, DataType::FP32);
    src_node->set_name("src");
        nntile::TensorRef dst_node = nntile::TensorRef::adopt(gt::sqrt(src_node));
    dst_node->set_name("dst");
        for (auto *ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }
        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data(src_node, src_data);
        runtime.execute();
        runtime.wait();
        tiled_result = runtime.get_output<float>(dst_node);
    }

    constexpr float tol = 1e-5f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for (size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}
