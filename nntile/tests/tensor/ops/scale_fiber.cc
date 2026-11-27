/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/scale_fiber.cc
 * Test TensorGraph scale_fiber operation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/scale_fiber.hh"

#include "context_fixture.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tile.hh"
#include "nntile/tensor/ops/scale_fiber.hh"
#include "nntile/tensor.hh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <numeric>

using namespace nntile;
using namespace nntile;
namespace gt = nntile::tensor;

namespace
{

constexpr Index axis_0 = 0;
constexpr Index axis_1 = 1;
constexpr Index batch_ndim_none = 0;
constexpr Scalar alpha = 2.5;
constexpr Scalar alpha_one = 1.0;
constexpr float tolerance = 1e-5f;
constexpr int distr_rank_single = 0;

constexpr Index dim_2 = 2;
constexpr Index dim_4 = 4;
constexpr Index dim_5 = 5;

} // anonymous namespace

//! Fiber shape: {dst_shape[axis]} for batch_ndim=0
static std::vector<Index> fiber_shape(
    const std::vector<Index> &dst_shape, Index axis, Index batch_ndim)
{
    std::vector<Index> out;
    out.reserve(batch_ndim + 1);
    for (Index i = 0; i < batch_ndim; ++i)
    {
        out.push_back(dst_shape[i]);
    }
    out.push_back(dst_shape[axis]);
    return out;
}

TEST_CASE("TensorGraph scale_fiber structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    nntile::TensorRef src = graph.data({dim_4});
    src->set_name("src");

    nntile::TensorRef dst = nntile::TensorRef::adopt(gt::scale_fiber(alpha, src, {dim_2, dim_4}, axis_1, batch_ndim_none)
            );
    dst->set_name("dst");

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(dst->shape() == (std::vector<Index>{dim_2, dim_4}));

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "SCALE_FIBER");
    REQUIRE(ops[0]->inputs().size() == 1);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE(
    "TensorGraph scale_fiber rejects duplicate tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    nntile::TensorRef src = graph.data({dim_4});
    src->set_name("src");

    REQUIRE_THROWS_AS(
        gt::scale_fiber(alpha, src, src, axis_1, batch_ndim_none),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph scale_fiber tiled matches untiled",
    "[graph][tensor]")
{
    const auto [dst_shape, axis, batch_ndim, alpha_val] =
        GENERATE(std::tuple{std::vector<Index>{2, 4}, Index(1), Index(0), 2.5},
            std::tuple{std::vector<Index>{2, 4}, Index(0), Index(0), 1.0});

    using T = nntile::fp32_t;
    using Y = T::repr_t;
    std::vector<Index> fiber_sh = fiber_shape(dst_shape, axis, batch_ndim);
    const Index fiber_nelems = std::accumulate(
        fiber_sh.begin(), fiber_sh.end(), Index(1), std::multiplies<>());

    std::vector<float> src_data(fiber_nelems);
    for (Index i = 0; i < fiber_nelems; ++i)
        src_data[i] = static_cast<float>(Y(i + 1));

    std::vector<float> untiled_result;
    {
        TensorGraph graph("scale_fiber_untiled");
        nntile::TensorRef src_node = graph.data(fiber_sh, DataType::FP32);
    src_node->set_name("src");
        nntile::TensorRef dst_node = nntile::TensorRef::adopt(gt::scale_fiber(alpha_val, src_node, dst_shape, axis, batch_ndim)
                );
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
        TensorGraph graph("scale_fiber_tiled");
        nntile::TensorRef src_node = graph.data(fiber_sh, DataType::FP32);
    src_node->set_name("src");
        nntile::TensorRef dst_node = nntile::TensorRef::adopt(gt::scale_fiber(alpha_val, src_node, dst_shape, axis, batch_ndim)
                );
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
