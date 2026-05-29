/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/softmax_inplace.cc
 * Test TensorGraph softmax_inplace operation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/softmax_inplace.hh"

#include "context_fixture.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tensor/ops/maxsumexp.hh"
#include "nntile/tile.hh"
#include "nntile/tensor/ops/clear.hh"
#include "nntile/tensor/ops/maxsumexp.hh"
#include "nntile/tensor/ops/softmax_inplace.hh"
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
constexpr int redux = 0;
constexpr Scalar alpha_one = 1.0;
constexpr float tolerance = 1e-4f;
constexpr int distr_rank_single = 0;

} // anonymous namespace

static std::vector<Index> maxsumexp_dst_shape(
    const std::vector<Index> &src_shape, Index axis)
{
    std::vector<Index> dst = {2};
    for (Index i = 0; i < static_cast<Index>(src_shape.size()); ++i)
    {
        if (i != axis)
        {
            dst.push_back(src_shape[i]);
        }
    }
    return dst;
}

TEST_CASE("TensorGraph softmax_inplace structure", "[graph][tensor]")
{
    constexpr Index dim0 = 4;
    constexpr Index dim1 = 5;

    TensorGraph graph("test");

    auto *maxsumexp_node = graph.data({2, dim0, dim1})->set_name("maxsumexp");
    auto *dst = graph.data({dim0, dim1})->set_name("dst");

    gt::softmax_inplace(maxsumexp_node, dst, alpha_one, axis_0);

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "SOFTMAX_INPLACE");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE("TensorGraph softmax_inplace rejects null", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto *mse = graph.data({2, 4, 5})->set_name("mse");
    auto *dst = graph.data({4, 5})->set_name("dst");

    REQUIRE_THROWS_AS(gt::softmax_inplace(nullptr, dst, alpha_one, axis_0),
        std::invalid_argument);
    REQUIRE_THROWS_AS(gt::softmax_inplace(mse, nullptr, alpha_one, axis_0),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph softmax_inplace tiled matches untiled",
    "[graph][tensor]")
{
    const auto [shape, axis, alpha] =
        GENERATE(std::tuple{std::vector<Index>{4, 6}, Index(0), 1.0},
            std::tuple{std::vector<Index>{3, 4}, Index(0), 0.5});

    using Y = nntile::fp32_t::repr_t;
    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    std::vector<float> src_data(nelems);
    std::vector<float> dst_data(nelems);
    for (Index i = 0; i < nelems; ++i)
    {
        src_data[i] = static_cast<float>(Y(i % 10 - 2));
        dst_data[i] = src_data[i];
    }

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("softmax_inplace_untiled");
        auto *src_node = graph.data(shape, DataType::FP32)->set_name("src");
        auto *dst_node = graph.data(shape, DataType::FP32)->set_name("dst");
        src_node->mark_input(true);
        dst_node->mark_input(true);
        dst_node->mark_output(true);

        auto *maxsumexp_node =
            gt::maxsumexp(src_node, axis, redux)->set_name("maxsumexp");
        gt::softmax_inplace(maxsumexp_node, dst_node, alpha, axis);

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data(src_node, src_data);
        runtime.bind_data(dst_node, dst_data);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>(dst_node);
    }

    // --- Tiled run ---
    std::vector<float> tiled_result;
    {
        TensorGraph graph("softmax_inplace_tiled");
        auto *src_node = graph.data(shape, DataType::FP32)->set_name("src");
        auto *dst_node = graph.data(shape, DataType::FP32)->set_name("dst");
        src_node->mark_input(true);
        dst_node->mark_input(true);
        dst_node->mark_output(true);

        auto *maxsumexp_node =
            gt::maxsumexp(src_node, axis, redux)->set_name("maxsumexp");
        gt::softmax_inplace(maxsumexp_node, dst_node, alpha, axis);
        auto *maxsumexp_dim0 = maxsumexp_node->axis(0);
        for (auto *ag : graph.axis_groups())
        {
            if (ag == maxsumexp_dim0)
            {
                ag->set_tiling(ag->extent);
            }
            else
            {
                ag->set_tiling((ag->extent + 1) / 2);
            }
        }

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data(src_node, src_data);
        runtime.bind_data(dst_node, dst_data);
        runtime.execute();
        runtime.wait();

        tiled_result = runtime.get_output<float>(dst_node);
    }

    // --- Compare ---
    constexpr float tol = 1e-4f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for (size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}
