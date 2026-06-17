/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/logsumexp.cc
 * Test TensorGraph logsumexp operation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/logsumexp.hh"

#include "context_fixture.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tensor/ops/maxsumexp.hh"
#include "nntile/tile.hh"
#include "nntile/tensor/ops/clear.hh"
#include "nntile/tensor/ops/logsumexp.hh"
#include "nntile/tensor/ops/maxsumexp.hh"
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
constexpr float tolerance = 1e-4f;
constexpr int distr_rank_single = 0;

} // anonymous namespace

// maxsumexp output shape: src.shape without axis + [2]
static std::vector<Index> maxsumexp_dst_shape(
    const std::vector<Index> &src_shape, Index axis)
{
    std::vector<Index> dst;
    for (Index i = 0; i < static_cast<Index>(src_shape.size()); ++i)
    {
        if (i != axis)
        {
            dst.push_back(src_shape[i]);
        }
    }
    dst.push_back(2);
    return dst;
}

TEST_CASE("TensorGraph logsumexp structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto *src =
        graph.data({4, 5, 2})->set_name("src"); // maxsumexp output shape
    auto *dst = gt::logsumexp(src)->set_name("dst");

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(dst->shape().size() == 2);
    REQUIRE(dst->shape()[0] == 4);
    REQUIRE(dst->shape()[1] == 5);

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "LOGSUMEXP");
    REQUIRE(ops[0]->inputs().size() == 1);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE("TensorGraph logsumexp rejects null", "[graph][tensor]")
{
    TensorGraph graph("test");

    REQUIRE_THROWS_AS(gt::logsumexp(nullptr), std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph logsumexp tiled matches untiled",
    "[graph][tensor]")
{
    const auto [shape, axis] =
        GENERATE(std::tuple{std::vector<Index>{4, 6}, Index(0)},
            std::tuple{std::vector<Index>{3, 4}, Index(0)});

    const Index src_nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    std::vector<float> src_data(src_nelems);
    for (Index i = 0; i < src_nelems; ++i)
    {
        src_data[i] = static_cast<float>(i % 10 - 2);
    }

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("logsumexp_untiled");
        auto *src_node = graph.data(shape, DataType::FP32)->set_name("src");
        src_node->mark_input(true);

        auto *maxsumexp_node =
            gt::maxsumexp(src_node, axis, 0)->set_name("maxsumexp");
        auto *logsumexp_node =
            gt::logsumexp(maxsumexp_node)->set_name("logsumexp");
        logsumexp_node->mark_output(true);

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data(src_node, src_data);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>(logsumexp_node);
    }

    // --- Tiled run ---
    std::vector<float> tiled_result;
    {
        TensorGraph graph("logsumexp_tiled");
        auto *src_node = graph.data(shape, DataType::FP32)->set_name("src");
        src_node->mark_input(true);

        auto *maxsumexp_node =
            gt::maxsumexp(src_node, axis, 0)->set_name("maxsumexp");
        auto *logsumexp_node =
            gt::logsumexp(maxsumexp_node)->set_name("logsumexp");
        logsumexp_node->mark_output(true);
        auto *maxsumexp_dim0 = maxsumexp_node->axis(0);
        auto *maxsumexp_pair = maxsumexp_node->axis(maxsumexp_node->ndim() - 1);
        for (auto *ag : graph.axis_groups())
        {
            if (ag == maxsumexp_pair)
            {
                ag->set_tiling(ag->extent);
            }
            else if (ag == maxsumexp_dim0)
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
        runtime.execute();
        runtime.wait();

        tiled_result = runtime.get_output<float>(logsumexp_node);
    }

    // --- Compare ---
    constexpr float tol = 1e-4f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for (size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}
