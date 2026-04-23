/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/scale_slice.cc
 * Test TensorGraph scale_slice operation against nntile::tensor::scale_slice.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/scale_slice.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/scale_slice.hh"
#include "nntile/tensor/tensor.hh"
#include "nntile/graph/tensor/axis_descriptor.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

namespace
{

constexpr Index axis_0 = 0;
constexpr Index axis_1 = 1;
constexpr Index axis_2 = 2;
constexpr Scalar alpha_one = 1.0;
constexpr Scalar alpha_half = 0.5;
constexpr Scalar alpha_two = 2.0;
constexpr float tolerance = 1e-5f;
constexpr int distr_rank_single = 0;

constexpr Index dim_2 = 2;
constexpr Index dim_3 = 3;
constexpr Index dim_4 = 4;
constexpr Index dim_5 = 5;

} // anonymous namespace

//! Slice shape: dst shape with axis removed
static std::vector<Index> slice_shape(
    const std::vector<Index>& dst_shape,
    Index axis)
{
    std::vector<Index> out;
    out.reserve(dst_shape.size() - 1);
    for(Index i = 0; i < static_cast<Index>(dst_shape.size()); ++i)
    {
        if(i != axis)
        {
            out.push_back(dst_shape[i]);
        }
    }
    return out;
}

template<typename T>
void check_scale_slice_vs_tensor_api(
    const std::vector<Index>& dst_shape,
    Index axis,
    Scalar alpha)
{
    using Y = typename T::repr_t;
    const Index dst_nelems = std::accumulate(
        dst_shape.begin(), dst_shape.end(), Index(1), std::multiplies<>());

    std::vector<Index> src_sh = slice_shape(dst_shape, axis);
    const Index src_nelems = std::accumulate(
        src_sh.begin(), src_sh.end(), Index(1), std::multiplies<>());
    const Index axis_size = dst_shape[axis];

    // --- TensorGraph path ---
    TensorGraph graph("scale_slice_test");
    auto* src_node = graph.data(src_sh, "src", DataType::FP32);
    src_node->mark_input(true);

    auto* out_node = gt::scale_slice(alpha, src_node, "out", axis, axis_size);
    out_node->mark_output(true);

    TileGraph runtime_tile = TileGraph::from_tensor_graph(graph);


    TileGraph::Runtime runtime(runtime_tile);
    runtime.compile();

    std::vector<float> src_data(src_nelems);
    for(Index i = 0; i < src_nelems; ++i)
    {
        src_data[i] = static_cast<float>(Y(i + 1));
    }

    runtime.bind_data("src", src_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("out");

    // --- Direct tensor API path ---
    nntile::tensor::TensorTraits src_traits(src_sh, src_sh);
    nntile::tensor::TensorTraits dst_traits(dst_shape, dst_shape);
    std::vector<int> src_distr(src_traits.grid.nelems, distr_rank_single);
    std::vector<int> dst_distr(dst_traits.grid.nelems, distr_rank_single);
    nntile::tensor::Tensor<T> src_t(src_traits, src_distr);
    nntile::tensor::Tensor<T> out_t(dst_traits, dst_distr);

    {
        auto tile = src_t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < src_nelems; ++i)
        {
            loc[i] = static_cast<Y>(src_data[i]);
        }
        loc.release();
    }

    nntile::tensor::scale_slice<T>(alpha, src_t, out_t, axis);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(dst_nelems);
    {
        auto tile = out_t.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        for(Index i = 0; i < dst_nelems; ++i)
        {
            tensor_result[i] = static_cast<float>(loc[i]);
        }
        loc.release();
    }

    REQUIRE(graph_result.size() == tensor_result.size());
    for(size_t i = 0; i < graph_result.size(); ++i)
    {
        REQUIRE(std::abs(graph_result[i] - tensor_result[i]) < tolerance);
    }
}

TEST_CASE("TensorGraph scale_slice structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto* src = graph.data({dim_4}, "src");  // slice for dst shape [dim_2, dim_4]
    auto* out = gt::scale_slice(alpha_one, src, "out", axis_0, dim_2);

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(out->shape() == (std::vector<Index>{dim_2, dim_4}));

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "SCALE_SLICE");
    REQUIRE(ops[0]->inputs().size() == 1);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == out);
}

TEST_CASE("TensorGraph scale_slice rejects duplicate tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* src = graph.data({dim_4}, "src");

    REQUIRE_THROWS_AS(
        gt::scale_slice(alpha_one, src, src, axis_0),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph scale_slice matches nntile::tensor::scale_slice", "[graph][tensor]")
{
    const auto [dst_shape, axis, alpha] = GENERATE(
        std::tuple{std::vector<Index>{dim_2, dim_4}, axis_1, alpha_one},
        std::tuple{std::vector<Index>{dim_2, dim_4}, axis_0, alpha_one},
        std::tuple{std::vector<Index>{dim_2, dim_3, dim_4}, axis_0, alpha_one},
        std::tuple{std::vector<Index>{dim_2, dim_3, dim_4}, axis_1, alpha_half},
        std::tuple{std::vector<Index>{dim_2, dim_3, dim_4}, axis_2, alpha_two});

    check_scale_slice_vs_tensor_api<nntile::fp32_t>(dst_shape, axis, alpha);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph scale_slice tiled matches untiled", "[graph][tensor]")
{
    const auto [dst_shape, axis, alpha] = GENERATE(
        std::tuple{std::vector<Index>{2, 4}, Index(1), 1.0},
        std::tuple{std::vector<Index>{2, 4}, Index(0), 1.0});

    using T = nntile::fp32_t;
    using Y = T::repr_t;
    std::vector<Index> src_sh = slice_shape(dst_shape, axis);
    const Index src_nelems = std::accumulate(
        src_sh.begin(), src_sh.end(), Index(1), std::multiplies<>());
    const Index axis_size = dst_shape[axis];

    std::vector<float> src_data(src_nelems);
    for(Index i = 0; i < src_nelems; ++i)
        src_data[i] = static_cast<float>(Y(i + 1));

    std::vector<float> untiled_result;
    {
        TensorGraph graph("scale_slice_untiled");
        auto* src_node = graph.data(src_sh, "src", DataType::FP32);
        src_node->mark_input(true);
        auto* out_node = gt::scale_slice(alpha, src_node, "out", axis, axis_size);
        out_node->mark_output(true);
        TileGraph runtime_tile = TileGraph::from_tensor_graph(graph);

        TileGraph::Runtime runtime(runtime_tile);
        runtime.compile();
        runtime.bind_data("src", src_data);
        runtime.execute();
        runtime.wait();
        untiled_result = runtime.get_output<float>("out");
    }

    std::vector<float> tiled_result;
    {
        TensorGraph graph("scale_slice_tiled");
        auto* src_node = graph.data(src_sh, "src", DataType::FP32);
        src_node->mark_input(true);
        auto* out_node = gt::scale_slice(alpha, src_node, "out", axis, axis_size);
        out_node->mark_output(true);
        for(auto* ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }
        TileGraph runtime_tile = TileGraph::from_tensor_graph(graph);

        TileGraph::Runtime runtime(runtime_tile);
        runtime.compile();
        runtime.bind_data("src", src_data);
        runtime.execute();
        runtime.wait();
        tiled_result = runtime.get_output<float>("out");
    }

    constexpr float tol = 1e-5f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for(size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}
