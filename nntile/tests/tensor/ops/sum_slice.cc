/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/sum_slice.cc
 * Test TensorGraph sum_slice operation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/sum_slice.hh"

#include "context_fixture.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tile.hh"
#include "nntile/tensor/ops/sum_slice.hh"
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
constexpr Index axis_2 = 2;
constexpr int redux_none = 0;
constexpr Scalar alpha_one = 1.0;
constexpr Scalar alpha_two = 2.0;
constexpr Scalar beta_zero = 0.0;
constexpr Scalar beta_half = 0.5;
constexpr float y_init_overwrite = 0.0f;
constexpr float y_init_accumulate = 1.0f;
constexpr float tolerance = 1e-5f;
constexpr int distr_rank_single = 0;
constexpr Index x_fill_offset = 1;

constexpr Index dim_2 = 2;
constexpr Index dim_3 = 3;
constexpr Index dim_4 = 4;
constexpr Index dim_5 = 5;
constexpr Index dim_6 = 6;

} // anonymous namespace

//! Output shape for sum_slice: src shape with axis removed
static std::vector<Index> sum_slice_output_shape(
    const std::vector<Index> &src_shape, Index axis)
{
    std::vector<Index> out;
    out.reserve(src_shape.size() - 1);
    for (Index i = 0; i < static_cast<Index>(src_shape.size()); ++i)
    {
        if (i != axis)
        {
            out.push_back(src_shape[i]);
        }
    }
    return out;
}

TEST_CASE("TensorGraph sum_slice structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    nntile::TensorRef src = graph.data({dim_4, dim_5});
    src->set_name("src");
    nntile::TensorRef dst = graph.data({dim_4});
    dst->set_name("dst"); // axis=1: sum over dim_5

    gt::sum_slice(src, dst, axis_1, redux_none, alpha_one, beta_zero);

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(dst->shape().size() == 1);
    REQUIRE(dst->shape()[0] == dim_4);

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "SUM_SLICE");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE("TensorGraph sum_slice rejects duplicate tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    nntile::TensorRef src = graph.data({dim_4, dim_5});
    src->set_name("src");

    REQUIRE_THROWS_AS(
        gt::sum_slice(src, src, axis_1, redux_none, alpha_one, beta_zero),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph sum_slice tiled matches untiled",
    "[graph][tensor]")
{
    const auto [src_shape, axis, redux, alpha, beta] =
        GENERATE(std::tuple{std::vector<Index>{dim_4, dim_5},
                     axis_0,
                     redux_none,
                     alpha_one,
                     beta_zero},
            std::tuple{std::vector<Index>{dim_2, dim_3, dim_4},
                axis_1,
                redux_none,
                alpha_one,
                beta_zero});

    using T = nntile::fp32_t;
    using Y = typename T::repr_t;
    const Index src_nelems = std::accumulate(
        src_shape.begin(), src_shape.end(), Index(1), std::multiplies<>());

    std::vector<Index> dst_shape = sum_slice_output_shape(src_shape, axis);
    const Index dst_nelems = std::accumulate(
        dst_shape.begin(), dst_shape.end(), Index(1), std::multiplies<>());

    std::vector<float> src_data(src_nelems);
    for (Index i = 0; i < src_nelems; ++i)
    {
        src_data[i] = static_cast<float>(Y(i + x_fill_offset));
    }
    std::vector<float> dst_data(dst_nelems);
    for (Index i = 0; i < dst_nelems; ++i)
    {
        dst_data[i] =
            (beta != beta_zero) ? y_init_accumulate : y_init_overwrite;
    }

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("sum_slice_untiled");
        nntile::TensorRef src_node = graph.data(src_shape, DataType::FP32);
    src_node->set_name("src");
        nntile::TensorRef dst_node = graph.data(dst_shape, DataType::FP32);
    dst_node->set_name("dst");

        gt::sum_slice(src_node, dst_node, axis, redux, alpha, beta);

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
        TensorGraph graph("sum_slice_tiled");
        nntile::TensorRef src_node = graph.data(src_shape, DataType::FP32);
    src_node->set_name("src");
        nntile::TensorRef dst_node = graph.data(dst_shape, DataType::FP32);
    dst_node->set_name("dst");

        gt::sum_slice(src_node, dst_node, axis, redux, alpha, beta);
        for (auto *ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
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
    constexpr float tol = 1e-5f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for (size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}
