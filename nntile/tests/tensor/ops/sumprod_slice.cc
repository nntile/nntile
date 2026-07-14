/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/sumprod_slice.cc
 * Test TensorGraph sumprod_slice operation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/sumprod_slice.hh"

#include "context_fixture.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tile.hh"
#include "nntile/tensor/ops/sumprod_slice.hh"
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
constexpr Scalar alpha_half = 0.5;
constexpr Scalar beta_zero = 0.0;
constexpr Scalar beta_one = 1.0;
constexpr float tolerance = 1e-5f;
constexpr int distr_rank_single = 0;

constexpr Index dim_2 = 2;
constexpr Index dim_3 = 3;
constexpr Index dim_4 = 4;
constexpr Index dim_5 = 5;

} // anonymous namespace

//! Dst shape for sumprod_slice: src shape with axis dimension removed
static std::vector<Index> sumprod_slice_dst_shape(
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

TEST_CASE("TensorGraph sumprod_slice structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    nntile::TensorRef src1 = graph.data({dim_2, dim_4});
    src1->set_name("src1");
    nntile::TensorRef src2 = graph.data({dim_2, dim_4});
    src2->set_name("src2");
    nntile::TensorRef dst = graph.data({dim_4});
    dst->set_name("dst"); // axis=0: sum over dim_2

    gt::sumprod_slice(
        src1, src2, dst, axis_0, redux_none, alpha_one, beta_zero);

    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(dst->shape() == (std::vector<Index>{dim_4}));

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "SUMPROD_SLICE");
    REQUIRE(ops[0]->inputs().size() == 3);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE(
    "TensorGraph sumprod_slice rejects duplicate tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    nntile::TensorRef src1 = graph.data({dim_2, dim_4});
    src1->set_name("src1");
    nntile::TensorRef src2 = graph.data({dim_2, dim_4});
    src2->set_name("src2");
    nntile::TensorRef dst = graph.data({dim_4});
    dst->set_name("dst");

    REQUIRE_THROWS_AS(
        gt::sumprod_slice(
            src1, src1, dst, axis_0, redux_none, alpha_one, beta_zero),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::sumprod_slice(
            src1, src2, src1, axis_0, redux_none, alpha_one, beta_zero),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph sumprod_slice tiled matches untiled",
    "[graph][tensor]")
{
    const auto [src_shape, axis, redux, alpha, beta] =
        GENERATE(std::tuple{std::vector<Index>{dim_2, dim_4},
                     axis_0,
                     redux_none,
                     alpha_one,
                     beta_zero},
            std::tuple{std::vector<Index>{dim_2, dim_3, dim_4},
                axis_2,
                redux_none,
                alpha_one,
                beta_one});

    using Y = nntile::fp32_t::repr_t;
    const Index src_nelems = std::accumulate(
        src_shape.begin(), src_shape.end(), Index(1), std::multiplies<>());
    std::vector<Index> dst_shape = sumprod_slice_dst_shape(src_shape, axis);
    const Index dst_nelems = std::accumulate(
        dst_shape.begin(), dst_shape.end(), Index(1), std::multiplies<>());

    std::vector<float> src1_data(src_nelems);
    std::vector<float> src2_data(src_nelems);
    std::vector<float> dst_data(dst_nelems);
    for (Index i = 0; i < src_nelems; ++i)
    {
        src1_data[i] = static_cast<float>(Y((i + 1) * (i + 2)));
        src2_data[i] = static_cast<float>(Y(1.0 / (i + 1)));
    }
    for (Index i = 0; i < dst_nelems; ++i)
    {
        dst_data[i] = (beta != beta_zero) ? 1.0f : 0.0f;
    }

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("sumprod_slice_untiled");
        nntile::TensorRef src1_node = graph.data(src_shape, DataType::FP32);
    src1_node->set_name("src1");
        nntile::TensorRef src2_node = graph.data(src_shape, DataType::FP32);
    src2_node->set_name("src2");
        nntile::TensorRef dst_node = graph.data(dst_shape, DataType::FP32);
    dst_node->set_name("dst");

        gt::sumprod_slice(
            src1_node, src2_node, dst_node, axis, redux, alpha, beta);

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data(src1_node, src1_data);
        runtime.bind_data(src2_node, src2_data);
        runtime.bind_data(dst_node, dst_data);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>(dst_node);
    }

    // --- Tiled run ---
    std::vector<float> tiled_result;
    {
        TensorGraph graph("sumprod_slice_tiled");
        nntile::TensorRef src1_node = graph.data(src_shape, DataType::FP32);
    src1_node->set_name("src1");
        nntile::TensorRef src2_node = graph.data(src_shape, DataType::FP32);
    src2_node->set_name("src2");
        nntile::TensorRef dst_node = graph.data(dst_shape, DataType::FP32);
    dst_node->set_name("dst");

        gt::sumprod_slice(
            src1_node, src2_node, dst_node, axis, redux, alpha, beta);
        for (auto *ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data(src1_node, src1_data);
        runtime.bind_data(src2_node, src2_data);
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
