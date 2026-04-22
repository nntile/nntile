/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/tile_graph.cc
 * Tests for TileGraph class: construction, from_tensor_graph, and execution.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tile.hh"
#include "nntile/graph/tensor.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;
namespace tg = nntile::graph::tile_graph;

TEST_CASE("TileGraph basic construction", "[graph][tile]")
{
    TileGraph graph("basic_test");

    REQUIRE(graph.name() == "basic_test");
    REQUIRE(graph.num_data() == 0);
    REQUIRE(graph.num_ops() == 0);
}

TEST_CASE("TileGraph data creation", "[graph][tile]")
{
    TileGraph graph("data_test");

    auto* x = graph.data({2, 3}, "x", DataType::FP32);
    REQUIRE(x != nullptr);
    REQUIRE(x->name() == "x");
    REQUIRE(x->shape() == std::vector<Index>{2, 3});
    REQUIRE(x->dtype() == DataType::FP32);
    REQUIRE(x->nelems() == 6);
    REQUIRE(graph.num_data() == 1);

    // Standalone tiles have no tensor descriptor
    REQUIRE(x->tensor_descriptor() == nullptr);
    REQUIRE(x->tile_coord().empty());

    auto* y = graph.data({4}, "y", DataType::FP64);
    REQUIRE(y->nelems() == 4);
    REQUIRE(graph.num_data() == 2);
}

TEST_CASE("TileGraph rejects duplicate data names", "[graph][tile]")
{
    TileGraph graph("dup_test");
    graph.data({2, 3}, "x");
    REQUIRE_THROWS_AS(graph.data({2, 3}, "x"), std::invalid_argument);
}

TEST_CASE("TileGraph lookup", "[graph][tile]")
{
    TileGraph graph("lookup_test");
    auto* x = graph.data({2, 3}, "x");

    REQUIRE(graph.get_tile_node("x") == x);
    REQUIRE(graph.get_tile_node("missing") == nullptr);
}

TEST_CASE("TileGraph to_string", "[graph][tile]")
{
    TileGraph graph("str_test");
    graph.data({2}, "x");
    graph.data({2}, "y");
    tg::add(1.0, graph.get_tile_node("x"), 1.0, graph.get_tile_node("y"), "z");

    std::string s = graph.to_string();
    REQUIRE(s.find("TileGraph") != std::string::npos);
    REQUIRE(s.find("TILE_ADD") != std::string::npos);
    REQUIRE(s.find("Tiles:") != std::string::npos);
}

TEST_CASE("TileGraph add_tensor_descriptor manual", "[graph][tile]")
{
    TileGraph graph("manual_desc_test");
    auto* t0 = graph.data({4}, "t0");

    TileGraph::TensorDescriptor desc;
    desc.tensor_name = "T";
    desc.tensor_shape = {4};
    desc.tile_shape = {4};
    desc.grid_shape = {1};
    desc.dtype = DataType::FP32;
    desc.tiles = {t0};

    auto* dp = graph.add_tensor_descriptor(std::move(desc));
    t0->set_tensor_info(dp, {0});

    REQUIRE(graph.num_tensors() == 1);
    REQUIRE(graph.get_tensor_descriptor("T") == dp);
    REQUIRE(graph.get_tensor_descriptor("missing") == nullptr);
    REQUIRE(t0->tensor_descriptor() == dp);
    REQUIRE(t0->tile_coord() == std::vector<Index>{0});
    REQUIRE(dp->tiles[0] == t0);
}

TEST_CASE("TileGraph to_mermaid", "[graph][tile]")
{
    TileGraph graph("mermaid_test");
    graph.data({2}, "x");
    graph.data({2}, "y");
    tg::add(1.0, graph.get_tile_node("x"), 1.0, graph.get_tile_node("y"), "z");

    std::string m = graph.to_mermaid();
    REQUIRE(m.find("graph TD") != std::string::npos);
    REQUIRE(m.find("local[2]") != std::string::npos);
}

TEST_CASE("TileGraph from_tensor_graph structure", "[graph][tile]")
{
    TensorGraph tg_graph("tensor_test");
    auto* x = tg_graph.data({2, 3}, "x", DataType::FP32);
    auto* y = tg_graph.data({2, 3}, "y", DataType::FP32);
    x->mark_input(true);
    y->mark_input(true);

    auto* z = gt::add(1.0, x, 1.0, y, "z");
    z->mark_output(true);

    TileGraph tile_graph = TileGraph::from_tensor_graph(tg_graph);

    REQUIRE(tile_graph.num_data() == 3);
    REQUIRE(tile_graph.num_ops() == 1);
    REQUIRE(tile_graph.num_tensors() == 3);

    auto* tx = tile_graph.get_tile_node("x");
    auto* ty = tile_graph.get_tile_node("y");
    auto* tz = tile_graph.get_tile_node("z");
    REQUIRE(tx != nullptr);
    REQUIRE(ty != nullptr);
    REQUIRE(tz != nullptr);

    REQUIRE(tx->is_input());
    REQUIRE(ty->is_input());
    REQUIRE(tz->is_output());

    REQUIRE(tx->shape() == std::vector<Index>{2, 3});
    REQUIRE(ty->shape() == std::vector<Index>{2, 3});
    REQUIRE(tz->shape() == std::vector<Index>{2, 3});

    REQUIRE(tile_graph.ops()[0]->op_name() == "TILE_ADD");

    // Verify tensor descriptors
    auto* xd = tile_graph.get_tensor_descriptor("x");
    REQUIRE(xd != nullptr);
    REQUIRE(xd->tensor_name == "x");
    REQUIRE(xd->tensor_shape == std::vector<Index>{2, 3});
    REQUIRE(xd->tile_shape == std::vector<Index>{2, 3});
    REQUIRE(xd->grid_shape == std::vector<Index>{1, 1});
    REQUIRE(xd->dtype == DataType::FP32);
    REQUIRE(xd->tiles.size() == 1);
    REQUIRE(xd->tiles[0] == tx);

    // Verify tile coordinates
    REQUIRE(tx->tensor_descriptor() == xd);
    REQUIRE(tx->tile_coord() == std::vector<Index>{0, 0});
    REQUIRE(ty->tile_coord() == std::vector<Index>{0, 0});
    REQUIRE(tz->tile_coord() == std::vector<Index>{0, 0});
}

TEST_CASE("TileGraph from_tensor_graph links source TensorNode",
          "[graph][tile]")
{
    TensorGraph tg_graph("hint_test");
    auto* x = tg_graph.data({2}, "x", DataType::FP32);
    x->mark_input(true);

    std::vector<std::uint8_t> hint(2 * sizeof(float), 0x42);
    x->set_bind_hint(hint);

    TileGraph tile_graph = TileGraph::from_tensor_graph(tg_graph);
    auto* tx = tile_graph.get_tile_node("x");
    REQUIRE(tx != nullptr);

    // Bind hints are not copied to TileNode; they stay on the TensorNode
    REQUIRE(tx->get_bind_hint() == nullptr);

    // Instead the TensorDescriptor references the source TensorNode
    auto* td = tile_graph.get_tensor_descriptor("x");
    REQUIRE(td != nullptr);
    REQUIRE(td->source_node == x);
    REQUIRE(td->source_node->get_bind_hint() != nullptr);
    REQUIRE(*td->source_node->get_bind_hint() == hint);
}

TEST_CASE("TileGraph from_tensor_graph with add_inplace and fill",
          "[graph][tile]")
{
    TensorGraph tg_graph("complex_test");
    auto* x = tg_graph.data({4}, "x", DataType::FP32);
    auto* y = tg_graph.data({4}, "y", DataType::FP32);
    x->mark_input(true);
    y->mark_input(true);
    y->mark_output(true);

    gt::fill(Scalar(1.0), x);
    gt::add_inplace(Scalar(2.0), x, Scalar(1.0), y);

    TileGraph tile_graph = TileGraph::from_tensor_graph(tg_graph);

    REQUIRE(tile_graph.num_data() == 2);
    REQUIRE(tile_graph.num_ops() == 2);
    REQUIRE(tile_graph.ops()[0]->op_name() == "TILE_FILL");
    REQUIRE(tile_graph.ops()[1]->op_name() == "TILE_ADD_INPLACE");
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TileGraph from_tensor_graph execute matches TensorGraph",
    "[graph][tile]")
{
    constexpr Scalar alpha = 2.0;
    constexpr Scalar beta = 3.0;
    std::vector<Index> shape = {3, 4};
    const Index nelems = 12;

    TensorGraph tensor_graph("tensor_exec");
    auto* tx = tensor_graph.data(shape, "x", DataType::FP32);
    auto* ty = tensor_graph.data(shape, "y", DataType::FP32);
    tx->mark_input(true);
    ty->mark_input(true);

    auto* tz = gt::add(alpha, tx, beta, ty, "z");
    tz->mark_output(true);

    TileGraph tile_graph = TileGraph::from_tensor_graph(tensor_graph);

    std::vector<float> x_data(nelems), y_data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        x_data[i] = static_cast<float>(i);
        y_data[i] = static_cast<float>(-i - 1);
    }

    TensorGraph::Runtime tensor_rt(tensor_graph);
    tensor_rt.compile();
    tensor_rt.bind_data("x", x_data);
    tensor_rt.bind_data("y", y_data);
    tensor_rt.execute();
    tensor_rt.wait();
    auto tensor_result = tensor_rt.get_output<float>("z");

    TileGraph::Runtime tile_rt(tile_graph);
    tile_rt.compile();
    tile_rt.bind_data("x", x_data);
    tile_rt.bind_data("y", y_data);
    tile_rt.execute();
    tile_rt.wait();
    auto tile_result = tile_rt.get_output<float>("z");

    REQUIRE(tensor_result.size() == tile_result.size());
    constexpr float tol = 1e-5f;
    for(size_t i = 0; i < tensor_result.size(); ++i)
    {
        REQUIRE(std::abs(tensor_result[i] - tile_result[i]) < tol);
    }
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TileGraph add_inplace execute", "[graph][tile]")
{
    std::vector<Index> shape = {4};
    const Index nelems = 4;

    TileGraph graph("inplace_exec");
    auto* x = graph.data(shape, "x", DataType::FP32);
    auto* y = graph.data(shape, "y", DataType::FP32);
    x->mark_input(true);
    y->mark_input(true);
    y->mark_output(true);

    tg::add_inplace(Scalar(2.0), x, Scalar(1.0), y);

    TileGraph::Runtime runtime(graph);
    runtime.compile();

    std::vector<float> x_data = {1, 2, 3, 4};
    std::vector<float> y_data = {10, 20, 30, 40};

    runtime.bind_data("x", x_data);
    runtime.bind_data("y", y_data);
    runtime.execute();
    runtime.wait();

    auto result = runtime.get_output<float>("y");
    REQUIRE(result.size() == 4);
    constexpr float tol = 1e-5f;
    REQUIRE(std::abs(result[0] - 12.0f) < tol);
    REQUIRE(std::abs(result[1] - 24.0f) < tol);
    REQUIRE(std::abs(result[2] - 36.0f) < tol);
    REQUIRE(std::abs(result[3] - 48.0f) < tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TileGraph compile resolves bind hints from source TensorNode",
    "[graph][tile]")
{
    // Build a TensorGraph with a bind hint on x, convert, execute
    std::vector<Index> shape = {2};
    TensorGraph tg_graph("bind_via_source");
    auto* tx = tg_graph.data(shape, "x", DataType::FP32);
    auto* ty = tg_graph.data(shape, "y", DataType::FP32);
    tx->mark_input(true);
    ty->mark_input(true);

    auto* tz = gt::add(1.0, tx, 1.0, ty, "z");
    tz->mark_output(true);

    // Set bind hint on the TensorNode (not on TileNode)
    float x_vals[2] = {10.0f, 20.0f};
    std::vector<std::uint8_t> x_hint(sizeof(x_vals));
    std::memcpy(x_hint.data(), x_vals, sizeof(x_vals));
    tx->set_bind_hint(x_hint);

    TileGraph tile_graph = TileGraph::from_tensor_graph(tg_graph);

    TileGraph::Runtime tile_rt(tile_graph);
    tile_rt.compile();

    // x was initialized from the source TensorNode's bind hint;
    // we only need to bind y
    std::vector<float> y_data = {1.0f, 2.0f};
    tile_rt.bind_data("y", y_data);
    tile_rt.execute();
    tile_rt.wait();

    auto result = tile_rt.get_output<float>("z");
    constexpr float tol = 1e-5f;
    REQUIRE(result.size() == 2);
    REQUIRE(std::abs(result[0] - 11.0f) < tol);
    REQUIRE(std::abs(result[1] - 22.0f) < tol);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TileGraph fill and clear execute", "[graph][tile]")
{
    std::vector<Index> shape = {3};

    TileGraph graph("fill_clear_test");
    auto* x = graph.data(shape, "x", DataType::FP32);
    x->mark_input(true);
    x->mark_output(true);

    tg::fill(Scalar(42.0), x);
    tg::clear(x);

    TileGraph::Runtime runtime(graph);
    runtime.compile();

    std::vector<float> x_data = {7.0f, 8.0f, 9.0f};
    runtime.bind_data("x", x_data);
    runtime.execute();
    runtime.wait();

    auto result = runtime.get_output<float>("x");
    constexpr float tol = 1e-5f;
    REQUIRE(result.size() == 3);
    for(auto v : result)
    {
        REQUIRE(std::abs(v) < tol);
    }
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TileGraph multitile execute matches TensorGraph",
    "[graph][tile]")
{
    constexpr Scalar alpha = 1.0;
    constexpr Scalar beta = 1.0;
    std::vector<Index> shape = {6, 5};
    const Index nelems = 30;

    TensorGraph tensor_graph("mixed_tile_tg");
    auto* tx = tensor_graph.data(shape, "x", DataType::FP32);
    auto* ty = tensor_graph.data(shape, "y", DataType::FP32);
    tx->mark_input(true);
    ty->mark_input(true);

    // 2x2 tile grid: dim0 uniform (3+3), dim1 base+remainder (3+2) so
    // TensorGraph::Runtime accepts the axis tiling while tiles differ in size.
    tx->axis(0)->set_tiling(Index{3});
    tx->axis(1)->set_tiling(std::vector<Index>{3, 2});

    auto* tz = gt::add(alpha, tx, beta, ty, "z");
    tz->mark_output(true);

    TileGraph tile_graph = TileGraph::from_tensor_graph(tensor_graph);
    REQUIRE(tile_graph.tiling_scheme() != nullptr);
    REQUIRE(tile_graph.num_data() == 12);
    REQUIRE(tile_graph.num_ops() == 4);
    REQUIRE(tile_graph.get_tile_node("x") == nullptr);
    REQUIRE(tile_graph.get_tile_node("x__t0") != nullptr);

    std::vector<float> x_data(nelems), y_data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        x_data[static_cast<size_t>(i)] = static_cast<float>(i + 1);
        y_data[static_cast<size_t>(i)] = static_cast<float>(i * 2);
    }

    TensorGraph::Runtime tensor_rt(tensor_graph);
    tensor_rt.compile();
    tensor_rt.bind_data("x", x_data);
    tensor_rt.bind_data("y", y_data);
    tensor_rt.execute();
    tensor_rt.wait();
    auto tensor_result = tensor_rt.get_output<float>("z");

    TileGraph::Runtime tile_rt(tile_graph);
    tile_rt.compile();
    tile_rt.bind_data("x", x_data);
    tile_rt.bind_data("y", y_data);
    tile_rt.execute();
    tile_rt.wait();
    auto tile_result = tile_rt.get_output<float>("z");

    REQUIRE(tensor_result.size() == tile_result.size());
    constexpr float tol = 1e-5f;
    for(size_t i = 0; i < tensor_result.size(); ++i)
    {
        REQUIRE(std::abs(tensor_result[i] - tile_result[i]) < tol);
    }
}
