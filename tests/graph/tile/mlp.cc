/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/mlp.cc
 * Tiled TileGraph vs untiled TensorGraph for a small ReLU MLP (batch tiling).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <random>
#include <vector>

#include "context_fixture.hh"
#include <nntile/graph.hh>
#include <nntile/graph/module/activation.hh>
#include <nntile/graph/module/mlp.hh>

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;
namespace mod = nntile::graph::module;

namespace
{

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b)
{
    float m = 0.f;
    const size_t n = std::min(a.size(), b.size());
    for(size_t i = 0; i < n; ++i)
    {
        m = std::max(m, std::abs(a[i] - b[i]));
    }
    return m;
}

void bind_same_weights(
    TensorGraph::Runtime& rt,
    const std::string& w1,
    const std::string& w2,
    const std::vector<float>& w1_data,
    const std::vector<float>& w2_data)
{
    rt.bind_data(w1, w1_data);
    rt.bind_data(w2, w2_data);
}

void bind_same_weights(
    TileGraph::Runtime& rt,
    const std::string& w1,
    const std::string& w2,
    const std::vector<float>& w1_data,
    const std::vector<float>& w2_data)
{
    rt.bind_data(w1, w1_data);
    rt.bind_data(w2, w2_data);
}

} // namespace

TEST_CASE("MLP tiled vs tensor runtime parity", "[graph][tile]")
{
    test::ContextFixture fx;

    constexpr Index batch = 4;
    constexpr Index in_dim = 8;
    constexpr Index hid_dim = 6;
    constexpr Index out_dim = 3;

    std::mt19937 gen(43);
    std::normal_distribution<float> dist(0.f, 1.f);
    std::normal_distribution<float> dist_w(0.f, 0.1f);

    std::vector<float> in_data(static_cast<size_t>(batch * in_dim));
    std::vector<float> w1_data(static_cast<size_t>(in_dim * hid_dim));
    std::vector<float> w2_data(static_cast<size_t>(hid_dim * out_dim));
    for(auto& v : in_data)
    {
        v = dist(gen);
    }
    for(auto& v : w1_data)
    {
        v = dist_w(gen);
    }
    for(auto& v : w2_data)
    {
        v = dist_w(gen);
    }

    NNGraph g_ref("mlp_ref");
    mod::Mlp mlp_ref(
        &g_ref,
        "mlp",
        in_dim,
        hid_dim,
        out_dim,
        mod::ActivationType::RELU,
        DataType::FP32);

    auto* inp_ref =
        g_ref.tensor({batch, in_dim}, "in", DataType::FP32, true);
    inp_ref->mark_input(true);
    auto* out_ref = mlp_ref.forward(inp_ref);
    out_ref->mark_output(true);

    auto [g_out_ref, _] = g_ref.get_or_create_grad(out_ref, "dloss");
    gt::fill(nntile::Scalar(1.0f), g_out_ref->data());

    mlp_ref.fc1().weight_tensor()->mark_input(true);
    mlp_ref.fc2().weight_tensor()->mark_input(true);
    out_ref->backward();

    mlp_ref.fc1().weight_tensor()->grad()->mark_output(true);
    mlp_ref.fc2().weight_tensor()->grad()->mark_output(true);
    inp_ref->grad()->mark_output(true);

    TensorGraph::Runtime rt_ref(g_ref.tensor_graph());
    rt_ref.compile();
    rt_ref.bind_data("in", in_data);
    bind_same_weights(
        rt_ref,
        mlp_ref.fc1().weight_tensor()->name(),
        mlp_ref.fc2().weight_tensor()->name(),
        w1_data,
        w2_data);
    rt_ref.execute();
    rt_ref.wait();

    const std::vector<float> out_ref_v =
        rt_ref.get_output<float>(out_ref->name());
    const std::vector<float> gw1_ref = rt_ref.get_output<float>(
        mlp_ref.fc1().grad_name("weight"));
    const std::vector<float> gw2_ref = rt_ref.get_output<float>(
        mlp_ref.fc2().grad_name("weight"));

    NNGraph g_tile("mlp_tile");
    mod::Mlp mlp_tile(
        &g_tile,
        "mlp",
        in_dim,
        hid_dim,
        out_dim,
        mod::ActivationType::RELU,
        DataType::FP32);

    auto* inp_tile =
        g_tile.tensor({batch, in_dim}, "in", DataType::FP32, true);
    inp_tile->mark_input(true);
    auto* out_tile = mlp_tile.forward(inp_tile);
    out_tile->mark_output(true);

    auto [g_out_tile, __] = g_tile.get_or_create_grad(out_tile, "dloss");
    gt::fill(nntile::Scalar(1.0f), g_out_tile->data());

    mlp_tile.fc1().weight_tensor()->mark_input(true);
    mlp_tile.fc2().weight_tensor()->mark_input(true);
    out_tile->backward();

    mlp_tile.fc1().weight_tensor()->grad()->mark_output(true);
    mlp_tile.fc2().weight_tensor()->grad()->mark_output(true);
    inp_tile->grad()->mark_output(true);

    inp_tile->data()->axis(0)->set_tiling(std::vector<Index>{2, 1, 1});

    TileGraph tile_g = TileGraph::from_tensor_graph(g_tile.tensor_graph());
    TileGraph::Runtime rt_tile(tile_g);
    rt_tile.compile();
    rt_tile.bind_data("in", in_data);
    bind_same_weights(
        rt_tile,
        mlp_tile.fc1().weight_tensor()->name(),
        mlp_tile.fc2().weight_tensor()->name(),
        w1_data,
        w2_data);
    rt_tile.execute();
    rt_tile.wait();

    const std::vector<float> out_tile_v =
        rt_tile.get_output<float>(out_tile->name());
    const std::vector<float> gw1_tile = rt_tile.get_output<float>(
        mlp_tile.fc1().grad_name("weight"));
    const std::vector<float> gw2_tile = rt_tile.get_output<float>(
        mlp_tile.fc2().grad_name("weight"));

    constexpr float tol = 1e-4f;
    REQUIRE(max_abs_diff(out_ref_v, out_tile_v) < tol);
    REQUIRE(max_abs_diff(gw1_ref, gw1_tile) < tol);
    REQUIRE(max_abs_diff(gw2_ref, gw2_tile) < tol);
}
