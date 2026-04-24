/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file examples/autograd_mlp_tile_compare_example.cc
 * ReLU MLP: TensorGraph reference run vs TileGraph::Runtime with
 * heterogeneous tiling on every tensor axis equivalence group, on one NNGraph
 * tensor graph. Dumps TensorGraph::to_string() (before/after tiling) and
 * TileGraph::to_string(). Compares forward output and weight gradients.
 *
 * @version 1.1.0
 * */

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <nntile/context.hh>
#include <nntile/graph.hh>
#include <nntile/graph/module/activation.hh>
#include <nntile/graph/module/mlp.hh>

namespace gt = nntile::graph::tensor;
namespace mod = nntile::graph::module;

//! Label each axis equivalence class for this MLP layout. Extents
//! batch / in_dim / hid_dim / out_dim must be pairwise distinct (true here).
static void name_mlp_axis_groups_from_extents(
    nntile::graph::TensorGraph& tg,
    nntile::Index batch,
    nntile::Index in_dim,
    nntile::Index hid_dim,
    nntile::Index out_dim)
{
    for(nntile::graph::AxisDescriptor* ad : tg.axis_groups())
    {
        if(!ad->name.empty())
        {
            continue;
        }
        if(ad->extent == batch)
        {
            ad->name = "batch";
        }
        else if(ad->extent == in_dim)
        {
            ad->name = "in_features";
        }
        else if(ad->extent == hid_dim)
        {
            ad->name = "hidden";
        }
        else if(ad->extent == out_dim)
        {
            ad->name = "out_features";
        }
        else if(!ad->members.empty())
        {
            auto* node = static_cast<nntile::graph::TensorGraph::TensorNode*>(
                ad->members[0].first);
            const int ax = ad->members[0].second;
            ad->name = node->name() + "_axis" + std::to_string(ax);
        }
        else
        {
            ad->name = "extent_" + std::to_string(static_cast<long long>(ad->extent));
        }
    }
}

//! Segment sizes for AxisDescriptor::set_tiling: positive, sum to `extent`.
//! Uses unequal tile sizes whenever extent >= 3 (extent 2 only allows 1+1).
static std::vector<nntile::Index> heterogeneous_tile_sizes(nntile::Index extent)
{
    if(extent < 1)
    {
        throw std::invalid_argument("heterogeneous_tile_sizes: extent >= 1");
    }
    if(extent == 1)
    {
        return {1};
    }
    if(extent == 2)
    {
        return {1, 1};
    }
    if(extent == 3)
    {
        return {1, 2};
    }
    if(extent == 4)
    {
        return {1, 1, 2};
    }
    return {1, 2, static_cast<nntile::Index>(extent - 3)};
}

//! Sets a heterogeneous split on every axis group that is still untiled.
static void tile_all_axis_groups_heterogeneous(nntile::graph::TensorGraph& tg)
{
    for(nntile::graph::AxisDescriptor* ad : tg.axis_groups())
    {
        if(!ad->is_tiled())
        {
            ad->set_tiling(heterogeneous_tile_sizes(ad->extent));
        }
    }
}

//! Max over elements of |ref_i - other_i| / max(|ref_i|, |other_i|, eps).
static float max_per_element_rel_error(
    const std::vector<float>& ref, const std::vector<float>& other)
{
    constexpr float eps = 1e-7f;
    float m = 0.f;
    const size_t n = std::min(ref.size(), other.size());
    for(size_t i = 0; i < n; ++i)
    {
        const float num = std::abs(ref[i] - other[i]);
        const float denom = std::max(
            std::max(std::abs(ref[i]), std::abs(other[i])), eps);
        m = std::max(m, num / denom);
    }
    return m;
}

//! ||ref - other||_F / max(||ref||_F, ||other||_F, eps).
static float frob_per_tensor_rel_error(
    const std::vector<float>& ref, const std::vector<float>& other)
{
    constexpr double eps = 1e-7;
    const size_t n = std::min(ref.size(), other.size());
    double diff_sq = 0.;
    double ref_sq = 0.;
    double other_sq = 0.;
    for(size_t i = 0; i < n; ++i)
    {
        const double ri = static_cast<double>(ref[i]);
        const double oi = static_cast<double>(other[i]);
        const double d = ri - oi;
        diff_sq += d * d;
        ref_sq += ri * ri;
        other_sq += oi * oi;
    }
    const double diff_norm = std::sqrt(diff_sq);
    const double ref_norm = std::sqrt(ref_sq);
    const double other_norm = std::sqrt(other_sq);
    const double denom =
        std::max(std::max(ref_norm, other_norm), eps);
    return static_cast<float>(diff_norm / denom);
}

static void bind_same_weights(
    nntile::graph::TileGraph::Runtime& rt,
    const std::string& w1,
    const std::string& w2,
    const std::vector<float>& w1_data,
    const std::vector<float>& w2_data)
{
    rt.bind_data(w1, w1_data);
    rt.bind_data(w2, w2_data);
}

int main()
{
    using namespace nntile::graph;
    using nntile::Index;

    nntile::Context context(
        1, // ncpu: StarPU CPU workers (-1 => STARPU_NCPU)
        0, // ncuda: CUDA workers (-1 => STARPU_NCUDA; 0 => none)
        0, // ooc: out-of-core (nonzero enables disk-backed buffers)
        "/tmp/nntile_ooc", // ooc_path: backing store when OOC is on
        16777216, // ooc_size: OOC disk budget in bytes (16 MiB here)
        0, // logger: nonzero enables remote logging client
        "localhost", // logger_addr: logging server hostname
        5001, // logger_port: logging server TCP port
        0); // verbose: nonzero enables extra NNTile diagnostics

    constexpr Index batch = 4;
    constexpr Index in_dim = 8;
    constexpr Index hid_dim = 6;
    constexpr Index out_dim = 3;

    std::mt19937 gen(42);
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

    // One NNGraph: the same TensorGraph is used for the untiled reference run
    // and, after heterogeneous tiling is set on all axis groups below, as the
    // source for TileGraph::from_tensor_graph (no duplicate graph / MLP).
    NNGraph nn("mlp_compare");
    mod::Mlp mlp(
        &nn,
        "mlp",
        in_dim,
        hid_dim,
        out_dim,
        mod::ActivationType::RELU,
        DataType::FP32);

    auto* inp = nn.tensor({batch, in_dim}, "in", DataType::FP32, true);
    inp->mark_input(true);
    auto* out = mlp.forward(inp);
    out->mark_output(true);

    auto [g_out, _] = nn.get_or_create_grad(out, "dloss");
    gt::fill(nntile::Scalar(1.0f), g_out->data());

    mlp.fc1().weight_tensor()->mark_input(true);
    mlp.fc2().weight_tensor()->mark_input(true);
    out->backward();

    mlp.fc1().weight_tensor()->grad()->mark_output(true);
    mlp.fc2().weight_tensor()->grad()->mark_output(true);
    inp->grad()->mark_output(true);

    // String keys for weight gradient tensors in this graph (for
    // Runtime::get_output after execute). Module::grad_name is
    // a pure naming helper: it returns "<module>_<local_param>_grad" (here
    // local param is "weight") and does not compute any derivatives.
    const std::string dW1_grad_tensor_name = mlp.fc1().grad_name("weight");
    const std::string dW2_grad_tensor_name = mlp.fc2().grad_name("weight");

    TensorGraph& tensor_g = nn.tensor_graph();
    name_mlp_axis_groups_from_extents(
        tensor_g, batch, in_dim, hid_dim, out_dim);

    // --- Reference (from_tensor_graph, default tiling) ---
    TileGraph rt_tensor_tile = TileGraph::from_tensor_graph(tensor_g);

    TileGraph::Runtime rt_tensor(rt_tensor_tile);
    rt_tensor.compile();
    rt_tensor.bind_data("in", in_data);
    bind_same_weights(
        rt_tensor,
        mlp.fc1().weight_tensor()->name(),
        mlp.fc2().weight_tensor()->name(),
        w1_data,
        w2_data);
    rt_tensor.execute();
    rt_tensor.wait();

    std::cout << "=== TensorGraph::to_string() (before per-axis tiling) ===\n"
              << tensor_g.to_string() << "\n";

    const std::vector<float> out_ref_v =
        rt_tensor.get_output<float>(out->name());
    const std::vector<float> gw1_ref =
        rt_tensor.get_output<float>(dW1_grad_tensor_name);
    const std::vector<float> gw2_ref =
        rt_tensor.get_output<float>(dW2_grad_tensor_name);

    // --- Tiled (TileGraph from the same tensor graph) ---
    // Run untiled first; then set a heterogeneous split on every axis
    // equivalence group so TensorGraphTiling lowers the whole graph to tiles.
    tile_all_axis_groups_heterogeneous(tensor_g);

    std::cout << "=== TensorGraph::to_string() (after per-axis tiling) ===\n"
              << tensor_g.to_string() << "\n";

    TileGraph tile_g = TileGraph::from_tensor_graph(tensor_g);
    std::cout << "=== TileGraph::to_string() ===\n"
              << tile_g.to_string() << "\n";
    TileGraph::Runtime rt_tile(tile_g);
    rt_tile.compile();
    rt_tile.bind_data("in", in_data);
    bind_same_weights(
        rt_tile,
        mlp.fc1().weight_tensor()->name(),
        mlp.fc2().weight_tensor()->name(),
        w1_data,
        w2_data);
    rt_tile.execute();
    rt_tile.wait();

    const std::vector<float> out_tile_v =
        rt_tile.get_output<float>(out->name());
    const std::vector<float> gw1_tile =
        rt_tile.get_output<float>(dW1_grad_tensor_name);
    const std::vector<float> gw2_tile =
        rt_tile.get_output<float>(dW2_grad_tensor_name);

    constexpr float tol = 5e-4f;
    const float e_out_el = max_per_element_rel_error(out_ref_v, out_tile_v);
    const float e_out_fr = frob_per_tensor_rel_error(out_ref_v, out_tile_v);
    const float e_g1_el = max_per_element_rel_error(gw1_ref, gw1_tile);
    const float e_g1_fr = frob_per_tensor_rel_error(gw1_ref, gw1_tile);
    const float e_g2_el = max_per_element_rel_error(gw2_ref, gw2_tile);
    const float e_g2_fr = frob_per_tensor_rel_error(gw2_ref, gw2_tile);

    std::cout << "output: per-element rel (max) " << e_out_el
              << ", per-tensor rel (Frobenius) " << e_out_fr << "\n";
    std::cout << "dW1:    per-element rel (max) " << e_g1_el
              << ", per-tensor rel (Frobenius) " << e_g1_fr << "\n";
    std::cout << "dW2:    per-element rel (max) " << e_g2_el
              << ", per-tensor rel (Frobenius) " << e_g2_fr << "\n";

    if(e_out_el > tol || e_out_fr > tol || e_g1_el > tol || e_g1_fr > tol
        || e_g2_el > tol || e_g2_fr > tol)
    {
        std::cerr << "autograd_mlp_tile_compare_example: tolerance exceeded\n";
        return 1;
    }

    std::cout << "autograd_mlp_tile_compare_example: OK\n";
    return 0;
}
