/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/flash_sdpa_bwd_cudnn.cc
 * Test TensorGraph flash_sdpa_bwd_cudnn operation (CUDA only).
 *
 * @version 1.1.0
 * */

#include <nntile/defs.h>

#ifdef NNTILE_USE_CUDA

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "context_fixture.hh"
#include "nntile/graph/tensor/flash_sdpa_bwd_cudnn.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tensor/axis_descriptor.hh"
#include "nntile/tensor/clear.hh"
#include "nntile/tensor/flash_sdpa_bwd_cudnn.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

namespace
{

constexpr float tolerance = 1e-2f;
constexpr int distr_rank_single = 0;

} // anonymous namespace

TEST_CASE("TensorGraph flash_sdpa_bwd_cudnn structure", "[graph][tensor][cuda]")
{
    TensorGraph graph("test");

    std::vector<Index> kv_shape{32, 64, 2, 1, 1};
    std::vector<Index> logsumexp_shape{64, 2, 1, 1};
    std::vector<Index> mask_shape{64, 64};

    auto* K = graph.data(kv_shape, "K", DataType::FP16);
    auto* Q = graph.data(kv_shape, "Q", DataType::FP16);
    auto* V = graph.data(kv_shape, "V", DataType::FP16);
    auto* A = graph.data(kv_shape, "A", DataType::FP16);
    auto* dA = graph.data(kv_shape, "dA", DataType::FP16);
    auto* mask = graph.data(mask_shape, "mask", DataType::FP16);
    auto* logsumexp = graph.data(logsumexp_shape, "logsumexp", DataType::FP32);
    auto* dK = graph.data(kv_shape, "dK", DataType::FP16);
    auto* dQ = graph.data(kv_shape, "dQ", DataType::FP16);
    auto* dV = graph.data(kv_shape, "dV", DataType::FP16);

    gt::flash_sdpa_bwd_cudnn(K, Q, V, A, dA, mask, logsumexp, dK, dQ, dV);

    REQUIRE(graph.num_data() == 10);
    REQUIRE(graph.num_ops() == 1);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "FLASH_SDPA_BWD_CUDNN");
    REQUIRE(ops[0]->inputs().size() == 10);
    REQUIRE(ops[0]->outputs().size() == 3);
}

TEST_CASE("TensorGraph flash_sdpa_bwd_cudnn rejects null tensors", "[graph][tensor][cuda]")
{
    TensorGraph graph("test");
    std::vector<Index> kv_shape{32, 64, 2, 1, 1};
    std::vector<Index> logsumexp_shape{64, 2, 1, 1};
    std::vector<Index> mask_shape{64, 64};

    auto* K = graph.data(kv_shape, "K", DataType::FP16);
    auto* Q = graph.data(kv_shape, "Q", DataType::FP16);
    auto* V = graph.data(kv_shape, "V", DataType::FP16);
    auto* A = graph.data(kv_shape, "A", DataType::FP16);
    auto* dA = graph.data(kv_shape, "dA", DataType::FP16);
    auto* mask = graph.data(mask_shape, "mask", DataType::FP16);
    auto* logsumexp = graph.data(logsumexp_shape, "logsumexp", DataType::FP32);
    auto* dK = graph.data(kv_shape, "dK", DataType::FP16);
    auto* dQ = graph.data(kv_shape, "dQ", DataType::FP16);
    auto* dV = graph.data(kv_shape, "dV", DataType::FP16);

    REQUIRE_THROWS_AS(
        gt::flash_sdpa_bwd_cudnn(nullptr, Q, V, A, dA, mask, logsumexp, dK, dQ, dV),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::flash_sdpa_bwd_cudnn(K, nullptr, V, A, dA, mask, logsumexp, dK, dQ, dV),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::flash_sdpa_bwd_cudnn(K, Q, V, A, dA, mask, logsumexp, nullptr, dQ, dV),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::CudaContextFixture,
    "TensorGraph flash_sdpa_bwd_cudnn matches tensor API", "[graph][tensor][cuda]")
{
    using T = nntile::fp16_t;
    using Y = typename T::repr_t;

    Index head_size = 32;
    Index n_seq = 64;
    Index n_batch = 2;
    Index kv_group_size = 1;
    Index n_head_kv = 1;

    std::vector<Index> kv_shape = {head_size, n_seq, n_batch, kv_group_size, n_head_kv};
    std::vector<Index> logsumexp_shape = {n_seq, n_batch, kv_group_size, n_head_kv};
    std::vector<Index> mask_shape = {n_seq, n_seq};

    const Index kv_nelems = std::accumulate(
        kv_shape.begin(), kv_shape.end(), Index(1), std::multiplies<>());
    const Index logsumexp_nelems = std::accumulate(
        logsumexp_shape.begin(), logsumexp_shape.end(), Index(1), std::multiplies<>());
    const Index mask_nelems = n_seq * n_seq;

    // --- TensorGraph path ---
    TensorGraph graph("flash_sdpa_bwd_test");
    auto* K_node = graph.data(kv_shape, "K", DataType::FP16);
    auto* Q_node = graph.data(kv_shape, "Q", DataType::FP16);
    auto* V_node = graph.data(kv_shape, "V", DataType::FP16);
    auto* A_node = graph.data(kv_shape, "A", DataType::FP16);
    auto* dA_node = graph.data(kv_shape, "dA", DataType::FP16);
    auto* mask_node = graph.data(mask_shape, "mask", DataType::FP16);
    auto* logsumexp_node = graph.data(logsumexp_shape, "logsumexp", DataType::FP32);
    auto* dK_node = graph.data(kv_shape, "dK", DataType::FP16);
    auto* dQ_node = graph.data(kv_shape, "dQ", DataType::FP16);
    auto* dV_node = graph.data(kv_shape, "dV", DataType::FP16);

    K_node->mark_input(true);
    Q_node->mark_input(true);
    V_node->mark_input(true);
    A_node->mark_input(true);
    dA_node->mark_input(true);
    mask_node->mark_input(true);
    logsumexp_node->mark_input(true);
    dK_node->mark_input(true);
    dQ_node->mark_input(true);
    dV_node->mark_input(true);
    dK_node->mark_output(true);
    dQ_node->mark_output(true);
    dV_node->mark_output(true);

    gt::flash_sdpa_bwd_cudnn(K_node, Q_node, V_node, A_node, dA_node,
                         mask_node, logsumexp_node, dK_node, dQ_node, dV_node);

    TensorGraph::Runtime runtime(graph);
    runtime.compile();

    std::vector<float> K_data(kv_nelems);
    std::vector<float> Q_data(kv_nelems);
    std::vector<float> V_data(kv_nelems);
    std::vector<float> A_data(kv_nelems);
    std::vector<float> dA_data(kv_nelems);
    std::vector<float> mask_data(mask_nelems);
    std::vector<float> logsumexp_data(logsumexp_nelems);
    std::vector<float> dK_data(kv_nelems, 0.0f);
    std::vector<float> dQ_data(kv_nelems, 0.0f);
    std::vector<float> dV_data(kv_nelems, 0.0f);

    for(Index i = 0; i < kv_nelems; ++i)
    {
        K_data[i] = 0.1f * static_cast<float>((i % 10) - 5);
        Q_data[i] = 0.1f * static_cast<float>(((i + 1) % 10) - 5);
        V_data[i] = 0.1f * static_cast<float>(((i + 2) % 10) - 5);
        A_data[i] = 0.1f * static_cast<float>(((i + 3) % 10) - 5);
        dA_data[i] = 0.1f * static_cast<float>(((i + 4) % 10) - 5);
    }
    for(Index i = 0; i < n_seq; ++i)
    {
        for(Index j = 0; j < n_seq; ++j)
        {
            Index idx = i * n_seq + j;
            mask_data[idx] = (std::abs(static_cast<long>(i) - static_cast<long>(j)) <= 32)
                ? 0.0f : -std::numeric_limits<float>::infinity();
        }
    }
    for(Index i = 0; i < logsumexp_nelems; ++i)
    {
        logsumexp_data[i] = 0.1f * static_cast<float>((i % 10) - 5);
    }

    runtime.bind_data("K", K_data);
    runtime.bind_data("Q", Q_data);
    runtime.bind_data("V", V_data);
    runtime.bind_data("A", A_data);
    runtime.bind_data("dA", dA_data);
    runtime.bind_data("mask", mask_data);
    runtime.bind_data("logsumexp", logsumexp_data);
    runtime.bind_data("dK", dK_data);
    runtime.bind_data("dQ", dQ_data);
    runtime.bind_data("dV", dV_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_dK = runtime.get_output<float>("dK");
    std::vector<float> graph_dQ = runtime.get_output<float>("dQ");
    std::vector<float> graph_dV = runtime.get_output<float>("dV");

    // --- Direct tensor API path ---
    nntile::tensor::TensorTraits kv_traits(kv_shape, kv_shape);
    nntile::tensor::TensorTraits mask_traits(mask_shape, mask_shape);
    nntile::tensor::TensorTraits logsumexp_traits(logsumexp_shape, logsumexp_shape);
    std::vector<int> distr(1, distr_rank_single);

    nntile::tensor::Tensor<T> K_t(kv_traits, distr);
    nntile::tensor::Tensor<T> Q_t(kv_traits, distr);
    nntile::tensor::Tensor<T> V_t(kv_traits, distr);
    nntile::tensor::Tensor<T> A_t(kv_traits, distr);
    nntile::tensor::Tensor<T> dA_t(kv_traits, distr);
    nntile::tensor::Tensor<T> mask_t(mask_traits, distr);
    nntile::tensor::Tensor<nntile::fp32_t> logsumexp_t(logsumexp_traits, distr);
    nntile::tensor::Tensor<T> dK_t(kv_traits, distr);
    nntile::tensor::Tensor<T> dQ_t(kv_traits, distr);
    nntile::tensor::Tensor<T> dV_t(kv_traits, distr);

    auto init_tile = [](nntile::tensor::Tensor<T>& t, const std::vector<float>& data)
    {
        auto tile = t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < static_cast<Index>(data.size()); ++i)
        {
            loc[i] = static_cast<Y>(data[i]);
        }
        loc.release();
    };
    auto init_logsumexp = [](nntile::tensor::Tensor<nntile::fp32_t>& t,
                            const std::vector<float>& data)
    {
        auto tile = t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < static_cast<Index>(data.size()); ++i)
        {
            loc[i] = data[i];
        }
        loc.release();
    };

    init_tile(K_t, K_data);
    init_tile(Q_t, Q_data);
    init_tile(V_t, V_data);
    init_tile(A_t, A_data);
    init_tile(dA_t, dA_data);
    init_tile(mask_t, mask_data);
    init_logsumexp(logsumexp_t, logsumexp_data);
    nntile::tensor::clear<T>(dK_t);
    nntile::tensor::clear<T>(dQ_t);
    nntile::tensor::clear<T>(dV_t);

    nntile::tensor::flash_sdpa_bwd_cudnn<T>(K_t, Q_t, V_t, A_t, dA_t, mask_t, logsumexp_t,
                                    dK_t, dQ_t, dV_t);
    starpu_task_wait_for_all();

    std::vector<float> tensor_dK(kv_nelems);
    std::vector<float> tensor_dQ(kv_nelems);
    std::vector<float> tensor_dV(kv_nelems);
    auto read_tensor = [&](nntile::tensor::Tensor<T>& t, std::vector<float>& out)
    {
        auto tile = t.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        for(Index i = 0; i < kv_nelems; ++i)
        {
            out[i] = static_cast<float>(loc[i]);
        }
        loc.release();
    };
    read_tensor(dK_t, tensor_dK);
    read_tensor(dQ_t, tensor_dQ);
    read_tensor(dV_t, tensor_dV);

    REQUIRE(graph_dK.size() == tensor_dK.size());
    REQUIRE(graph_dQ.size() == tensor_dQ.size());
    REQUIRE(graph_dV.size() == tensor_dV.size());
    for(size_t i = 0; i < graph_dK.size(); ++i)
    {
        REQUIRE(std::abs(graph_dK[i] - tensor_dK[i]) < tolerance);
        REQUIRE(std::abs(graph_dQ[i] - tensor_dQ[i]) < tolerance);
        REQUIRE(std::abs(graph_dV[i] - tensor_dV[i]) < tolerance);
    }
}

TEST_CASE_METHOD(nntile::test::CudaContextFixture,
    "TensorGraph flash_sdpa_bwd_cudnn tiled matches untiled", "[graph][tensor][cuda]")
{
    Index head_size = 32;
    Index n_seq = 64;
    Index n_batch = 2;
    Index kv_group_size = 1;
    Index n_head_kv = 1;

    std::vector<Index> kv_shape = {head_size, n_seq, n_batch, kv_group_size, n_head_kv};
    std::vector<Index> logsumexp_shape = {n_seq, n_batch, kv_group_size, n_head_kv};
    std::vector<Index> mask_shape = {n_seq, n_seq};

    const Index kv_nelems = std::accumulate(
        kv_shape.begin(), kv_shape.end(), Index(1), std::multiplies<>());
    const Index logsumexp_nelems = std::accumulate(
        logsumexp_shape.begin(), logsumexp_shape.end(), Index(1), std::multiplies<>());
    const Index mask_nelems = n_seq * n_seq;

    std::vector<float> K_data(kv_nelems);
    std::vector<float> Q_data(kv_nelems);
    std::vector<float> V_data(kv_nelems);
    std::vector<float> A_data(kv_nelems);
    std::vector<float> dA_data(kv_nelems);
    std::vector<float> mask_data(mask_nelems);
    std::vector<float> logsumexp_data(logsumexp_nelems);
    std::vector<float> dK_data(kv_nelems, 0.0f);
    std::vector<float> dQ_data(kv_nelems, 0.0f);
    std::vector<float> dV_data(kv_nelems, 0.0f);

    for(Index i = 0; i < kv_nelems; ++i)
    {
        K_data[i] = 0.1f * static_cast<float>((i % 10) - 5);
        Q_data[i] = 0.1f * static_cast<float>(((i + 1) % 10) - 5);
        V_data[i] = 0.1f * static_cast<float>(((i + 2) % 10) - 5);
        A_data[i] = 0.1f * static_cast<float>(((i + 3) % 10) - 5);
        dA_data[i] = 0.1f * static_cast<float>(((i + 4) % 10) - 5);
    }
    for(Index i = 0; i < n_seq; ++i)
    {
        for(Index j = 0; j < n_seq; ++j)
        {
            mask_data[i * n_seq + j] =
                (std::abs(static_cast<long>(i) - static_cast<long>(j)) <= 32)
                    ? 0.0f : -std::numeric_limits<float>::infinity();
        }
    }
    for(Index i = 0; i < logsumexp_nelems; ++i)
    {
        logsumexp_data[i] = 0.1f * static_cast<float>((i % 10) - 5);
    }

    auto run_graph = [&](bool tiled)
        -> std::tuple<std::vector<float>, std::vector<float>, std::vector<float>>
    {
        TensorGraph graph(tiled ? "bwd_tiled" : "bwd_untiled");
        auto* K_node = graph.data(kv_shape, "K", DataType::FP16);
        auto* Q_node = graph.data(kv_shape, "Q", DataType::FP16);
        auto* V_node = graph.data(kv_shape, "V", DataType::FP16);
        auto* A_node = graph.data(kv_shape, "A", DataType::FP16);
        auto* dA_node = graph.data(kv_shape, "dA", DataType::FP16);
        auto* mask_node = graph.data(mask_shape, "mask", DataType::FP16);
        auto* logsumexp_node = graph.data(logsumexp_shape, "logsumexp",
                                          DataType::FP32);
        auto* dK_node = graph.data(kv_shape, "dK", DataType::FP16);
        auto* dQ_node = graph.data(kv_shape, "dQ", DataType::FP16);
        auto* dV_node = graph.data(kv_shape, "dV", DataType::FP16);

        K_node->mark_input(true);
        Q_node->mark_input(true);
        V_node->mark_input(true);
        A_node->mark_input(true);
        dA_node->mark_input(true);
        mask_node->mark_input(true);
        logsumexp_node->mark_input(true);
        dK_node->mark_input(true);
        dQ_node->mark_input(true);
        dV_node->mark_input(true);
        dK_node->mark_output(true);
        dQ_node->mark_output(true);
        dV_node->mark_output(true);

        gt::flash_sdpa_bwd_cudnn(K_node, Q_node, V_node, A_node, dA_node,
                                 mask_node, logsumexp_node,
                                 dK_node, dQ_node, dV_node);

        if(tiled)
        {
            auto* head_axis = K_node->axis(0);
            for(auto* ag : graph.axis_groups())
            {
                if(ag == head_axis)
                {
                    ag->set_tiling(ag->extent);
                }
                else
                {
                    ag->set_tiling((ag->extent + 1) / 2);
                }
            }
        }

        TensorGraph::Runtime runtime(graph);
        runtime.compile();
        runtime.bind_data("K", K_data);
        runtime.bind_data("Q", Q_data);
        runtime.bind_data("V", V_data);
        runtime.bind_data("A", A_data);
        runtime.bind_data("dA", dA_data);
        runtime.bind_data("mask", mask_data);
        runtime.bind_data("logsumexp", logsumexp_data);
        runtime.bind_data("dK", dK_data);
        runtime.bind_data("dQ", dQ_data);
        runtime.bind_data("dV", dV_data);
        runtime.execute();
        runtime.wait();
        return {runtime.get_output<float>("dK"),
                runtime.get_output<float>("dQ"),
                runtime.get_output<float>("dV")};
    };

    auto [untiled_dK, untiled_dQ, untiled_dV] = run_graph(false);
    auto [tiled_dK, tiled_dQ, tiled_dV] = run_graph(true);

    REQUIRE(tiled_dK.size() == untiled_dK.size());
    REQUIRE(tiled_dQ.size() == untiled_dQ.size());
    REQUIRE(tiled_dV.size() == untiled_dV.size());
    for(size_t i = 0; i < tiled_dK.size(); ++i)
    {
        REQUIRE(std::abs(tiled_dK[i] - untiled_dK[i]) < tolerance);
        REQUIRE(std::abs(tiled_dQ[i] - untiled_dQ[i]) < tolerance);
        REQUIRE(std::abs(tiled_dV[i] - untiled_dV[i]) < tolerance);
    }
}

#endif // NNTILE_USE_CUDA
