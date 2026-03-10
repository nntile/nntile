/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/flash_sdpa_fwd_cudnn.cc
 * Test TensorGraph flash_sdpa_fwd_cudnn operation (CUDA only).
 *
 * @version 1.1.0
 * */

#include "nntile/defs.h"

#ifdef NNTILE_USE_CUDA

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "context_fixture.hh"
#include "nntile/graph/tensor/flash_sdpa_fwd_cudnn.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/flash_sdpa_fwd_cudnn.hh"
#include "nntile/tensor/tensor.hh"
#include "nntile/tensor/clear.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

namespace
{

constexpr float tolerance = 1e-2f;
constexpr int distr_rank_single = 0;

} // anonymous namespace

TEST_CASE("TensorGraph flash_sdpa_fwd_cudnn structure", "[graph][tensor][cuda]")
{
    TensorGraph graph("test");

    // K, Q, V, A: 5D (head_size, n_seq, n_batch, kv_group_size, n_head_kv)
    std::vector<Index> kv_shape{32, 64, 2, 1, 1};
    std::vector<Index> logsumexp_shape{64, 2, 1, 1};
    std::vector<Index> mask_shape{64, 64};

    auto* K = graph.data(kv_shape, "K", DataType::FP16);
    auto* Q = graph.data(kv_shape, "Q", DataType::FP16);
    auto* mask = graph.data(mask_shape, "mask", DataType::FP16);
    auto* V = graph.data(kv_shape, "V", DataType::FP16);

    auto* A = gt::flash_sdpa_fwd_cudnn(K, Q, mask, V, "logsumexp", "A");

    REQUIRE(graph.num_data() == 6);
    REQUIRE(graph.num_ops() == 1);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "FLASH_SDPA_FWD_CUDNN");
    REQUIRE(ops[0]->inputs().size() == 4);
    REQUIRE(ops[0]->outputs().size() == 2);
    REQUIRE(ops[0]->outputs()[1] == A);
    REQUIRE(A->shape() == kv_shape);
}

TEST_CASE("TensorGraph flash_sdpa_fwd_cudnn rejects null tensors", "[graph][tensor][cuda]")
{
    TensorGraph graph("test");
    std::vector<Index> kv_shape{32, 64, 2, 1, 1};
    std::vector<Index> mask_shape{64, 64};

    auto* K = graph.data(kv_shape, "K", DataType::FP16);
    auto* Q = graph.data(kv_shape, "Q", DataType::FP16);
    auto* mask = graph.data(mask_shape, "mask", DataType::FP16);
    auto* V = graph.data(kv_shape, "V", DataType::FP16);

    REQUIRE_THROWS_AS(
        gt::flash_sdpa_fwd_cudnn(nullptr, Q, mask, V, "logsumexp", "A"),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::flash_sdpa_fwd_cudnn(K, nullptr, mask, V, "logsumexp", "A"),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::flash_sdpa_fwd_cudnn(K, Q, nullptr, V, "logsumexp", "A"),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::flash_sdpa_fwd_cudnn(K, Q, mask, nullptr, "logsumexp", "A"),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::CudaContextFixture,
    "TensorGraph flash_sdpa_fwd_cudnn matches tensor API", "[graph][tensor][cuda]")
{
    using T = nntile::fp16_t;
    using Y = typename T::repr_t;

    Index head_size = 32;
    Index n_seq = 64;
    Index n_batch = 2;
    Index kv_group_size = 1;
    Index n_head_kv = 1;

    std::vector<Index> K_shape = {head_size, n_seq, n_batch, kv_group_size, n_head_kv};
    std::vector<Index> logsumexp_shape = {n_seq, n_batch, kv_group_size, n_head_kv};
    std::vector<Index> mask_shape = {n_seq, n_seq};

    const Index kv_nelems = std::accumulate(
        K_shape.begin(), K_shape.end(), Index(1), std::multiplies<>());
    const Index logsumexp_nelems = std::accumulate(
        logsumexp_shape.begin(), logsumexp_shape.end(), Index(1), std::multiplies<>());
    const Index mask_nelems = n_seq * n_seq;

    // --- TensorGraph path ---
    TensorGraph graph("flash_sdpa_fwd_test");
    auto* K_node = graph.data(K_shape, "K", DataType::FP16);
    auto* Q_node = graph.data(K_shape, "Q", DataType::FP16);
    auto* mask_node = graph.data(mask_shape, "mask", DataType::FP16);
    auto* V_node = graph.data(K_shape, "V", DataType::FP16);
    K_node->mark_input(true);
    Q_node->mark_input(true);
    mask_node->mark_input(true);
    V_node->mark_input(true);

    auto* A_node = gt::flash_sdpa_fwd_cudnn(K_node, Q_node, mask_node, V_node,
                                        "logsumexp", "A");
    A_node->mark_output(true);

    TensorGraph::Runtime runtime(graph);
    runtime.compile();

    std::vector<float> K_data(kv_nelems);
    std::vector<float> Q_data(kv_nelems);
    std::vector<float> mask_data(mask_nelems);
    std::vector<float> V_data(kv_nelems);
    for(Index i = 0; i < kv_nelems; ++i)
    {
        K_data[i] = 0.1f * static_cast<float>((i % 10) - 5);
        Q_data[i] = 0.1f * static_cast<float>(((i + 1) % 10) - 5);
        V_data[i] = 0.1f * static_cast<float>(((i + 2) % 10) - 5);
    }
    for(Index i = 0; i < n_seq; ++i)
    {
        for(Index j = 0; j < n_seq; ++j)
        {
            Index idx = i * n_seq + j;
            mask_data[idx] = (j <= i) ? 0.0f : -std::numeric_limits<float>::infinity();
        }
    }

    runtime.bind_data("K", K_data);
    runtime.bind_data("Q", Q_data);
    runtime.bind_data("mask", mask_data);
    runtime.bind_data("V", V_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_A = runtime.get_output<float>("A");

    // --- Direct tensor API path ---
    nntile::tensor::TensorTraits K_traits(K_shape, K_shape);
    nntile::tensor::TensorTraits Q_traits(K_shape, K_shape);
    nntile::tensor::TensorTraits V_traits(K_shape, K_shape);
    nntile::tensor::TensorTraits A_traits(K_shape, K_shape);
    nntile::tensor::TensorTraits mask_traits(mask_shape, mask_shape);
    nntile::tensor::TensorTraits logsumexp_traits(logsumexp_shape, logsumexp_shape);
    std::vector<int> distr(1, distr_rank_single);

    nntile::tensor::Tensor<T> K_t(K_traits, distr);
    nntile::tensor::Tensor<T> Q_t(Q_traits, distr);
    nntile::tensor::Tensor<T> V_t(V_traits, distr);
    nntile::tensor::Tensor<T> A_t(A_traits, distr);
    nntile::tensor::Tensor<T> mask_t(mask_traits, distr);
    nntile::tensor::Tensor<nntile::fp32_t> logsumexp_t(logsumexp_traits, distr);

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
    auto init_logsumexp = [](nntile::tensor::Tensor<nntile::fp32_t>& t)
    {
        auto tile = t.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < tile.nelems; ++i)
        {
            loc[i] = -std::numeric_limits<float>::infinity();
        }
        loc.release();
    };

    init_tile(K_t, K_data);
    init_tile(Q_t, Q_data);
    init_tile(V_t, V_data);
    init_tile(mask_t, mask_data);
    init_logsumexp(logsumexp_t);
    nntile::tensor::clear_async(A_t);

    nntile::tensor::flash_sdpa_fwd_cudnn<T>(K_t, Q_t, mask_t, logsumexp_t, V_t, A_t);

    std::vector<float> tensor_A(kv_nelems);
    {
        auto tile = A_t.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        for(Index i = 0; i < kv_nelems; ++i)
        {
            tensor_A[i] = static_cast<float>(loc[i]);
        }
        loc.release();
    }

    REQUIRE(graph_A.size() == tensor_A.size());
    for(size_t i = 0; i < graph_A.size(); ++i)
    {
        REQUIRE(std::abs(graph_A[i] - tensor_A[i]) < tolerance);
    }
}

#endif // NNTILE_USE_CUDA
