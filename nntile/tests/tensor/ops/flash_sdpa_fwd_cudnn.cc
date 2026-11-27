/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/flash_sdpa_fwd_cudnn.cc
 * Test TensorGraph flash_sdpa_fwd_cudnn operation (CUDA only).
 *
 * @version 1.1.0
 * */

#include "nntile/defs.h"

#ifdef NNTILE_USE_CUDA

#include "context_fixture.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tensor/ops/flash_sdpa_fwd_cudnn.hh"
#include "nntile/tile.hh"
#include "nntile/tensor/ops/clear.hh"
#include "nntile/tensor/ops/flash_sdpa_fwd_cudnn.hh"
#include "nntile/tensor.hh"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

using namespace nntile;
using namespace nntile;
namespace gt = nntile::tensor;

namespace
{

constexpr float tolerance = 1e-2f;
constexpr int distr_rank_single = 0;

} // anonymous namespace

TEST_CASE(
    "TensorGraph flash_sdpa_fwd_cudnn structure", "[graph][tensor][cuda]")
{
    TensorGraph graph("test");

    // K, Q, V, A: 5D (head_size, n_seq, n_batch, kv_group_size, n_head_kv)
    std::vector<Index> kv_shape{32, 64, 2, 1, 1};
    std::vector<Index> logsumexp_shape{64, 2, 1, 1};
    std::vector<Index> mask_shape{64, 64};

    nntile::TensorRef K = graph.data(kv_shape, DataType::FP16);
    K->set_name("K");
    nntile::TensorRef Q = graph.data(kv_shape, DataType::FP16);
    Q->set_name("Q");
    nntile::TensorRef mask = graph.data(mask_shape, DataType::FP16);
    mask->set_name("mask");
    nntile::TensorRef V = graph.data(kv_shape, DataType::FP16);
    V->set_name("V");

    nntile::TensorRef A = nntile::TensorRef::adopt(gt::flash_sdpa_fwd_cudnn(K, Q, mask, V, "logsumexp"));
    A->set_name("A");

    REQUIRE(graph.num_data() == 6);
    REQUIRE(graph.num_ops() == 1);

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "FLASH_SDPA_FWD_CUDNN");
    REQUIRE(ops[0]->inputs().size() == 5);
    REQUIRE(ops[0]->outputs().size() == 2);
    REQUIRE(ops[0]->outputs()[1] == A);
    REQUIRE(A->shape() == kv_shape);
}

TEST_CASE("TensorGraph flash_sdpa_fwd_cudnn rejects null tensors",
    "[graph][tensor][cuda]")
{
    TensorGraph graph("test");
    std::vector<Index> kv_shape{32, 64, 2, 1, 1};
    std::vector<Index> mask_shape{64, 64};

    nntile::TensorRef K = graph.data(kv_shape, DataType::FP16);
    K->set_name("K");
    nntile::TensorRef Q = graph.data(kv_shape, DataType::FP16);
    Q->set_name("Q");
    nntile::TensorRef mask = graph.data(mask_shape, DataType::FP16);
    mask->set_name("mask");
    nntile::TensorRef V = graph.data(kv_shape, DataType::FP16);
    V->set_name("V");

    REQUIRE_THROWS_AS(
        gt::flash_sdpa_fwd_cudnn(nullptr, Q, mask, V, "logsumexp"),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::flash_sdpa_fwd_cudnn(K, nullptr, mask, V, "logsumexp"),
        std::invalid_argument);
    REQUIRE_THROWS_AS(gt::flash_sdpa_fwd_cudnn(K, Q, nullptr, V, "logsumexp"),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::flash_sdpa_fwd_cudnn(K, Q, mask, nullptr, "logsumexp"),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::CudaContextFixture,
    "TensorGraph flash_sdpa_fwd_cudnn tiled matches untiled",
    "[graph][tensor][cuda]")
{
    Index head_size = 32;
    Index n_seq = 64;
    Index n_batch = 2;
    Index kv_group_size = 1;
    Index n_head_kv = 1;

    std::vector<Index> K_shape = {
        head_size, n_seq, n_batch, kv_group_size, n_head_kv};
    std::vector<Index> mask_shape = {n_seq, n_seq};

    const Index kv_nelems = std::accumulate(
        K_shape.begin(), K_shape.end(), Index(1), std::multiplies<>());
    const Index mask_nelems = n_seq * n_seq;

    std::vector<float> K_data(kv_nelems);
    std::vector<float> Q_data(kv_nelems);
    std::vector<float> V_data(kv_nelems);
    std::vector<float> mask_data(mask_nelems);
    for (Index i = 0; i < kv_nelems; ++i)
    {
        K_data[i] = 0.1f * static_cast<float>((i % 10) - 5);
        Q_data[i] = 0.1f * static_cast<float>(((i + 1) % 10) - 5);
        V_data[i] = 0.1f * static_cast<float>(((i + 2) % 10) - 5);
    }
    for (Index i = 0; i < n_seq; ++i)
    {
        for (Index j = 0; j < n_seq; ++j)
        {
            mask_data[i * n_seq + j] =
                (j <= i) ? 0.0f : -std::numeric_limits<float>::infinity();
        }
    }

    auto run_graph = [&](bool tiled) -> std::vector<float>
    {
        TensorGraph graph(tiled ? "fwd_tiled" : "fwd_untiled");
        nntile::TensorRef K_node = graph.data(K_shape, DataType::FP16);
    K_node->set_name("K");
        nntile::TensorRef Q_node = graph.data(K_shape, DataType::FP16);
    Q_node->set_name("Q");
        nntile::TensorRef mask_node = graph.data(mask_shape, DataType::FP16);
    mask_node->set_name("mask");
        nntile::TensorRef V_node = graph.data(K_shape, DataType::FP16);
    V_node->set_name("V");

        nntile::TensorRef A_node = nntile::TensorRef::adopt(gt::flash_sdpa_fwd_cudnn(
            K_node, Q_node, mask_node, V_node, "logsumexp")
                           );
    A_node->set_name("A");

        if (tiled)
        {
            auto *head_axis = K_node->axis(0);
            auto *seq_axis = K_node->axis(1);
            for (auto *ag : graph.axis_groups())
            {
                if (ag == head_axis || ag == seq_axis)
                {
                    if (ag == head_axis)
                    {
                        ag->set_tiling(ag->extent);
                    }
                    else
                    {
                        ag->set_tiling((ag->extent + 1) / 2);
                    }
                }
                else
                {
                    ag->set_tiling(ag->extent);
                }
            }
        }

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data(K_node, K_data);
        runtime.bind_data(Q_node, Q_data);
        runtime.bind_data(mask_node, mask_data);
        runtime.bind_data(V_node, V_data);
        runtime.execute();
        runtime.wait();
        return runtime.get_output<float>(A_node);
    };

    auto untiled_A = run_graph(false);
    auto tiled_A = run_graph(true);

    REQUIRE(tiled_A.size() == untiled_A.size());
    for (size_t i = 0; i < tiled_A.size(); ++i)
    {
        REQUIRE(std::abs(tiled_A[i] - untiled_A[i]) < tolerance);
    }
}

#endif // NNTILE_USE_CUDA
