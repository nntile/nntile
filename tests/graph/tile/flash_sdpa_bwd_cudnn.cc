/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/flash_sdpa_bwd_cudnn.cc
 * Test TileGraph flash sdpa bwd cudnn vs nntile::tile (parity).
 *
 * @version 1.1.0
 * */

#include "nntile/defs.h"
#ifdef NNTILE_USE_CUDA
#include <catch2/catch_test_macros.hpp>
#include "context_fixture.hh"
#include "nntile/graph/tile/flash_sdpa_bwd_cudnn.hh"
#include "nntile/graph/tile.hh"
using namespace nntile;
using namespace nntile::graph;
namespace tg = nntile::graph::tile_graph;
TEST_CASE("TileGraph flash_sdpa_bwd_cudnn structure", "[graph][tile][cuda]")
{
    nntile::test::CudaContextFixture fx;
    (void)fx;
    std::vector<Index> kv{32, 64, 2, 1, 1};
    std::vector<Index> lse{64, 2, 1, 1};
    std::vector<Index> msk{64, 64};
    TileGraph g("flash_bwd");
    auto* K = g.data(kv, "K", DataType::FP16);
    auto* Q = g.data(kv, "Q", DataType::FP16);
    auto* V = g.data(kv, "V", DataType::FP16);
    auto* A = g.data(kv, "A", DataType::FP16);
    auto* dA = g.data(kv, "dA", DataType::FP16);
    auto* M = g.data(msk, "mask", DataType::FP16);
    auto* L = g.data(lse, "logsumexp", DataType::FP32);
    auto* dK = g.data(kv, "dK", DataType::FP16);
    auto* dQ = g.data(kv, "dQ", DataType::FP16);
    auto* dV = g.data(kv, "dV", DataType::FP16);
    for(auto* t : {K, Q, V, A, dA, M, L}) t->mark_input(true);
    dK->mark_output(true);
    dQ->mark_output(true);
    dV->mark_output(true);
    tg::flash_sdpa_bwd_cudnn(K, Q, V, A, dA, M, L, dK, dQ, dV);
    REQUIRE(g.num_ops() == 1);
    REQUIRE(g.ops()[0]->op_name() == "TILE_FLASH_SDPA_BWD_CUDNN");
}
#endif
