/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/flash_sdpa_fwd_cudnn.cc
 * Test TileGraph flash sdpa fwd cudnn vs nntile::tile (parity).
 *
 * @version 1.1.0
 * */

#include "nntile/defs.h"
#ifdef NNTILE_USE_CUDA
#include <catch2/catch_test_macros.hpp>
#include <numeric>
#include "context_fixture.hh"
#include "nntile/graph/tile/flash_sdpa_fwd_cudnn.hh"
#include "nntile/graph/tile.hh"
using namespace nntile;
using namespace nntile::graph;
namespace tg = nntile::graph::tile_graph;
TEST_CASE("TileGraph flash_sdpa_fwd_cudnn structure", "[graph][tile][cuda]")
{
    nntile::test::CudaContextFixture fx;
    (void)fx;
    std::vector<Index> kv{32, 64, 2, 1, 1};
    std::vector<Index> lse{64, 2, 1, 1};
    std::vector<Index> mask{64, 64};
    TileGraph g("flash_fwd");
    auto* K = g.data(kv, "K", DataType::FP16);
    auto* Q = g.data(kv, "Q", DataType::FP16);
    auto* M = g.data(mask, "mask", DataType::FP16);
    auto* L = g.data(lse, "logsumexp", DataType::FP32);
    auto* V = g.data(kv, "V", DataType::FP16);
    auto* A = g.data(kv, "A", DataType::FP16);
    K->mark_input(true);
    Q->mark_input(true);
    M->mark_input(true);
    L->mark_input(true);
    V->mark_input(true);
    A->mark_output(true);
    tg::flash_sdpa_fwd_cudnn(K, Q, M, L, V, A);
    REQUIRE(g.num_ops() == 1);
    REQUIRE(g.ops()[0]->op_name() == "TILE_FLASH_SDPA_FWD_CUDNN");
}
#endif
