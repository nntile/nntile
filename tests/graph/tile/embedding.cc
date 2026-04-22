/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/embedding.cc
 * Test TileGraph embedding vs nntile::tile (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include "context_fixture.hh"
#include "nntile/graph/tile/embedding.hh"
#include "nntile/graph/tile.hh"
#include "nntile/tile/embedding.hh"
#include "nntile/tile/tile.hh"

using namespace nntile;
using namespace nntile::graph;
namespace tg = nntile::graph::tile_graph;

TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph embedding", "[graph][tile]")
{
    const Index m = 2, n = 2, k = 3, k0 = 0, ks = 3;
    const std::vector<Index> ih = {m, n}, vh = {ks, 5}, eh = {m, k, n};
    TileGraph g("g");
    auto* index = g.data(ih, "index", DataType::INT64);
    auto* vocab = g.data(vh, "vocab", DataType::FP32);
    auto* embed = g.data(eh, "embed", DataType::FP32);
    index->mark_input(true);
    vocab->mark_input(true);
    embed->mark_output(true);
    tg::embedding(m, n, k, k0, ks, index, vocab, embed);
    TileGraph::Runtime r(g);
    r.compile();
    std::vector<std::int64_t> iv(4);
    iv[0] = 0;
    iv[1] = 2;
    iv[2] = 4;
    iv[3] = 1;
    std::vector<float> voc(15);
    for(int i = 0; i < 15; ++i)
    {
        voc[static_cast<size_t>(i)] = static_cast<float>(i + 1);
    }
    std::vector<float> out(12, 0.f);
    r.bind_data("index", iv);
    r.bind_data("vocab", voc);
    r.bind_data("embed", out);
    r.execute();
    r.wait();
    const auto gout = r.get_output<float>("embed");
    nntile::tile::Tile<nntile::int64_t> I(ih);
    nntile::tile::Tile<fp32_t> V(vh), E(eh);
    {
        auto a = I.acquire(STARPU_W);
        a[0] = 0;
        a[1] = 2;
        a[2] = 4;
        a[3] = 1;
        a.release();
    }
    using Yv = typename fp32_t::repr_t;
    {
        auto b = V.acquire(STARPU_W);
        for(int j = 0; j < 15; ++j)
        {
            b[j] = Yv(voc[static_cast<size_t>(j)]);
        }
        b.release();
    }
    {
        auto c = E.acquire(STARPU_W);
        for(int t = 0; t < 12; ++t)
        {
            c[t] = Yv(0.0f);
        }
        c.release();
    }
    nntile::tile::embedding<fp32_t>(m, n, k, k0, ks, I, V, E);
    starpu_task_wait_for_all();
    std::vector<float> tr(12);
    {
        auto L = E.acquire(STARPU_R);
        for(int t = 0; t < 12; ++t)
        {
            tr[static_cast<size_t>(t)] = static_cast<float>(L[t]);
        }
        L.release();
    }
    for(int t = 0; t < 12; ++t)
    {
        REQUIRE(
            std::abs(
                gout[static_cast<size_t>(t)]
                - tr[static_cast<size_t>(t)])
            < 1e-3f);
    }
}
