#include <nntile/tensor/tensor_ref.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tile_graph/logsumexp.cc
 * TileGraph logsumexp vs nntile::core::logsumexp (small parity B).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <numeric>
#include "context_fixture.hh"
#include "test_frobenius.hh"
#include "nntile/tile/ops/logsumexp.hh"
#include "nntile/tile.hh"
#include "nntile/tile.hh"
#include "nntile/core/logsumexp.hh"
#include "nntile/core/tile.hh"
using namespace nntile;
using namespace nntile;
namespace tg = nntile::tile;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph logsumexp matches tile", "[graph][tile]")
{
    const std::vector<Index> sh_src = {3, 2, 2};
    const std::vector<Index> sh_dst = {3, 2};
    const Index n_src = 12, n_dst = 6;
    TileGraph g("g");
    auto *s = g.data(sh_src, "s", DataType::FP32);
    auto *d = g.data(sh_dst, "d", DataType::FP32);
    tg::logsumexp(s, d);
    Runtime runtime(g);
    runtime.compile();
    std::vector<float> sv(n_src);
    using Y = nntile::fp32_t::repr_t;
    for(Index i = 0; i < n_src; i += 2)
    {
        sv[static_cast<size_t>(i)] = static_cast<float>(Y(0.5) * (Y(i / 2) + Y(1)));
        sv[static_cast<size_t>(i + 1)] = static_cast<float>(std::exp((Y(i) + Y(1)) / Y(20)));
    }
    runtime.bind_data(s, sv);
    std::vector<float> dv(n_dst, 0.f);
    runtime.bind_data(d, dv);
    runtime.execute();
    runtime.wait();
    const std::vector<float> gout = runtime.get_output<float>(d);
    nntile::core::Tile<fp32_t> ts(sh_src), td(sh_dst);
    {
        using Y2 = typename nntile::fp32_t::repr_t;
        auto l1 = ts.acquire(STARPU_W);
        for(Index i = 0; i < n_src; ++i) { l1[i] = Y2(sv[static_cast<size_t>(i)]); }
        l1.release();
    }
    nntile::core::logsumexp<fp32_t>(-1, ts, td);
    starpu_task_wait_for_all();
    std::vector<float> tref(n_dst);
    {
        auto l2 = td.acquire(STARPU_R);
        for(Index i = 0; i < n_dst; ++i) { tref[static_cast<size_t>(i)] = static_cast<float>(l2[i]); }
        l2.release();
    }
    nntile::test::require_relative_element_error(gout, tref);
}
