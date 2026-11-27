#include <nntile/tensor/tensor_ref.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tile_graph/mask_scalar.cc
 * Test TileGraph mask scalar vs nntile::core (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include "context_fixture.hh"
#include "test_frobenius.hh"
#include "nntile/tile/ops/mask_scalar.hh"
#include "nntile/tile.hh"
#include "nntile/tile.hh"
#include "nntile/core/mask_scalar.hh"
#include "nntile/core/tile.hh"
#include <array>
using namespace nntile;
using namespace nntile;
namespace tg = nntile::tile;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph mask_scalar", "[graph][tile]")
{
    const std::vector<Index> sh = {3, 2};
    const Index n = 6;
    const Scalar val = -9.0;
    const Index batch = 0;
    TileGraph g("g");
    auto *mask = g.data(sh, "mask", DataType::BOOL);
    auto *a = g.data(sh, "a", DataType::FP32);
    tg::mask_scalar(mask, val, a, batch);
    Runtime r(g);
    r.compile();
    std::array<bool, 6> mb{};
    for(Index i = 0; i < n; ++i) { mb[static_cast<size_t>(i)] = (static_cast<int>(i) % 3) != 0; }
    std::vector<float> av(n);
    for(Index i = 0; i < n; ++i) { av[static_cast<size_t>(i)] = static_cast<float>(i + 1); }
    r.bind_data(mask, mb.data(), static_cast<size_t>(n));
    r.bind_data(a, av);
    r.execute();
    r.wait();
    const std::vector<float> gout = r.get_output<float>(a);
    nntile::core::Tile<bool_t> Tm(sh);
    nntile::core::Tile<fp32_t> Ta(sh);
    using Y = typename fp32_t::repr_t;
    { auto mloc = Tm.acquire(STARPU_W);
      for(Index i = 0; i < n; ++i) { mloc[i] = nntile::bool_t(mb[static_cast<size_t>(i)]); }
      mloc.release(); }
    { auto aloc = Ta.acquire(STARPU_W);
      for(Index i = 0; i < n; ++i) { aloc[i] = Y(static_cast<float>(i + 1)); }
      aloc.release(); }
    nntile::core::mask_scalar<fp32_t>(-1, Tm, val, Ta, batch);
    starpu_task_wait_for_all();
    std::vector<float> tref(n);
    { auto L = Ta.acquire(STARPU_R);
      for(Index i = 0; i < n; ++i) { tref[static_cast<size_t>(i)] = static_cast<float>(L[i]); }
      L.release(); }
    nntile::test::require_relative_element_error(gout, tref);
}
