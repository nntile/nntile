#include <nntile/tensor/tensor_ref.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tile_graph/copy_intersection.cc
 * Test TileGraph copy intersection vs nntile::core (parity).
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include "context_fixture.hh"
#include "test_frobenius.hh"
#include "nntile/tile/ops/copy_intersection.hh"
#include "nntile/tile.hh"
#include "nntile/tile.hh"
#include "nntile/core/copy_intersection.hh"
#include "nntile/core/tile.hh"
#include <memory>
using namespace nntile; using namespace nntile; namespace tg = nntile::tile;
TEST_CASE_METHOD(nntile::test::ContextFixture, "TileGraph copy_intersection", "[graph][tile]")
{
    const std::vector<Index> sh = {2,2,3};
    const std::vector<Index> sc = {6};
    const Index n = 12;
    TileGraph g("g");
    auto *s = g.data(sh, "s", DataType::FP32);
    auto *d = g.data(sh, "d", DataType::FP32);
    auto *scra = g.data(sc, "scratch", DataType::INT64);
    tg::copy_intersection(s, {0,0,0}, d, {0,0,0}, scra);
    Runtime r(g);
    r.compile();
    std::vector<float> sv(n), dv(n, 0.f);
    for(Index i=0;i<n;++i) sv[static_cast<size_t>(i)]=static_cast<float>(i+1);
    std::vector<std::int64_t> scv(6, 0);
    r.bind_data(s, sv);
    r.bind_data(d, dv);
    r.bind_data(scra, scv);
    r.execute();
    r.wait();
    const auto gout = r.get_output<float>(d);
    nntile::core::Tile<fp32_t> S(sh), D(sh);
    nntile::core::Tile<nntile::int64_t> Sc(sc);
    using Y = typename fp32_t::repr_t;
    { auto a=S.acquire(STARPU_W), b=D.acquire(STARPU_W);
      for(Index i=0;i<n;++i) { a[i]=Y(sv[static_cast<size_t>(i)]); b[i]=Y(0);} a.release(); b.release(); }
    { auto L=Sc.acquire(STARPU_W); for(Index j=0;j<6;++j) L[j]=0; L.release(); }
    nntile::core::copy_intersection<fp32_t>(-1, S, {0, 0, 0}, D, {0, 0, 0}, Sc);
    starpu_task_wait_for_all();
    std::vector<float> tr(n);
    { auto L=D.acquire(STARPU_R);
      for(Index i=0;i<n;++i) tr[static_cast<size_t>(i)]=static_cast<float>(L[i]);
      L.release(); }
    nntile::test::require_relative_element_error(gout, tr);
}

TEST_CASE_METHOD(
    nntile::test::ContextFixture,
    "TileGraph copy_intersection INT64",
    "[graph][tile]")
{
    const std::vector<Index> sh = {4, 3};
    const std::vector<Index> sc = {4};
    const Index n = 12;
    TileGraph g("g_int64");
    auto *s = g.data(sh, "s", DataType::INT64);
    auto *d = g.data(sh, "d", DataType::INT64);
    auto *scra = g.data(sc, "scratch", DataType::INT64);
    // Partial overlap: copy src[1:, :] into dst[0:, :] starting at dst offset 0.
    tg::copy_intersection(s, {1, 0}, d, {0, 0}, scra);
    Runtime r(g);
    r.compile();
    std::vector<std::int64_t> sv(n), dv(n, 0);
    for (Index i = 0; i < n; ++i)
    {
        sv[static_cast<size_t>(i)] = static_cast<std::int64_t>(i + 10);
    }
    std::vector<std::int64_t> scv(4, 0);
    r.bind_data(s, sv);
    r.bind_data(d, dv);
    r.bind_data(scra, scv);
    r.execute();
    r.wait();
    const auto gout = r.get_output<std::int64_t>(d);

    nntile::core::Tile<nntile::int64_t> S(sh), D(sh);
    nntile::core::Tile<nntile::int64_t> Sc(sc);
    {
        auto a = S.acquire(STARPU_W), b = D.acquire(STARPU_W);
        for (Index i = 0; i < n; ++i)
        {
            a[i] = sv[static_cast<size_t>(i)];
            b[i] = 0;
        }
        a.release();
        b.release();
    }
    {
        auto L = Sc.acquire(STARPU_W);
        for (Index j = 0; j < 4; ++j)
        {
            L[j] = 0;
        }
        L.release();
    }
    nntile::core::copy_intersection<nntile::int64_t>(
        -1, S, {1, 0}, D, {0, 0}, Sc);
    starpu_task_wait_for_all();
    std::vector<std::int64_t> tr(n);
    {
        auto L = D.acquire(STARPU_R);
        for (Index i = 0; i < n; ++i)
        {
            tr[static_cast<size_t>(i)] =
                static_cast<std::int64_t>(L[i]);
        }
        L.release();
    }
    REQUIRE(gout.size() == tr.size());
    for (size_t i = 0; i < gout.size(); ++i)
    {
        REQUIRE(gout[i] == tr[i]);
    }
}

TEST_CASE_METHOD(
    nntile::test::ContextFixture,
    "TileGraph copy_intersection BOOL",
    "[graph][tile]")
{
    const std::vector<Index> sh = {4, 3};
    const std::vector<Index> sc = {4};
    const Index n = 12;
    TileGraph g("g_bool");
    auto *s = g.data(sh, "s", DataType::BOOL);
    auto *d = g.data(sh, "d", DataType::BOOL);
    auto *scra = g.data(sc, "scratch", DataType::INT64);
    // Partial overlap: copy src[1:, :] into dst[0:, :] starting at dst offset 0.
    tg::copy_intersection(s, {1, 0}, d, {0, 0}, scra);
    Runtime r(g);
    r.compile();
    // Avoid std::vector<bool> (no .data()); bind via contiguous bool buffer.
    std::unique_ptr<bool[]> sv(new bool[static_cast<size_t>(n)]);
    std::unique_ptr<bool[]> dv(new bool[static_cast<size_t>(n)]);
    for (Index i = 0; i < n; ++i)
    {
        sv[static_cast<size_t>(i)] = (i % 2) == 0;
        dv[static_cast<size_t>(i)] = false;
    }
    std::vector<std::int64_t> scv(4, 0);
    r.bind_data(s, sv.get(), static_cast<size_t>(n));
    r.bind_data(d, dv.get(), static_cast<size_t>(n));
    r.bind_data(scra, scv);
    r.execute();
    r.wait();
    const auto gout = r.get_output<bool>(d);

    nntile::core::Tile<nntile::bool_t> S(sh), D(sh);
    nntile::core::Tile<nntile::int64_t> Sc(sc);
    {
        auto a = S.acquire(STARPU_W), b = D.acquire(STARPU_W);
        for (Index i = 0; i < n; ++i)
        {
            a[i] = nntile::bool_t(sv[static_cast<size_t>(i)]);
            b[i] = nntile::bool_t(false);
        }
        a.release();
        b.release();
    }
    {
        auto L = Sc.acquire(STARPU_W);
        for (Index j = 0; j < 4; ++j)
        {
            L[j] = 0;
        }
        L.release();
    }
    nntile::core::copy_intersection<nntile::bool_t>(
        -1, S, {1, 0}, D, {0, 0}, Sc);
    starpu_task_wait_for_all();
    std::vector<bool> tr(static_cast<size_t>(n));
    {
        auto L = D.acquire(STARPU_R);
        for (Index i = 0; i < n; ++i)
        {
            tr[static_cast<size_t>(i)] = static_cast<bool>(L[i]);
        }
        L.release();
    }
    REQUIRE(gout.size() == tr.size());
    for (size_t i = 0; i < gout.size(); ++i)
    {
        REQUIRE(gout[i] == tr[i]);
    }
}
