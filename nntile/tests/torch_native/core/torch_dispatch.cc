/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file nntile/tests/torch_native/core/torch_dispatch.cc
 * core::torch_*_out family vs matching CPU aten.
 */

#include "aten_ref.hh"
#include "context_fixture.hh"

#include <nntile/core/torch_dispatch.hh>
#include <nntile/core/torch_meta.hh>

#include <ATen/ops/add.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/transpose_copy.h>

#include <catch2/catch_test_macros.hpp>

#include <vector>

using namespace nntile;
using namespace nntile::core;
namespace tn = nntile::test::torch_native;

TEST_CASE_METHOD(
    nntile::test::ContextFixture,
    "core torch_unary Relu matches aten",
    "[torch_native][core]")
{
    const std::vector<Index> shape = {2, 4};
    const Index nelems = 8;
    Tile<fp32_t> in(shape), out(shape);
    std::vector<float> host(nelems), ref(nelems, 0.f);

    {
        auto loc = in.acquire(STARPU_W);
        for (Index i = 0; i < nelems; ++i)
        {
            host[static_cast<size_t>(i)] =
                static_cast<float>(i) - 3.5f;
            loc[i] = host[static_cast<size_t>(i)];
        }
        loc.release();
    }

    tn::with_cpu_aten(
        [&]
        {
            at::Tensor t_in = tn::blob_cpu(host.data(), shape);
            at::Tensor t_ref = tn::blob_cpu(ref.data(), shape);
            at::relu_out(t_ref, t_in);
        });

    const TorchTileMeta meta = make_contiguous_torch_meta(shape);
    torch_unary_out(
        -1,
        starpu::TorchKind::Relu,
        in,
        meta,
        out,
        meta);
    starpu_task_wait_for_all();

    std::vector<float> got(nelems);
    {
        auto loc = out.acquire(STARPU_R);
        for (Index i = 0; i < nelems; ++i)
        {
            got[static_cast<size_t>(i)] = tn::as_float(loc[i]);
        }
        loc.release();
    }
    tn::require_close(got, ref);
}

TEST_CASE_METHOD(
    nntile::test::ContextFixture,
    "core torch_binary Add matches aten::add_out",
    "[torch_native][core]")
{
    const std::vector<Index> shape = {2, 2};
    const Index nelems = 4;
    const Scalar alpha = 1.5f;
    Tile<fp32_t> a(shape), b(shape), out(shape);
    std::vector<float> a_h = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> b_h = {2.f, 0.5f, -1.f, 3.f};
    std::vector<float> ref(nelems, 0.f);

    {
        auto la = a.acquire(STARPU_W);
        auto lb = b.acquire(STARPU_W);
        for (Index i = 0; i < nelems; ++i)
        {
            la[i] = a_h[static_cast<size_t>(i)];
            lb[i] = b_h[static_cast<size_t>(i)];
        }
        la.release();
        lb.release();
    }

    tn::with_cpu_aten(
        [&]
        {
            at::Tensor ta = tn::blob_cpu(a_h.data(), shape);
            at::Tensor tb = tn::blob_cpu(b_h.data(), shape);
            at::Tensor tr = tn::blob_cpu(ref.data(), shape);
            at::add_out(tr, ta, tb, static_cast<double>(alpha));
        });

    const TorchTileMeta meta = make_contiguous_torch_meta(shape);
    starpu::TorchDispatchArgs extra;
    extra.scalars[0] = alpha;
    torch_binary_out(
        -1,
        starpu::TorchKind::Add,
        a,
        meta,
        b,
        meta,
        out,
        meta,
        extra);
    starpu_task_wait_for_all();

    std::vector<float> got(nelems);
    {
        auto loc = out.acquire(STARPU_R);
        for (Index i = 0; i < nelems; ++i)
        {
            got[static_cast<size_t>(i)] = tn::as_float(loc[i]);
        }
        loc.release();
    }
    tn::require_close(got, ref);
}

TEST_CASE_METHOD(
    nntile::test::ContextFixture,
    "core torch_binary Linear matches aten::linear",
    "[torch_native][core]")
{
    // input 3x4, weight 2x4 → out 3x2
    Tile<fp32_t> input({3, 4}), weight({2, 4}), out({3, 2});
    std::vector<float> in_h(12), w_h(8), ref(6, 0.f);

    {
        auto li = input.acquire(STARPU_W);
        auto lw = weight.acquire(STARPU_W);
        for (Index i = 0; i < 12; ++i)
        {
            in_h[static_cast<size_t>(i)] = static_cast<float>(i) * 0.1f;
            li[i] = in_h[static_cast<size_t>(i)];
        }
        for (Index i = 0; i < 8; ++i)
        {
            w_h[static_cast<size_t>(i)] = static_cast<float>(i) * 0.2f;
            lw[i] = w_h[static_cast<size_t>(i)];
        }
        li.release();
        lw.release();
    }

    tn::with_cpu_aten(
        [&]
        {
            at::Tensor tin = tn::blob_cpu(in_h.data(), {3, 4});
            at::Tensor tw = tn::blob_cpu(w_h.data(), {2, 4});
            at::Tensor tr = tn::blob_cpu(ref.data(), {3, 2});
            at::linear_out(tr, tin, tw, c10::nullopt);
        });

    const TorchTileMeta in_meta = make_contiguous_torch_meta({3, 4});
    const TorchTileMeta w_meta = make_contiguous_torch_meta({2, 4});
    const TorchTileMeta out_meta = make_contiguous_torch_meta({3, 2});
    torch_binary_out(
        -1,
        starpu::TorchKind::Linear,
        input,
        in_meta,
        weight,
        w_meta,
        out,
        out_meta);
    starpu_task_wait_for_all();

    std::vector<float> got(6);
    {
        auto loc = out.acquire(STARPU_R);
        for (Index i = 0; i < 6; ++i)
        {
            got[static_cast<size_t>(i)] = tn::as_float(loc[i]);
        }
        loc.release();
    }
    tn::require_close(got, ref);
}

TEST_CASE_METHOD(
    nntile::test::ContextFixture,
    "core torch_unary TransposeCopy matches aten",
    "[torch_native][core]")
{
    Tile<fp32_t> in({2, 3}), out({3, 2});
    std::vector<float> host = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    std::vector<float> ref(6, 0.f);

    {
        auto loc = in.acquire(STARPU_W);
        for (Index i = 0; i < 6; ++i)
        {
            loc[i] = host[static_cast<size_t>(i)];
        }
        loc.release();
    }

    tn::with_cpu_aten(
        [&]
        {
            at::Tensor t_in = tn::blob_cpu(host.data(), {2, 3});
            at::Tensor t_ref = tn::blob_cpu(ref.data(), {3, 2});
            at::transpose_copy_out(t_ref, t_in, 0, 1);
        });

    starpu::TorchDispatchArgs extra{};
    extra.iargs[0] = 0;
    extra.iargs[1] = 1;
    torch_unary_out(
        -1,
        starpu::TorchKind::TransposeCopy,
        in,
        make_contiguous_torch_meta({2, 3}),
        out,
        make_contiguous_torch_meta({3, 2}),
        extra);
    starpu_task_wait_for_all();

    std::vector<float> got(6);
    {
        auto loc = out.acquire(STARPU_R);
        for (Index i = 0; i < 6; ++i)
        {
            got[static_cast<size_t>(i)] = tn::as_float(loc[i]);
        }
        loc.release();
    }
    tn::require_close(got, ref);
}
